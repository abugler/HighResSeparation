import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from nussl import STFTParams
from nussl.ml.networks.modules import STFT, AmplitudeToDB

from .constants import ALIGN_CORNERS
from timm.models.hrnet import _BN_MOMENTUM
from .blocks import HRNetV2Upsample, HRNetV2


class HRNet(nn.Module):
    """
    Wrapper for the High-Resolution Network in `timm`.

    (1) https://arxiv.org/pdf/1908.07919.pdf

    Includes processing for resampling and computing STFTs.

    Things to think about:
    + HRNet turns the input image into a stem, which has 1/4 of the resolution, but 64 channels.
      This is likely fine, but STFTs, especially for audio with 16k sampling rate, can be significantly
      smaller than images.
    + HRNet is designed for images. Could it be made to work on waveform representations? Maybe,
      but that is outside the scope of this work.
    
    Notes about classification vs. Segmentation:
    A Variation of the HRNetV2 head will be used for separation, as it is used for semantic segmentation in (1).

    The downsampling head will be used for classification.
    Parameters
    -----------
    num_classes : int
        Number of classes for either separation or tagging
    pretrained : bool
        If true, use imagenet weights. Default: False
    width : int
        Number of channels in first stream
    stft_params : STFTParams
        nussl STFTParams for STFT
    separate : bool
        If True, perform separation. Else, do classification.
    audio_channels : int
        Number of audio channels.
    """
    def __init__(self,
                 num_classes : int,
                 pretrained : bool=False,
                 width : int = 18,
                 stft_params : STFTParams=None,
                 head : str='classification',
                 audio_channels : int = 1):
        super().__init__()

        self.num_classes = num_classes
        self.audio_channels = audio_channels
        self.hrnet = timm.create_model(
            f'hrnet_w{width}',
            pretrained=pretrained,
            head=head,
            in_chans=audio_channels,
            num_classes=num_classes)
        # This ensures that all height dimensions will match if window_length is a power of 2
        self.hrnet.conv1.padding = (0, 1)

        if stft_params is None:
            stft_params = STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
        self.stft_params = stft_params
        self.stft = STFT(stft_params.window_length,
                         hop_length=stft_params.hop_length,
                         window_type=stft_params.window_length)
        self.amplitude_to_db = AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(audio_channels, momentum=_BN_MOMENTUM)
        if head == 'separation':
            self.hrnet.last_layer = HRNetV2Upsample(
                width,
                num_classes,
                stft_params.window_length // 2 + 1,
                audio_channels
            )
        elif head == 'classification':
            self.sigmoid = nn.Sigmoid()
        elif head == 'segmentation':
            self.hrnet.last_layer = HRNetV2(
                width,
                num_classes
            )
        else:
            raise ValueError("Invalid head!")


    def _separation_forward(self, spec):
        """
        `timm` doesn't include semantic segmentation support. We have to use a manual forward
        that uses the components of the HRNet.

        Upsampling is copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py#L455
        """
        # BN on spec
        spec = self.spec_bn(spec)
        # Make stem
        x = self.hrnet.conv1(spec)
        x = self.hrnet.bn1(x)
        x = self.hrnet.act1(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        stem = self.hrnet.act2(x)
        # stem: (batch, 64, spec.freqs / 4, self.frames / 4)

        # Stages
        x = self.hrnet.stages(stem)
        # x: List of four tensors, each with decreasing feature map size and increasing channels

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x1, x2, x3], 1)
        # x: (batch, 15 * width, spec.freqs / 4, spec.frames / 4)

        # Compute Masks
        out = self.hrnet.last_layer(x, spec)
        # x: (batch, num_classes, spec.freq, spec.frames)

        return out

    def preprocess(self, waveform):
        stft = self.stft(waveform, direction='transform')
        # stft : (batch, num_frames, num_freqs * 2, channels)
        magnitude, phase = torch.split(stft, stft.shape[2] // 2, dim=2)
        # magnitude, phase : (batch, num_frames, num_freqs, channels)
        data = self.amplitude_to_db(magnitude).permute(0, 3, 2, 1)
        return data, magnitude, phase

    def masks_and_audio(self, magnitude, phase, out):
        batch, num_frames = out.shape[0], out.shape[-1]
        num_freqs = self.stft_params.window_length // 2 + 1
        out = torch.reshape(out, (batch, self.num_classes, self.audio_channels,
                            num_freqs, num_frames))
        masks = out.permute(0, 4, 3, 2, 1)
        # masks: (batch, num_frames, num_freqs, audio_channels, num_classes)
        estimates = magnitude.unsqueeze(-1) * masks
        # estimates: (batch, num_frames, num_freqs, audio_channels, num_classes)
        _phase = torch.stack([phase] * self.num_classes, dim=4)
        estimates_with_phase = torch.cat([estimates, _phase], dim=2)
        audio = self.stft(estimates_with_phase, direction='inverse')
        # audio: (batch, audio_channels, num_samples, num_classes)
        return masks, estimates, audio

    def pad_frames(self, data):
        num_frames = data.shape[-1]
        pad_frames = (num_frames // 32 + 1) * 32 - num_frames
        data = F.pad(data, (0, pad_frames))
        return data, pad_frames

    def unpad_frames(self, data, pad_frames):
        if pad_frames > 0:
            return data[..., :-pad_frames]
        else:
            return data

    def forward(self, waveform):
        
        # audio: (batch, num_channels, num_samples)
        data, magnitude, phase = self.preprocess(waveform)
        
        data, pad_frames = self.pad_frames(data)
        # TODO: check if audio must be padded.
        # data: (batch, num_channels, num_freqs, num_frames)
        if self.hrnet.head == 'classification':
            data = self.spec_bn(data)
            out = {
                'tags': self.sigmoid(self.hrnet(data))
            }
        if self.hrnet.head == 'separation':
            out = self._separation_forward(data)
            out = self.unpad_frames(out, pad_frames)
            masks, estimates, audio = self.masks_and_audio(magnitude, phase, out)
            out = {
                'masks': masks,
                'estimates': estimates,
                'audio': audio
            }
        return out
    
    def save(self, location, **kwargs):
        """
        Overriding SeparationModel's save
        """
        torch.save(self, location)
        return location
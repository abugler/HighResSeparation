import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from nussl import STFTParams
from nussl.ml.networks.modules import STFT, AmplitudeToDB

from timm.models.hrnet import HighResolutionNet, _BN_MOMENTUM, cfg_cls as HRNetConfigurations
from timm.models.resnet import Bottleneck
from .blocks import HRNetV2Skip, HRNetV2, StemEncoder, SpecEncoder

ALIGN_CORNERS = None

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
    closure_key : str
        Closure key for the configuration in `timm.models.hrnet.cls_cfg`.
        Initial configurations may be found here:
        https://github.com/rwightman/pytorch-image-models/blob/54a6cca27a9a3e092a07457f5d56709da56e3cf5/timm/models/hrnet.py#L61
        We are currently experimenting with `hrnet_w18_small_v2`.
    num_classes : int
        Number of classes for either separation or tagging
    pretrained : bool
        If true, use imagenet weights. Default: False
    stft_params : STFTParams
        nussl STFTParams for STFT
    separate : bool
        If True, perform separation. Else, do classification.
    audio_channels : int
        Number of audio channels.
    stem : bool
        If True, downsamples input to a "stem", which is a high channel feature map
        with 1/4 of the resolutiion. Otherwise, the HRNet operates on a spectrogram,
        with adjusted dimensions such that the height is a power of 2.
    skip : bool
        Only for separation head.
        If True, then the original magnitude spectrogram is appended to the
        feature map before the final convolution.
    """
    def __init__(self,
                 closure_key : str,
                 num_classes : int,
                 pretrained : bool=False,
                 stft_params : STFTParams=None,
                 head : str = 'classification',
                 stem : bool = False, 
                 audio_channels : int = 1,
                 skip : bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.audio_channels = audio_channels
        self.skip = skip

        # Load and edit configuration
        hrnet_config = HRNetConfigurations[closure_key]
        width = hrnet_config['STAGE2']['NUM_CHANNELS'][0]
        
        if stem:
            self.encoder = StemEncoder(audio_channels)
        else:
            hrnet_config['STEM_WIDTH'] = width
            hrnet_config['STAGE1']['NUM_CHANNELS'] = (width,)
            self.encoder = SpecEncoder(audio_channels, width)

        self.hrnet = HighResolutionNet(
            hrnet_config,
            in_chans=audio_channels,
            num_classes=num_classes,
            head=head
        )
        # Remove stem encoder. We use a custom one for spectrograms.
        self.clear_stem_encoder()
        if not stem:
            self.hrnet.layer1 = self.hrnet._make_layer(
                Bottleneck, width, width,
                hrnet_config['STAGE1']['NUM_BLOCKS'][0]
            )

        if stft_params is None:
            stft_params = STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
        self.stft_params = stft_params
        self.stft = STFT(stft_params.window_length,
                         hop_length=stft_params.hop_length,
                         window_type=stft_params.window_type)
        self.amplitude_to_db = AmplitudeToDB()

        if self.hrnet.head == 'separation':
            if skip:
                self.hrnet.final_layer = HRNetV2Skip(
                    width,
                    num_classes,
                    audio_channels,
                    stem
                )
            else:
                self.hrnet.final_layer = HRNetV2(width, num_classes, stem)
        elif head == 'classification':
            self.sigmoid = nn.Sigmoid()
        else:
            raise ValueError("Invalid head!")


    def _forward(self, spec):
        """
        `timm` doesn't include semantic segmentation support. We have to use a manual forward
        that uses the components of the HRNet.

        Upsampling is copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py#L455
        """
        # Encode Spectrogram
        x = self.encoder(spec)
        # if stem:
            # x: (batch, 64, spec.freqs / 4, self.frames / 4)
        # else:
            # x: (batch, width, spec.freqs - 1, self.frames)

        # Stages
        hr_out = self.hrnet.stages(x)
        # x: List of four tensors, each with decreasing feature map size and increasing channels
        if self.hrnet.head == 'separation':
            # Upsampling
            x0_h, x0_w = hr_out[0].size(2), hr_out[0].size(3)
            x1 = F.interpolate(hr_out[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(hr_out[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x3 = F.interpolate(hr_out[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x = torch.cat([hr_out[0], x1, x2, x3], 1)
            # if stem:
                # x: (batch, 15 * width, spec.freqs / 4, self.frames / 4)
            # else:
                # x: (batch, 15 * width, spec.freqs - 1, self.frames)
        else:
            x = self.hrnet.incre_modules[0](hr_out[0])
            for i, down in enumerate(self.hrnet.downsamp_modules):
                x = self.hrnet.incre_modules[i + 1](hr_out[i + 1]) + down(x)
            # if stem:
                # x: (batch, 1024, spec.freqs / 4, self.frames / 4)
            # else:
                # x: (batch, 1024, spec.freqs - 1, self.frames)

        # Compute Masks
        if self.skip:
            out = self.hrnet.final_layer(x, spec)
        else:
            out = self.hrnet.final_layer(x)
        # x: (batch, num_classes, spec.freq, spec.frames)

        return out

    def clear_stem_encoder(self):
        del self.hrnet.conv1
        del self.hrnet.bn1
        del self.hrnet.act1
        del self.hrnet.conv2
        del self.hrnet.bn2
        del self.hrnet.act2
    

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
        out = self._forward(data)
        if self.hrnet.head == 'classification':
            out = self.hrnet.global_pool(out)
            if self.hrnet.drop_rate > 0.:
                out = F.dropout(out, p=self.hrnet.drop_rate, training=self.training)
            out = self.sigmoid(self.hrnet.classifier(out))
            out = {
                'tags': out
            }
        if self.hrnet.head == 'separation':
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
        torch.save(self.state_dict(), location)
        return location
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.hrnet import _BN_MOMENTUM
from nussl import STFTParams
from nussl.ml.networks.modules import STFT

ALIGN_CORNERS = None

class HRNetV2LastLayer(nn.Module):
    """
    Overrides last layer in HrNet for separation.

    Includes two blocks:
    + Conversion back into the original resolution.
    + Computation of mask.

    Copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py
    """
    def __init__(self, 
                 last_inp_channels : int,
                 num_classes : int,
                 audio_channels : int):
        super().__init__()
        self.from_stem = nn.Sequential(
            nn.ConvTranspose2d(last_inp_channels, last_inp_channels,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(last_inp_channels, momentum=_BN_MOMENTUM),
            nn.ConvTranspose2d(last_inp_channels, last_inp_channels,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(last_inp_channels, momentum=_BN_MOMENTUM),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=_BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes * audio_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.from_stem(x)
        return self.layers(x)

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
    The HRNetV2 head will be used for separation, as it is used for semantic segmentation in (1).

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
                 separate : bool=True,
                 audio_channels : int = 1):
        super().__init__()

        head = 'classification' if separate else 'separation'
        self.hrnet = timm.create_model(
            f'hrnet_w{width}',
            pretrained=pretrained,
            head=head,
            in_chans=audio_channels,
            num_classes=num_classes)
        if head == 'separation':
            self.hrnet.last_layer = HRNetV2LastLayer(
                15 * width,
                num_classes
            )
        if stft_params is None:
            stft_params = STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
        self.stft_params = stft_params
        self.stft = STFT(stft_params.window_length,
                         hop_length=stft_params.hop_length,
                         window_type=stft_params.window_length)

    def _separation_forward(self, spec):
        """
        `timm` doesn't include semantic segmentation support. We have to use a manual forward
        that uses the components of the HRNet.

        Upsampling is copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py#L455
        """
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
        # x: (batch, 15 * width, spec.freqs / 4, spec.freqs / 4)

        # Compute Masks
        out = self.hrnet.last_layer(x)
        # x: (batch, num_classes, spec.freq, spec.freqs)

        return out

    def masks_and_audio(self, audio, spec, out):
        num_frames = spec.shape[-1]
        out = torch.reshape()

    def forward(self, audio):
        # audio: (batch, num_channels, num_samples)
        spec = self.stft(audio, direction='transform')
        # TODO: check if audio must be padded.
        # spec: (batch, num_channels, num_freqs, num_frames)
        if self.hrnet.head == 'classification':
            out = self.hrnet(spec)
            # out: (batch, num_classes)
        if self.hrnet.head == 'separation':
            out = self._separation_forward(spec)
        return out

        




import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.hrnet import _BN_MOMENTUM

###############
# HRNet Heads #
###############

class HRNetV2Skip(nn.Module):
    """
    Overrides last layer in HrNet for separation.

    Similar to the HRNetV2 head used for semantic segmentation, with some differences:
    + An upsampling layer is added if stem is True, to put the stem in the same dimensions as the spectrogram.
    + To conserve memory, the stem with 15 * C channels is reduces to C channels.

    Copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py
    """
    def __init__(self, 
                 width : int,
                 num_classes : int,
                 audio_channels : int,
                 stem : bool
                 ):
        super().__init__()
        self.stem = stem
        width_multi = 15
        last_inp_channels = width * width_multi
        self.spec_bn = nn.BatchNorm2d(audio_channels)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=width,
                kernel_size=1,
                stride=1,
                padding=1),
            nn.BatchNorm2d(width, momentum=_BN_MOMENTUM),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=width + audio_channels,
                out_channels=width,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=width,
                out_channels=num_classes * audio_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, spec):

        x = self.layer1(x)
        if self.stem:
            upsample = nn.Upsample(size=spec.shape[-2:], mode='bilinear')
            x = upsample(x)
        spec = self.spec_bn(spec)
        x = torch.cat([x, spec], dim=1)
        x = self.layer2(x)
        return x

class HRNetV2(nn.Module):
    """
    The HRNetV2 head, with additional padding on the input, and a Sigmoid activation.
    """
    def __init__(self,
                 width : int,
                 num_classes : int,
                 stem : bool):
        super().__init__()
        last_inp_channels = 15 * width
        self.stem = stem
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=width,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(width, momentum=_BN_MOMENTUM),
            nn.ReLU()
        )
        if stem:
            self.upsample = nn.Upsample(scale_factor=4.0)
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=width,
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        if self.stem:
            x = self.upsample(x)
        x = F.pad(x, (0, 0, 0, 1))
        x = self.layer2(x)
        return x

############
# Encoders #
############

class StemEncoder(nn.Module):
    """
    Encodes a spectrogram into a stem.

    This is similar to the encoder the HRNet Segmentation Network uses.

    The only difference is the change of padding, to account for n_freqs being
    a power of 2 + 1.
    """
    def __init__(self,
                 audio_channels : int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(audio_channels, 64, kernel_size=3, stride=2, padding=(0, 1), bias=False),
            nn.BatchNorm2d(64, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class SpecEncoder(nn.Module):
    """
    Encodes a spectrogram into a feature map with a height with a power of two.
    """
    def __init__(self,
                 audio_channels : int,
                 width : int):
        super().__init__()
        self.conv = nn.Conv2d(audio_channels, width, kernel_size=3, padding=(0, 1))
        self.bn = nn.BatchNorm2d(width)
    
    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 1))
        x = self.conv(x)
        x = self.bn(x)
        return x

########################
# Normalization Blocks #
########################

class WaveformNorm(nn.Module):
    def __init__(self):
        super().__init__()

class PeakNorm(WaveformNorm):
    """
    Performs peak norm.

    direction == 'forward' normalizes audio.
    direction == 'backward' unnormalizes audio.

    NOTE: `direction == 'backward` is not thread-safe. Use with care.
    """
    def __init__(self, max_val=.9, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.max_val = .9
        self.peak = None

    def forward(self, waveform, direction='forward'):
        assert direction in ['forward', 'backward']
        with torch.no_grad():
            if direction == 'forward':
                peak = torch.max(torch.abs(waveform)) + epsilon
                waveform = waveform / peak
                waveform = waveform * self.max_val
                self.peak = peak
            else:
                peak = self.peak
                waveform = waveform / self.max_val
                waveform =  waveform * peak
        return waveform

class WhiteningNorm(WaveformNorm):
    """
    Performs "unit" norm, similar to Demucs.

    direction == 'forward' normalizes audio.
    direction == 'backward' unnormalizes audio.

    NOTE: `direction == 'backward` is not thread-safe. Use with care.
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon  = epsilon
        self.mean = None
        self.std = None

    def forward(self, waveform, direction='forward'):
        assert direction in ['forward', 'backward']
        with torch.no_grad():
            if direction == 'forward':
                mean, std = waveform.mean(dim=[1, 2]) + self.epsilon, waveform.std(dim=[1, 2])
                waveform = (waveform - std.unsqueeze(-1).unsqueeze(-1)) / mean.unsqueeze(-1).unsqueeze(-1)
                self.mean, self.std = mean, std
            else:
                mean, std = self.mean, self.std
                mean, std = mean.unsqueeze(-1).unsqueeze(-1), std.unsqueeze(-1).unsqueeze(-1)
                waveform = waveform * mean + std
        return waveform
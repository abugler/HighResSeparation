import torch
import torch.nn as nn
from timm.models.hrnet import _BN_MOMENTUM

class HRNetV2Upsample(nn.Module):
    """
    Overrides last layer in HrNet for separation.

    Similar to the HRNetV2 head used for semantic segmentation, with some differences:
    + An upsampling layer is added, to put the stem in the same dimensions as the spectrogram.
    + To conserve memory, the stem with 15 * C channels is reduces to C channels.
    

    Copied from https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py
    """
    def __init__(self, 
                 width : int,
                 num_classes : int,
                 num_freqs : int,
                 audio_channels : int,
                 ):
        super().__init__()
        width_multi = 15
        last_inp_channels = width * width_multi
        self.stem_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=width,
                kernel_size=1,
                stride=1,
                padding=1),
            nn.BatchNorm2d(width, momentum=_BN_MOMENTUM),
            nn.ReLU()
        )
        self.spec_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=width + audio_channels,
                out_channels=num_classes * audio_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, spec):
        
        x = self.stem_layer(x)
        upsample = nn.Upsample(size=spec.shape[-2:], mode='bilinear')
        x = upsample(x)
        x = torch.cat([x, spec], dim=1)
        x = self.spec_layer(x)
        return x

class HRNetV2(nn.Module):
    def __init__(self,
                 width : int,
                 num_classes : int):
        super().__init__()
        last_inp_channels = 15 * width
        self.last_layer = nn.Sequential(
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
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding=1)
        )

    def forward(self, x):
        x = self.last_layer(x)
        return x
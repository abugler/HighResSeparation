import torch.nn as nn
import timm
from nussl import STFTParams
from nussl.ml.networks.modules import STFT

class HRNet(nn.Module):
    """
    Wrapper for the High-Resolution Network in `timm`.

    1. https://arxiv.org/pdf/1908.07919.pdf

    Includes processing for resampling and computing STFTs.

    Things to think about:
    + HRNet turns the input image into a stem, which has 1/4 of the resolution, but 64 channels.
      This is likely fine, but STFTs, especially for audio with 16k sampling rate, can be significantly
      smaller than images.
    + HRNet is designed for images. Could it be made to work on waveform representations? Maybe,
      but that is outside the scope of this work.
       
    """
    def __init__(self,
                 pretrained : bool=False,
                 width : int=32,
                 stft_params : STFTParams=None,
                 separate : bool=True):
        super().__init__()
        self.hrnet = timm.create_model(
            f'hrnet_w{width}',
            pretrained=pretrained)
        if stft_params is None:
          stft_params = STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
        self.stft_params = stft_params
        self.stft = STFT(stft_params.window_length,
                         hop_length=stft_params.hop_length,
                         window_type=stft_params.window_length)

    def forward(self, audio):
      x = self.stft(audio)
      
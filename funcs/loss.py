import torch
import torch.nn.functional as F
from nussl.ml.train.loss import SISDRLoss
import torch.nn as nn

ALIGN_CORNERS = None

class ReconstructionLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.L1Loss(**kwargs)

    def forward(self, output, batch, stft_function):
        with torch.no_grad():
            src_stft = stft_function(batch['source_audio'])
            src_magnitude, _ = torch.split(src_stft, src_stft.shape[2] // 2, dim=2)
        loss = self.loss_func(output['estimates'], src_magnitude)
        return loss


def classification_loss(output, batch):
    loss = F.binary_cross_entropy(
        output['tags'],
        batch['tags']
    )
    return loss

class SISDR(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = SISDRLoss()
    
    def forward(self, output, batch):
        sisdr = - self.loss(batch['source_audio'], output['audio'])
        return sisdr


class CrossEntropy(nn.Module):
    """
    Taken from 
    https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/0bbb2880446ddff2d78f8dd7e8c4c610151d5a51/lib/core/criterion.py#L14

    Unnecessary functionalities are removed.
    """
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

import torch
import torch.nn.functional as F
from nussl.ml.train.loss import SISDRLoss
import torch.nn as nn

ALIGN_CORNERS = None

class ReconstructionLoss(nn.Module):
    def __init__(self, multiclass : bool = True, **kwargs):
        super().__init__()
        self.multiclass = multiclass
        if multiclass:
            self.loss_func = nn.L1Loss(**kwargs)
        else:
            self.loss_func = nn.CrossEntropyLoss(**kwargs)

    def forward(self, output, batch, stft_function):
        with torch.no_grad():
            src_stft = stft_function(batch['source_audio'])
            gt, _ = torch.split(src_stft, src_stft.shape[2] // 2, dim=2)
        if self.multiclass:
            # This makes the ideal binary masks
            _, gt_max_idx = gt.max(dim=-1)
            for jdx in range(gt.shape[-1]):
                idx = gt_max_idx == jdx
                gt[..., jdx][idx] = 1
                gt[..., jdx][~idx] = 0
            loss = self.loss_func(output['masks'], gt)
        else:
            loss = self.loss_func(output['estimates'], gt)
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

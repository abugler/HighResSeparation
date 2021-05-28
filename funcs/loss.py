import torch
import torch.nn.functional as F

def reconstruction_loss(output, batch, stft_function):
    with torch.no_grad():
        src_stft = stft_function(batch['source_audio'])
        src_magnitude, _ = torch.split(src_stft, stft.shape[2] // 2, dim=2)
    loss = F.l1_loss(output['estimates'], src_magnitude)
    return loss

def classification_loss(output, batch):
    loss = F.binary_cross_entropy(
        output['tags'],
        batch['tags']
    )
    return loss
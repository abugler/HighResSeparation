import torch
import torchaudio

def resample_batch(batch : dict[torch.Tensor],
                   resampler : torchaudio.transforms.Resample):
    with torch.no_grad():
        batch['mix_audio'] = resampler(batch['mix_audio'])
        if 'source_audio' in batch:
            batch['source_audio'] = resampler(
                batch['source_audio'].permute(0, 1, 3, 2).contiguous()
            ).permute(0, 1, 3, 2)

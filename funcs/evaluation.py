from pathlib import Path
from tqdm import tqdm
import json
import yaml
import glob
import nussl
import torch
import torchaudio
import logging
import numpy as np
from nussl.evaluation import BSSEvalScale, BSSEvalV4
import sklearn.metrics as sk_metrics

class Evaluator(BSSEvalScale, BSSEvalV4):
    """
    Evaluator does the following:
     + Runs BSSEvalScale and BSSEvalV4
    """
    def __init__(self, *args, bss_evalv4=True, **kwargs):
        self.bss_evalv4 = bss_evalv4
        super().__init__(*args, **kwargs)

    def evaluate_helper(self,
                        references,
                        estimates,
                        compute_sir_sar=True,
                        **kwargs):

        scale_scores = BSSEvalScale.evaluate_helper(
            self,
            references,
            estimates,
            compute_sir_sar=compute_sir_sar)
        if self.bss_evalv4:
            v4_scores = BSSEvalV4.evaluate_helper(
                self,
                references,
                estimates,
                **kwargs
            )
        else:
            v4_scores = [{}] * len(scale_scores)
        # This line requires python 3.9+
        scores = [scale | v4 for scale, v4 in zip(scale_scores, v4_scores)]
        return scores

def tensor_to_audio_signals(tensor : torch.Tensor,
                            sample_rate : int):
    signals = []
    tensor = tensor.squeeze(0)
    for src_idx in range(tensor.shape[-1]):
        audio_data = tensor[..., src_idx].cpu().numpy()
        signal = nussl.AudioSignal(
            audio_data_array=audio_data,
            sample_rate=sample_rate
        )
        signals.append(signal)
    return signals

def evaluate(
    model : torch.nn.Module,
    evaluator : nussl.evaluation.EvaluationBase,
    resampler : torchaudio.transforms.Resample,
    test_data : torch.utils.data.Dataset,
    output_folder : str = '.',
    device : str = 'cuda:0',
    num_workers : int = 0
):
    # lazy load
    from . import resample_batch

    output_folder = Path('.').absolute() / 'results'
    output_folder.mkdir(parents=True, exist_ok=True)
    order_of_mag = int(np.log(len(test_data))) + 1
    dataloader = torch.utils.data.DataLoader(test_data, num_workers=num_workers)

    pbar = tqdm(total=len(test_data))

    for i, item in enumerate(dataloader):
        with torch.no_grad():
            resample_batch(item, resampler)
            output = model(item['mix_audio'].to(device))
            estimates = tensor_to_audio_signals(
                output['audio'],
                resampler.new_freq
            )

        sources = tensor_to_audio_signals(
            item['source_audio'],
            resampler.new_freq
        )

        evaluator.estimated_sources_list = estimates
        evaluator.true_sources_list = list(sources)
        evaluator.source_labels = list(item['metadata']['labels'][0])

        file_name = f'{str(i).zfill(order_of_mag)}.json'
        pbar.set_description(file_name)
        try:
            scores = evaluator.evaluate()
        except:
            continue
        with open(output_folder / file_name, 'w') as f:
            json.dump(scores, f, indent=4)
        pbar.update()
    
    json_files = glob.glob(f"{output_folder}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)
    report_card = nussl.evaluation.report_card(df)

    logging.info(report_card)

    with open('report_card.txt', 'w') as f:
        f.write(report_card)

def evaluate_tagger(
    test_data : torch.utils.data.Dataset,
    model : torch.nn.Module, 
    device : str, 
    num_workers : int):
    """
    Records ROC-AUC score
    """
    output_folder = Path('.').absolute() / 'results'
    output_folder.mkdir(parents=True, exist_ok=True)

    classes = list(map(lambda x: x[0],
                       sorted(test_data.class_map.items(),
                       key=lambda x: x[1])))
    test_dataloader = torch.utils.data.DataLoader(test_data, num_workers=num_workers)
    pbar = tqdm(total=len(test_data))
    actual, predicted = [], []
    for i, item in enumerate(test_dataloader):
        output_file = output_folder / f'{i}.json'
        with torch.no_grad():
            model.eval()
            out = model(item['mix_audio'].to(device))
            out["tags"] = out["tags"].flatten().cpu().numpy()
        item["tags"] = item["tags"].flatten().int().cpu().numpy()
        to_yaml = {
            "actual": item["tags"].tolist(),
            "raw_output": out["tags"].tolist()
        }
        actual.append(item["tags"])
        predicted.append(out["tags"])
        with open(output_file, "w") as f:
            f.write(yaml.safe_dump(to_yaml))
        pbar.update()
    actual, predicted = np.array(actual, dtype=int), np.array(predicted, dtype=float)
    roc_auc = sk_metrics.roc_auc_score(actual, predicted)
    pr_auc = sk_metrics.average_precision_score(actual, predicted)
    report_card = f"""
    ROC-AUC: {roc_auc}
    PR-AUC: {pr_auc}
    """
    with open(output_folder / 'report_card.txt', 'w') as f:
        f.write(report_card)

def dummy_signal(sample_rate : int):
    return nussl.AudioSignal(
        audio_data_array = np.zeros((1, 100)),
        sample_rate = sample_rate
    )
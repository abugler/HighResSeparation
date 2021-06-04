from pathlib import Path
from tqdm import tqdm
import json
import glob
import nussl
import logging
import numpy as np
from nussl.evaluation import BSSEvalScale, BSSEvalV4
from nussl.separation.composite import OverlapAdd

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

def evaluate(
    separator,
    evaluator,
    resampler,
    test_data,
    output_folder : str = '.',
):
    output_folder = Path('.').absolute() / 'results'
    output_folder.mkdir(parents=True, exist_ok=True)
    order_of_mag = int(np.log(len(test_data))) + 1

    pbar = tqdm.tqdm(total=len(test_data))

    for i, item in enumerate(test_data):
        separator.audio_signal = item['mix']
        estimates = separator()

        evaluator.estimated_sources_list = estimates
        evaluator.true_sources_list = list(item['sources'].values())
        evaluator.source_labels = list(item['sources'].keys())

        file_name = f'{str(i).zfill(order_of_mag)}.json'
        pbar.set_description(file_name)
        scores = evaluator.evaluate()
        with open(output_folder / file_name, 'w') as f:
            json.dump(scores, f, indent=4)
        pbar.update()
    
    json_files = glob.glob(f"{output_folder}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)
    report_card = nussl.evaluation.report_card(df)

    logging.info(report_card)

    with open('report_card.txt', 'w') as f:
        f.write(report_card)
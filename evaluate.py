import os
import yaml.parser
from nussl.datasets.hooks import MUSDB18
from nussl.datasets.transforms import ToSeparationModel
from nussl import STFTParams
from nussl.separation.deep import DeepAudioEstimation
import torch
import torchaudio.transforms as tfa_transforms

import models
import data
import funcs

# Hyperparams
train_path = ''
device = 'cuda:0'
num_workers = 0

model_path = os.path.join(train_path, "checkpoints/best.model.pth")

class Bunch():
    def __init__(self, d : dict):
        self.__dict__.update(d)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except:
            self.__setattr__(name, None)
        return super().__getattribute__(name)

# This loads all the flags from training into a namespace.
with open(os.path.join(train_path, '.guild/attrs/flags')) as f:
    train_config = yaml.safe_load(f)
    train_namespace = Bunch(train_config)

# Dataset
if train_namespace.dataset == 'musdb':
    with open('.guild/sourcecode/data_conf/musdb_args_eval.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_musdb(True, **kwargs, sources=train_namespace.sources)[0]
elif train_namespace.dataset == 'openmic':
    with open('.guild/sourcecode/data_conf/openmic_args.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_openmic(True, **kwargs)
elif train_namespace.dataset == 'mtg_jamendo':
    with open('.guild/sourcecode/data_conf/mtg_jamendo_args.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_mtg_jamendo(True, **kwargs)[0]


# Model 
model = models.HRNet(
    train_namespace.closure_key,
    test_dataset.num_classes,
    pretrained=train_namespace.imagenet_pretrained,
    stft_params=STFTParams(window_length=train_namespace.window_length,
                           hop_length=train_namespace.hop_length,
                           window_type=train_namespace.window_type),
    head=train_namespace.task,
    stem=train_namespace.stem,
    audio_channels=1,
    skip=train_namespace.skip,
    spec_norm=train_namespace.spec_norm,
    waveform_norm=train_namespace.waveform_form,
    activation='sigmoid' if train_namespace.multiclass else 'softmax'
).to(device)

state_dict = torch.load(model_path, map_location=device)
models.state_dict_back_compat(state_dict)
model.load_state_dict(state_dict)

resampler = tfa_transforms.Resample(
    test_dataset.sample_rate,
    train_namespace.sample_rate
).to(device)

# Evaluation

if train_namespace.task == 'separation':
    evaluator = funcs.Evaluator(
        [funcs.dummy_signal(train_namespace.sample_rate)],
        [funcs.dummy_signal(train_namespace.sample_rate)],
        best_permutation_key='SI-SDR',
        bss_evalv4=True)
    funcs.evaluate(
        model, evaluator, 
        resampler, test_dataset,
        device=device
    )
elif train_namespace.task == 'classification':
    funcs.evaluate_tagger(test_dataset, model, device, num_workers)
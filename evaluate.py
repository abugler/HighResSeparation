import os
import yaml.parser
from nussl.datasets.hooks import MUSDB18
from nussl.datasets.transforms import ToSeparationModel
from nussl import STFTParams
import torch
import torchaudio.transforms as tfa_transforms

import models
import data

# Hyperparams
train_path = ''

model_path = os.path.join(train_path, "checkpoints/best.model.pth")

class Bunch():
    def __init__(self, d : dict):
        super().__dict__.update(d)

# This loads all the flags from training into a namespace.
with open(os.path.join(train_path, '.guild/attrs/flags')) as f:
    train_config = yaml.safe_load(f)
    train_namespace = Bunch(train_config)

# Dataset
if train_namespace.dataset == 'musdb':
    with open('.guild/sourcecode/data_conf/musdb_args.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_musdb(True, **kwargs)
elif train_namespace.dataset == 'openmic':
    with open('.guild/sourcecode/data_conf/openmic_args.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_openmic(True, **kwargs)
elif train_namespace.dataset == 'mtg_jamendo':
    with open('.guild/sourcecode/data_conf/mtg_jamendo_args.yml') as s:
        kwargs = yaml.load(s)
    test_dataset = data.build_mtg_jamendo(True, **kwargs)




# Model 
separate = train_namespace.task == 'separation'

model = torch.load(model_path)

resampler = tfa_transforms.Resample(
    test_dataset.sample_rate,
    train_namespace.sample_rate
).to(train_namespace.device)

# Evaluation
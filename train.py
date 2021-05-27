import torch
import timm
import data
import yaml
from yaml import Loader
import os
import logging

# Hyperparameters
hrnet_pretrained = False
hrnet_width = 32
dataset = 'musdb'
toy_dataset = False
task = 'separation'

# Setup Logging
logger = logging.getLogger('train')
logger.info(f"The code is being run from {os.getcwd()}")

# Dataset
if dataset == 'musdb':
    with open('.guild/sourcecode/data_conf/musdb_args.yml') as s:
        kwargs = yaml.load(s)
    if toy_dataset:
        kwargs['num_tracks'] = 1
    train_dataset, val_dataset, test_dataset = data.build_musdb(**kwargs)
elif dataset == 'openmic':
    with open('.guild/sourcecode/data_conf/openmic_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset, test_dataset = data.build_openmic(**kwargs)
elif dataset == 'mtg_jamendo':
    with open('.guild/sourcecode/data_conf/mtg_jamendo_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset, test_dataset = data.build_mtg_jamendo(**kwargs)

# Model 
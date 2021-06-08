import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as tfa_transforms
import timm
import data
import models
import yaml
import nussl
from ignite.engine import Events
from yaml import Loader
import os
import logging
from nussl import STFTParams
import funcs

# Hyperparameters
seed = 0

dataset = 'musdb'
toy_dataset = False

imagenet_pretrained = False
closure_key = 'hrnet_w18_small_v2'
task = 'separation'
window_length = 512
hop_length = 128
window_type = 'sqrt_hann'
sample_rate = 22_050
stem = False
skip = False

resume = None
batch_size = 8
learning_rate = .01
momentum = .9
weight_decay = .0005
epochs = 100
epoch_length = 100
valid_epoch_length = None
poly_power = .9
num_workers = 8
device = 'cuda:0'
optimizer = 'sgd'

# Seeding
torch.manual_seed(seed)
np.random.seed(seed)

# Setup Logging
logger = logging.getLogger('train')
logger.info(f"The code is being run from {os.getcwd()}")

os.mkdir('tensorboard')
writer = SummaryWriter(log_dir='tensorboard')

# Dataset
if dataset == 'musdb':
    with open('.guild/sourcecode/data_conf/musdb_args.yml') as s:
        kwargs = yaml.load(s)
    if toy_dataset:
        kwargs['num_tracks'] = 1
    train_dataset, val_dataset = data.build_musdb(False, **kwargs)
    if toy_dataset:
        train_dataset.num_excerpts = 1
        val_dataset = train_dataset
elif dataset == 'openmic':
    with open('.guild/sourcecode/data_conf/openmic_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset = data.build_openmic(False, **kwargs)
elif dataset == 'mtg_jamendo':
    with open('.guild/sourcecode/data_conf/mtg_jamendo_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset = data.build_mtg_jamendo(False, **kwargs)

# Model and Optimizer
model = models.HRNet(
    closure_key,
    train_dataset.num_classes,
    pretrained=imagenet_pretrained,
    stft_params=STFTParams(window_length=window_length,
                           hop_length=hop_length,
                           window_type=window_type),
    head=task,
    stem=stem,
    audio_channels=1,
    skip=skip
).to(device)

if optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
elif optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

if resume:
    resume_model_path = os.path.join(resume, 'checkpoints/best.model.pth')
    model_state_dict = torch.load(resume_model_path)
    model.load_state_dict(model_state_dict)

    resume_optimizer_path = os.path.join(resume, 'checkpoints/best.optimizer.pth')
    optimizer_state_dict = torch.load(resume_optimizer_path)
    optimizer.load_state_dict(optimizer_state_dict)

resampler = tfa_transforms.Resample(
    train_dataset.sample_rate,
    sample_rate
).to(device)

if task == 'separation':
    sisdr = funcs.SISDR()
    recon_loss = funcs.ReconstructionLoss()
if task == 'segmentation':
    ce_loss = funcs.CrossEntropy()



# Training Setup                        

train_dataloader = DataLoader(train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(val_dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=True)

def dict_to_item(d : dict):
    for k, v in d.items():
        d[k] = v.item()
    return d

def train_step(engine, batch):
    
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    funcs.resample_batch(batch, resampler)
    output = model(batch['mix_audio'])
    if task == 'separation':
        model.stft.direction = 'transform'
        loss_dict = {
            'loss': recon_loss(output, batch, model.stft),
            'si-sdr': sisdr(output, batch)
        }
    elif task == 'classification':
        loss_dict = {
            'loss': funcs.classification_loss(output, batch)
        }

    elif task == 'segmentation':
        loss_dict = {
            'loss': ce_loss(output, batch)
        }
    loss_dict['loss'].backward()
    optimizer.step()
    return dict_to_item(loss_dict)

def val_step(engine, batch):
    with torch.no_grad():
        model.eval()
        funcs.resample_batch(batch, resampler)
        output = model(batch['mix_audio'])
        if task == 'separation':
            model.stft.direction = 'transform'
            loss_dict = {
                'loss': recon_loss(output, batch, model.stft),
                'si-sdr': sisdr(output, batch)
            }
        elif task == 'classification':
            loss_dict = {
                'loss': funcs.classification_loss(output, batch)
            }

        elif task == 'segmentation':
            loss_dict = {
                'loss': ce_loss(output, batch)
            }
    return dict_to_item(loss_dict)

trainer, validator = nussl.ml.train.create_train_and_validation_engines(
    train_step, val_step, device=device
)

# Ignite Handlers

@trainer.on(Events.ITERATION_COMPLETED)
def on_iteration_completed(engine):
    # Log iteration metrics
    for key in engine.state.iter_history:
        if engine.state.iteration % 1 == 0:
            writer.add_scalar(
                'iter/' + key,
                engine.state.iter_history[key][-1], 
                engine.state.iteration
            )
    # Poly-Learning Rate
    poly_lr = (learning_rate
               * (1 - engine.state.iteration / max_iterations) ** poly_power)
    for g in optimizer.param_groups:
        g['lr'] = poly_lr

if valid_epoch_length is not None:
    @trainer.on(Events.EPOCH_STARTED)
    def load_validator_state_dict(_):
        # Load validation epoch length
        validator.load_state_dict({
            'iteration': 0,
            'max_epochs': 1,
            'epoch_length': valid_epoch_length
        })

@trainer.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
def on_epoch_completed(engine):
    # Log validation metrics
    for key in engine.state.epoch_history:
        writer.add_scalar(
            key,
            engine.state.epoch_history[key][-1], 
            engine.state.epoch
        )

max_iterations = epochs * epoch_length

nussl.ml.train.add_stdout_handler(trainer, validator)
nussl.ml.train.add_validate_and_checkpoint(os.getcwd(), model, 
    optimizer, train_dataset, trainer, val_dataloader, validator)    
nussl.ml.train.add_progress_bar_handler(validator)
nussl.ml.train.add_progress_bar_handler(trainer)
trainer.run(train_dataloader, epoch_length=epoch_length, max_epochs=epochs)

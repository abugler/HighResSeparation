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
dataset = 'musdb'
toy_dataset = False

imagenet_pretrained = False
hrnet_width = 18
task = 'separation'
window_length = 512
hop_length = 128
window_type = 'sqrt_hann'
sample_rate = 22_500

batch_size = 16
learning_rate = .01
momentum = .9
weight_decay = .0005
epochs = 100
epoch_length = 100
poly_power = .9
num_workers = 8
device = 'cuda:0'

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
elif dataset == 'openmic':
    with open('.guild/sourcecode/data_conf/openmic_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset = data.build_openmic(False, **kwargs)
elif dataset == 'mtg_jamendo':
    with open('.guild/sourcecode/data_conf/mtg_jamendo_args.yml') as s:
        kwargs = yaml.load(s)
    train_dataset, val_dataset = data.build_mtg_jamendo(False, **kwargs)

# Model 
separate = task == 'separation'
model = models.HRNet(
    train_dataset.num_classes,
    pretrained=imagenet_pretrained,
    width=hrnet_width,
    stft_params=STFTParams(window_length=window_length,
                           hop_length=hop_length,
                           window_type=window_type),
    separate=separate,
    audio_channels=1
).to(device)

resampler = tfa_transforms.Resample(
    train_dataset.sample_rate,
    sample_rate
).to(device)

# Training Setup

optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weight_decay)

train_dataloader = DataLoader(train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(val_dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=True)

def resample_batch(batch):
    with torch.no_grad():
        batch['mix_audio'] = resampler(batch['mix_audio'])
        if 'source_audio' in batch:
            batch['source_audio'] = resampler(
                batch['source_audio'].permute(0, 1, 3, 2).contiguous()
            ).permute(0, 1, 3, 2)

def train_step(engine, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    resample_batch(batch)
    output = model(batch['mix_audio'])
    if separate:
        model.stft.direction = 'transform'
        loss = funcs.reconstruction_loss(output, batch, model.stft)
    else:
        loss = funcs.classification_loss(output, batch)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item()}

def val_step(engine, batch):
    with torch.no_grad():
        model.eval()
        resample_batch(batch)
        output = model(batch['mix_audio'])
        if separate:
            model.stft.direction = 'transform'
            loss = funcs.reconstruction_loss(output, batch, model.stft)
        else:
            loss = funcs.classification_loss(output, batch)
    return {'loss': loss.item()}

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

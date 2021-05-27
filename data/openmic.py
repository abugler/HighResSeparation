import os
import librosa
import torch
import torch.utils.data as data
import nussl
import numpy as np
import pathlib
import yaml
from typing import List


class OpenMIC(data.Dataset):
    """
    OpenMIC polyphonic instrument identification dataset.
    Number of channels is inconsistent, and all stereo signals will
    be converted to mono.
    Sampling rate is 44_100. This is converted to 16_000 in the model.
    Args:
    audio_dir : str
      Directory containing all audio.
    tags_file : str
      CSV file containing tags for directory.
    split_file : str
      CSV file containing sample_keys for this split.
    class_map_file : str, optional
      JSON file containing mapping from instrument to int.
      If not provided, it will be infered from the audio path.
    audio_duration : float, optional
      Duration of each audio signal in seconds. Default: 10.0
    sample_rate : int, optional
      Sample rate of returned sound. If not 44_100, then audio
      will be resampled on the fly. Default: 44_100
    device : str, optional
      Device for resampling. Default: 'cpu'.
      You likely want the same device as the model.
    """
    def __init__(self,
                 audio_dir : str,
                 tags_file : str,
                 split_file : str,
                 class_map_file : str = None,
                 audio_duration : float = 10.0):

        self.audio_dir = audio_dir
        self.num_classes = 20
        assert audio_duration <= 10.0
        self.audio_duration = audio_duration
        self.sample_rate = 44_100
        self.num_channels = 1
        self.stft_params = nussl.STFTParams()
        self.metadata = {}
        self.num_samples = int(44_100 * audio_duration)
        super().__init__()
        with open(split_file) as file:
            self.split_keys = set(file.read().split("\n")[1:])
        if class_map_file is None:
            class_map_file = os.path.join(pathlib.Path(audio_dir).parent, "class-map.json")
        with open(class_map_file) as file:
            self.class_map = yaml.safe_load(file.read())
        self.inverted_class_map = {
            v: k for k, v in self.class_map.items()
        }
        with open(tags_file) as file:
            tags_lines = file.read().split("\n")[1:]
        self._make_paths_and_tags(tags_lines)

    def _make_paths_and_tags(self, tags_lines):
        """
        Saves paths and tags into self.paths_and_tags
        """
        tags_dict = {}
        for line in tags_lines:
            try:
                sample_key, instrument, _, _ = line.split(',')
            except ValueError:
                continue
            tags_dict[sample_key] = tags_dict.get(sample_key, []) + [self.class_map[instrument]]
        self.paths_and_tags = []
        for subfolder in os.listdir(self.audio_dir):
            abs_subfolder = os.path.join(self.audio_dir, subfolder)
            for audio_path in os.listdir(abs_subfolder):
                sample_key, _ = os.path.splitext(audio_path)
                if sample_key not in self.split_keys:
                    continue
                tags = tags_dict[sample_key]
                tag_array = np.zeros(20)
                for tag in tags:
                    tag_array[tag] = 1
                self.paths_and_tags.append({
                    'path': os.path.join(abs_subfolder, audio_path),
                    'tags': tag_array
                })

    def __len__(self):
        return len(self.paths_and_tags)

    def __getitem__(self, index):
        item = self.paths_and_tags[index]
        audio_signal = nussl.AudioSignal(
            path_to_input_file=item['path'],
            sample_rate=44_100).to_mono()
        full_signal_length = audio_signal.signal_length
        random_sample_shift = torch.randint(full_signal_length - self.num_samples, ())
        audio_signal.audio_data = audio_signal.audio_data[:, random_sample_shift:]
        audio_signal.truncate_samples(self.num_samples)
        instruments = [self.inverted_class_map[i] for i, tag in enumerate(item['tags'])
                       if tag == 1]
        return {
            'mix_audio': audio_signal.audio_data,
            'tags': item['tags']
        }
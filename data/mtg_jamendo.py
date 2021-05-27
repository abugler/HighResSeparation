import os
import librosa
import torch
import torch.utils.data as data
import nussl
import numpy as np


num_classes_dict = {
    "genre": 87,
    "instrument": 40,
    "moodtheme": 56,
    "top50tags": 50,
    "all": 183
}

class MTGJamendo(data.Dataset):
    """
    Indexing returns:
     - "audio": Audio data, in shape (channels, samples)
     - "tags": One-hot tags, in shape (num_tags)
    __getitem__ obtains the center section of the track.
    All songs are turned into mono tracks.
    
    Adapted from 
    https://github.com/MTG/mtg-jamendo-dataset/blob/ca8bc9aa676e0bb69611c841242f8cefec97e40e/scripts/baseline/data_loader.py#L29
    """
    def __init__(self, audio_root, metadata_root,
                 split_num=0, split_tvt='train',
                 tag_type='genre', audio_duration=10):
        assert tag_type in num_classes_dict.keys()
        assert split_tvt in ["train", "test", "validation"]
        assert audio_duration < 30
        str_tag = f"_{tag_type}" if tag_type != "all" else ""
        self.metadata_file = f"{metadata_root}/splits/split-{split_num}/autotagging{str_tag}-{split_tvt}.tsv"  # noqa
        self.num_classes = num_classes_dict[tag_type]
        self.audio_root = audio_root
        self.audio_duration = audio_duration
        self.tag_type = tag_type

        # Metadata for nussl
        self.sample_rate = 44_100
        self.num_channels = 1
        self.stft_params = nussl.STFTParams()
        self.metadata = {'name': "MTGJamendo"}
        self.generate_tags()


    def __getitem__(self, index):
        audio_path = os.path.join(self.audio_root, self.song_names[index])
        duration = librosa.get_duration(filename=audio_path, sr=44_100)
        begin = (duration // 2 - self.audio_duration // 2)
        audio_signal = nussl.AudioSignal(
            path_to_input_file=audio_path, duration=self.audio_duration,
            sample_rate=44_100, offset=begin
        )
        audio_data = torch.Tensor(audio_signal.to_mono().audio_data)
        tags = self.genre_labels[index]

        return {
            "mix_audio": audio_data,
            "tags": tags
        }

    def generate_tags(self):
        self.class_map = {}
        with open(self.metadata_file) as file:
            lines = [line.strip().split("\t") for line in file.readlines()[1:]]
        self.song_names, self.genre_labels = [], np.zeros((len(lines), self.num_classes), dtype=np.float32)
        for idx, line in enumerate(lines):
            path, tags = line[3], line[5:]
            self.song_names.append(path)
            for tag in tags:
                if self.tag_type != "all" and self.tag_type not in tag:
                    continue
                jdx = self.class_map.get(tag, len(self.class_map))

                if jdx == len(self.class_map):
                    self.class_map[tag] = len(self.class_map)
                self.genre_labels[idx, jdx] = 1
        self.inverted_class_map = {v: k for k, v in self.class_map.items()}

    def __len__(self):
        return len(self.song_names)
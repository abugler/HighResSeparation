import nussl
import musdb as musdb_sigsep
import tqdm
import numpy as np
from collections import OrderedDict

class SegmentedMUSDB(nussl.datasets.BaseDataset):
    """
    FastLoadMUSDB is MUSDB, but does not load entire songs
    when __getitem__ is called, but instead returns a small segment of the song.
    This is different from using the GetExcerpt transform, since the usage of
    GetExcerpt requires that the entire audio array is loaded into memory.
    This is effective at reducing disk read times on machines with
    slower disk read times.
    Parameters
    ----------
    folder : str
        Root folder for musdb dataset.
    excerpt_duration : float
        Excerpt Length in seconds.
    hop_duration : float
        Distance in seconds between each start of the excerpt.
    num_tracks : int, optional
        Number of MUSDB tracks to use. Default is all of them.
    nonsilent_sources : List[str], optional
        If provided, and hop_duration is also provided, these sources are guarenteed
        to not be silent in these sources.
    
    """
    def __init__(self, folder='', is_wav=False, excerpt_duration=10, hop_duration=0,
                 num_tracks=None, subsets=None, split=None, **kwargs):
        self.sample_rate = 44_100
        self.excerpt_duration = excerpt_duration
        self.excerpt_length = int(self.sample_rate * excerpt_duration)
        self.hop_duration = hop_duration
        self.num_tracks = num_tracks
        self.num_classes = 4

        subsets = ['train', 'test'] if subsets is None else subsets
        self.musdb = musdb_sigsep.DB(root=folder, is_wav=is_wav, download=False, 
                              subsets=subsets, split=split)
        self.make_excerpt_ranges()
        super().__init__(folder, **kwargs)
        self.metadata['subsets'] = subsets
        self.metadata['split'] = split

    def cache_stems(self, track):
        track._stems = [src.copy() for src in track.stems.astype(np.float32)]
        track._audio = track.stems[0]

    def excerpts_from_track(self, idx):
        track = self.musdb[idx]
        self.cache_stems(track)
        mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

        return np.arange(
            0, 
            mix.signal_duration - self.excerpt_duration,
            self.hop_duration
        )
    
    def filter_tracks(self):
        self.musdb.tracks = sorted(self.musdb.tracks, key=lambda t: t.path)
        if self.num_tracks is not None:
            self.musdb.tracks = self.musdb.tracks[:self.num_tracks]

    def make_excerpt_ranges(self):
        self.filter_tracks()
        excerpt_mappings = OrderedDict()
        excerpts = []
        num_excerpts = 0
        for idx in tqdm.tqdm(range(len(self.musdb.tracks)),
                             desc='Loading excerpt locations...'):
            track_excerpts = self.excerpts_from_track(idx)
            excerpts.append(track_excerpts)
            excerpt_mappings[num_excerpts] = idx
            num_excerpts += track_excerpts.shape[0]
        self.excerpt_mappings = excerpt_mappings
        self.excerpts = excerpts
        self.num_excerpts = num_excerpts

    def __len__(self):
        return self.num_excerpts
    
    def get_items(self, folder):
        return list(range(self.num_excerpts))

    def _get_idx(self, item):
        """
        Returns (`musdb_item`, `excerpt_idx`), such that
        self.excerpts[musdb_item][excerpt_idx] gets the first sample for the
        segment referred to by item.
        This is used during evaluation, to ensure that every excerpt is evaluated over once.
        """
        musdb_item = -1
        keys = self.excerpt_mappings.keys()
        for key in keys:  # keys are sorted
            if item >= key:
                musdb_item += 1
            else: break
        return musdb_item, item - key

    def musdb_track_to_audio_signals(self, item):
        item, excerpt_num = self._get_idx(item)
        track = self.musdb[item]
        excerpts = self.excerpts[item]

        track_start = excerpts[excerpt_num]

        mix, sources = nussl.utils.musdb_track_to_audio_signals(track)
        samples_start = int(track_start * 44_100)
        samples_end = samples_start + self.excerpt_length
        mix = mix.make_copy_with_audio_data(mix.audio_data[:1, samples_start:samples_end])
        mix.zero_pad(0, self.excerpt_length - mix.signal_length)
        for source, signal in sources.items():
            sources[source] = signal.make_copy_with_audio_data(
                signal.audio_data[:1, samples_start:samples_end])
            sources[source].zero_pad(0, self.excerpt_length - sources[source].signal_length)

        return mix, sources

    def process_item(self, item):
        mix, sources = self.musdb_track_to_audio_signals(item)
        self._setup_audio_signal(mix)
        source_names = sorted(list(sources.values()))
        source_audio = []
        for source in source_names:
            self._setup_audio_signal(source)
            source_audio.append(source.audio_data)

        source_audio_data = np.stack(source_audio, axis=-1)
        output = {
            'mix_audio': mix.audio_data,
            'source_audio': source_audio_data,
            'metadata': {
                'labels': ['bass', 'drums', 'other', 'vocals']
            }
        }
        return output
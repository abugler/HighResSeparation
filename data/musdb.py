import nussl
import musdb as musdb_sigsep
import tqdm
import numpy as np
import librosa
from collections import OrderedDict


default_sources = ['bass', 'drums', 'other', 'vocals']
class SegmentedMUSDB(nussl.datasets.BaseDataset):
    """
    SegmentedMUSDB is a dataset than indexes excerpt of tracks from MUSDB, rather than tracks, like the nussl MUSDB18 dataset.

    For example:
    >>> dataset = SegmentedMUSDB(*args **kwargs)
    >>> item1 = dataset[0]
    >>> item2 = dataset[1]

    `item1` and `item2` may be from the same track, but they will contain audio data from different parts of the track.

    If `sources` differs from the default, then only excerpts that contain at least one active source will be included.

    Parameters
    ----------
    folder : str
        Root folder for musdb dataset.
    is_wav : bool, optional
        Expect wav files rather than stems. Default: False
    excerpt_duration : float
        Excerpt Length in seconds.
    hop_duration : float
        Distance in seconds between each start of the excerpt.
    num_tracks : int, optional
        Number of MUSDB tracks to use. Default is all of them.
    sources : list[str], optional
        List of sources to separate. Default is ['bass', 'drums', 'other', 'vocals'].
    threshold_db : float, options
    """
    
    def __init__(self, folder='', is_wav=False, excerpt_duration=10, hop_duration=5,
                 num_tracks=None, subsets=None, split=None, sources=None, 
                 threshold_db=-45, **kwargs):
        self.sample_rate = 44_100
        self.excerpt_duration = excerpt_duration
        self.excerpt_length = int(self.sample_rate * excerpt_duration)
        self.hop_duration = hop_duration
        self.num_tracks = num_tracks
        self.sources = default_sources if sources is None else sources
        assert all([source in default_sources for source in self.sources])
        self.num_classes = len(self.sources)
        self.threshold_db = threshold_db

        subsets = ['train', 'test'] if subsets is None else subsets
        self.musdb = musdb_sigsep.DB(root=folder, is_wav=is_wav, download=False, 
                              subsets=subsets, split=split)
        self.make_excerpt_ranges()
        super().__init__(folder, **kwargs)
        self.metadata['subsets'] = subsets
        self.metadata['split'] = split

    def cache_stems(self, track):
        """
        Stems are held in memory for fast loading.
        """
        track._stems = [src.copy() for src in track.stems.astype(np.float32)]
        track._audio = track.stems[0]

    def excerpts_from_track(self, idx):
        track = self.musdb[idx]
        self.cache_stems(track)
        mix, sources = nussl.utils.musdb_track_to_audio_signals(track)
        
        if self.num_classes < 4:
            starts = set()
            for source in self.sources:
                salient = set(
                    find_salient_starts(
                        sources[source].audio_data,
                        self.excerpt_duration,
                        self.hop_duration / self.excerpt_duration,
                        sr=44_100,
                        threshold_db=self.threshold_db
                    ) / 44_100
                )
                starts = starts.union(salient)
            starts = np.array(list(starts))
        else:
            starts = np.arange(
                0, 
                mix.signal_duration - self.excerpt_duration,
                self.hop_duration
            )
        return starts
    
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
        excerpt referred to by item.

        This is especially useful during evaluation, to ensure that every excerpt is evaluated over once.
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
        source_names = sorted(list(self.sources))
        source_audio = []
        for source_name in source_names:
            source = sources[source_name]
            self._setup_audio_signal(source)
            source_audio.append(source.audio_data)

        source_audio_data = np.stack(source_audio, axis=-1)
        output = {
            'mix_audio': mix.audio_data,
            'source_audio': source_audio_data,
            'metadata': {
                'labels': source_names
            }
        }
        return output

def find_salient_starts(audio, duration_sec, hop_ratio, sr, threshold_db=-60.0):
    # finds frames in the audio where the RMS is above a dB threshold
    
    dur = int(sr * duration_sec)
    hop_dur = int(dur * hop_ratio)
    threshold = np.power(10.0, threshold_db / 20.0)
    rms = librosa.feature.rms(audio, frame_length=dur, hop_length=hop_dur)[0, :]
    loud = np.squeeze(np.argwhere(rms > threshold))
    fr = lambda t: np.atleast_1d(librosa.frames_to_samples(t,
                                                           hop_length=hop_dur))
    return fr(loud)
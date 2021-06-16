from .musdb import SegmentedMUSDB
from .openmic import OpenMIC
from .mtg_jamendo import MTGJamendo

ALIGN_CORNERS = None

musdb_source_mapping = {
    'b': 'bass',
    'd': 'drums',
    'o': 'other',
    'v': 'vocals'
}

def build_musdb(evaluate : bool,
                folder : str,
                is_wav : bool,
                excerpt_duration : float,
                hop_duration : float,
                num_tracks : int,
                sources : str = None,
                **kwargs):
    """
    Returns a train, validiation, and testing dataset for MUSDB
    """
    # musdb is strange with their splits
    splits = (
        [(['train'], 'train'),
        (['train'], 'valid')] if not evaluate else [(['test'], None)]
    )
    sources = 'bdov' if sources is None else sources
    sources = [musdb_source_mapping[s] for s in sources]
    datasets = []
    for subsets, split in splits:
        datasets.append(
            SegmentedMUSDB(
                folder, is_wav, excerpt_duration,
                hop_duration, num_tracks,
                subsets, split, sources=sources)
        )
    return datasets

def build_mtg_jamendo(evaluate : bool,
                      audio_root : str,
                      metadata_root : str,
                      split_num : int,
                      tag_type : str,
                      audio_duration : int):
    splits = ['train', 'validation'] if not evaluate else ['test']
    return [
        MTGJamendo(audio_root, metadata_root,
                   split_num, split, tag_type, audio_duration)
        for split in splits
    ]

def build_openmic(evaluate : bool,
                  audio_dir : str,
                  tags_file : str,
                  train_split_file : str,
                  val_split_file : str,
                  test_split_file : str,
                  class_map_file : str = None,
                  audio_duration : float = 10.0):
    if not evaluate:
        return (
            OpenMIC(audio_dir, tags_file, train_split_file,
                    class_map_file, audio_duration),
            OpenMIC(audio_dir, tags_file, val_split_file,
                    class_map_file, audio_duration))
    else:
        return OpenMIC(audio_dir, tags_file, test_split_file,
                       class_map_file, audio_duration)

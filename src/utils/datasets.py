import json
from pathlib import Path
import pandas as pd
import re

class BaseDataset:
    """Base dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        self.cfg = cfg
        self.dataset = {'train': None, 'valid': None, 'test': None}
        
        if cfg is None:
            self.path = Path(root_dir)
        else:
            self.path = Path(cfg.paths.dataset_dir)


class Clotho(BaseDataset):
    """Clotho dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        self.path = self.path / 'audio_text/Clotho'

        captions = pd.read_csv(self.path / 'clotho_captions_development.csv').values
        self.dataset['train'] = {
            'audio': list((self.path / 'development').glob('*.wav')),
            'text': {row[0]: list(row[1:]) for row in captions},
        }
        captions = pd.read_csv(self.path / 'clotho_captions_validation.csv').values
        self.dataset['valid'] = {
            'audio': list((self.path / 'validation').glob('*.wav')),
            'text': {row[0]: list(row[1:]) for row in captions},
        }
        captions = pd.read_csv(self.path / 'clotho_captions_evaluation.csv').values
        self.dataset['test'] = {
            'audio': list((self.path / 'evaluation').glob('*.wav')),
            'text': {row[0]: list(row[1:]) for row in captions},
        }


class AudioCaps(BaseDataset):
    """AudioCaps dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        audiopath = self.path / 'audio_text/AudioCaps/audio'
        metapath = self.path / 'audio_text/AudioCaps/metafile'

        train_audiofiles = list(audiopath.glob('train/*.wav'))
        val_audiofiles = list(audiopath.glob('val/*.wav'))
        test_audiofiles = list(audiopath.glob('test/*.wav'))
        train_metafiles = pd.read_csv(metapath / 'train.csv')
        val_metafiles = pd.read_csv(metapath / 'val.csv')
        test_metafiles = pd.read_csv(metapath / 'test.csv')
        train_metadata, val_metadata, test_metadata = {}, {}, {}
        for _, row in train_metafiles.iterrows():
            audiofile = 'Y' + row['youtube_id'] + '.wav'
            if audiofile not in train_metadata:
                train_metadata[audiofile] = []
            train_metadata[audiofile].append(row['caption'])
        for _, row in val_metafiles.iterrows():
            audiofile = 'Y' + row['youtube_id'] + '.wav'
            if audiofile not in val_metadata:
                val_metadata[audiofile] = []
            val_metadata[audiofile].append(row['caption'])
        for _, row in test_metafiles.iterrows():
            audiofile = 'Y' + row['youtube_id'] + '.wav'
            if audiofile not in test_metadata:
                test_metadata[audiofile] = []
            test_metadata[audiofile].append(row['caption'])
        self.dataset['train'] = {
            'audio': train_audiofiles,
            'text': train_metadata,
        }
        self.dataset['valid'] = {
            'audio': val_audiofiles,
            'text': val_metadata,
        }
        self.dataset['test'] = {
            'audio': test_audiofiles,
            'text': test_metadata,
        }


class Freesound(BaseDataset):
    '''Freesound dataset.'''
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        self.path = self.path / 'audio_text/Freesound'

        train_audiofiles = list((self.path / 'train').glob('*.flac'))
        test_audiofiles = list((self.path / 'test').glob('*.flac'))
        train_metadata = {}
        test_metadata = {}
        for audiofile in train_audiofiles:
            if audiofile.name not in train_metadata:
                train_metadata[audiofile.name] = []
            with open(audiofile.with_suffix('.json')) as f:
                train_metadata[audiofile.name].extend(json.load(f)['text'])
        for audiofile in test_audiofiles:
            if audiofile.name not in test_metadata:
                test_metadata[audiofile.name] = []
            with open(audiofile.with_suffix('.json')) as f:
                test_metadata[audiofile.name].extend(json.load(f)['text'])
        self.dataset['train'] = {
            'audio': train_audiofiles,
            'text': train_metadata,
        }
        self.dataset['valid'] = {
            'audio': test_audiofiles,
            'text': test_metadata,
        }
        self.dataset['test'] = {
            'audio': test_audiofiles,
            'text': test_metadata,
        }
    

class ESC50(BaseDataset):
    """ESC-50 dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        self.path = self.path / 'audio/ESC-50'
        metafile = pd.read_csv(self.path / 'meta/esc50.csv')
        self.dataset['train'] = {
            'audio': list((self.path / 'audio').glob('*.wav')),
            'text': {row['filename']: row['category'] for _, row in metafile.iterrows()},
        }
        # Using different folders for validation and test
        self.dataset['valid'] = self.dataset['train'] 
        self.dataset['test'] = self.dataset['train']
        self.labels_dict = json.load(open(self.path / 'ESC50_class_labels_indices.json'))


class sAudioCaps(BaseDataset):
    """spatial AudioCaps dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        dataset_name = kwargs['dataset_name'][1:]
        self.path = self.path / 'spatial_audio_text' / dataset_name

        self.dataset['train'] = {
            'audio': sorted((self.path / 'audio/train').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/train').glob('*.json')),
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/train').glob('*.json')),
        }
        self.dataset['valid'] = {
            'audio': sorted((self.path / 'audio/valid').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/valid').glob('*.json')),
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/valid').glob('*.json')),
        }
        self.dataset['test'] = {
            'audio': sorted((self.path / 'audio/test').glob('*.flac')),
            # 'audio': [file for file in (self.path / 'audio/test').glob('*.flac') if '_0.flac' in str(file)],
            'metadata': sorted((self.path / 'metadata/test').glob('*.json')),
            # 'metadata': [file for file in (self.path / 'metadata/test').glob('*.json') if '_0.json' in str(file)],
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/test').glob('*.json')),
        }


class sClotho(BaseDataset):
    """spatial Clotho dataset."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        dataset_name = kwargs['dataset_name'][1:]
        self.path = self.path / 'temporal_spatial_audio_text' / dataset_name
        print('sClotho:', self.path)

        self.dataset['train'] = {
            'audio': sorted((self.path / 'audio/train').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/train').glob('*.json')),
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/train').glob('*.json')),
        }
        self.dataset['valid'] = {
            'audio': sorted((self.path / 'audio/valid').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/valid').glob('*.json')),
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/valid').glob('*.json')),
        }
        self.dataset['test'] = {
            'audio': sorted((self.path / 'audio/test').glob('*.flac')),
            # 'audio': [file for file in (self.path / 'audio/test').glob('*.flac') if '_0.flac' in str(file)],
            'metadata': sorted((self.path / 'metadata/test').glob('*.json')),
            # 'metadata': [file for file in (self.path / 'metadata/test').glob('*.json') if '_0.json' in str(file)],
            # 'metadata': sorted((self.path / 'metadata_qwen3-8b/test').glob('*.json')),
        }
    
class stClotho(BaseDataset):
    """spatial Clotho dataset with nested negative samples."""
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        dataset_name = kwargs['dataset_name'][2:]
        self.path = self.path / 'temporal_spatial_audio_text' / dataset_name
        self.neg_path = self.path.parent / 'stClotho_negative'
        print('stClotho:', self.path)
        print('stClotho negative samples:', self.neg_path)

        def _id_from_stem(stem: str):
            # temporal_00000 / temporal_negative_00000 / spatial_negative_00000 -> 00000
            m = re.search(r'(\d+)$', stem)
            return m.group(1) if m else None

        for split in ['train', 'valid', 'test']:
            pos_audio_files = list((self.path / f'audio/{split}').glob('*.flac'))
            pos_meta_files  = list((self.path / f'metadata/{split}').glob('*.json'))

            neg_audio_files = list((self.neg_path / f'audio/{split}').rglob('*.flac'))
            neg_meta_files  = list((self.neg_path / f'metadata/{split}').rglob('*.json'))

            pos_audio = { _id_from_stem(p.stem): p for p in pos_audio_files }
            pos_meta  = { _id_from_stem(p.stem): p for p in pos_meta_files }

            neg_temporal_audio = {}
            neg_spatial_audio  = {}
            for p in neg_audio_files:
                k = _id_from_stem(p.stem)
                if k is None:
                    continue
                if p.stem.startswith('temporal_negative_'):
                    neg_temporal_audio[k] = p
                elif p.stem.startswith('spatial_negative_'):
                    neg_spatial_audio[k] = p

            neg_temporal_meta = {}
            neg_spatial_meta  = {}
            for p in neg_meta_files:
                k = _id_from_stem(p.stem)
                if k is None:
                    continue
                if p.stem.startswith('temporal_negative_'):
                    neg_temporal_meta[k] = p
                elif p.stem.startswith('spatial_negative_'):
                    neg_spatial_meta[k] = p

            keys = sorted(set(pos_audio) & set(pos_meta) &
                          set(neg_temporal_audio) & set(neg_spatial_audio) &
                          set(neg_temporal_meta) & set(neg_spatial_meta))

            assert len(keys) > 0, f"No matched triplets found in split={split}"

            audio_triplets = [(pos_audio[k], neg_temporal_audio[k], neg_spatial_audio[k]) for k in keys]
            meta_triplets  = [(pos_meta[k],  neg_temporal_meta[k],  neg_spatial_meta[k])  for k in keys]

            self.dataset[split] = {
                'audio': audio_triplets,
                'metadata': meta_triplets,
            }


        

class sFreesound(BaseDataset):
    '''Freesound dataset.'''
    def __init__(self, cfg=None, root_dir='datasets', **kwargs):
        super().__init__(cfg, root_dir)
        dataset_name = 'Freesound'
        self.path = self.path / 'spatial_audio_text' / dataset_name

        self.dataset['train'] = {
            'audio': sorted((self.path / 'audio/train').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/train').glob('*.json')),
        }
        self.dataset['valid'] = {
            'audio': sorted((self.path / 'audio/test').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/test').glob('*.json')),
        }
        self.dataset['test'] = {
            'audio': sorted((self.path / 'audio/test').glob('*.flac')),
            'metadata': sorted((self.path / 'metadata/test').glob('*.json')),
        }
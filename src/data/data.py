import json
import numpy as np
import random
import ast

import torch
import torchaudio

from .components.data import BaseDataset
from utils.config import get_dataset



class CLAPDataset(BaseDataset):
    """Dataset for retrieval task"""

    def __init__(self, cfg, dataset_name, splits, dataset_type='train'):
        super().__init__(cfg, dataset_name, splits, dataset_type)
        
        # dataset
        dataset = get_dataset(dataset_name, cfg)
        self.texts = {}
        if dataset_name in ['Clotho', 'AudioCaps']: # Used audio retrieval
            for split in splits:
                self.audiofiles += dataset.dataset[split]['audio']
                self.texts.update(dataset.dataset[split]['text'])
        elif dataset_name == 'ESC50': # Used for zero-shot classification
            audiofiles = dataset.dataset['train']['audio'] # train, valid, test are the same
            texts = dataset.dataset['train']['text']
            self.audiofiles = [audiofile for audiofile in audiofiles 
                               for split in splits if str(audiofile.stem).startswith(f'{split}-')]
            self.texts = texts
            self.labels_dict = dataset.labels_dict
            all_texts = [f"This is the sound of {t}" for t in self.labels_dict.keys()]
            for t in all_texts:
                text = self.tokenize(t)
                if self.text_embed is None: self.text_embed = {k: [] for k in text.keys()}
                self.text_embed['input_ids'].append(text['input_ids'])
                self.text_embed['attention_mask'].append(text['attention_mask'])
            self.text_embed = {k: torch.stack(v, dim=0) for k, v in self.text_embed.items()}
        elif dataset_name in ['sClotho', 'sAudioCaps', 'sFreesound']: # Used audio retrieval
            for split in splits:
                self.audiofiles += dataset.dataset[split]['audio']
        else: 
            raise ValueError(f"Unknown dataset: {dataset_name}")
        

    def __getitem__(self, idx):
        """
        Read waveform from the dataset
        """

        # process audio
        audiofile = self.audiofiles[idx]
        audio, sr = torchaudio.load(audiofile)
        audio = audio[[0]]
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        if audio.shape[-1] <= self.chunklen:
            longer = torch.tensor([False])
            if audio.shape[-1] < self.chunklen:
                audio = self.data_filling(audio)
            if self.cfg.data.truncation == 'fusion':
                # audio = torch.cat([audio, audio, audio, audio], dim=0)
                audio = audio.repeat(4, 1)
        else: 
            longer = torch.tensor([True])
            if self.cfg.data.truncation == 'fusion':
                audio = self.data_truncation_fusion(audio)
            elif self.cfg.data.truncation == 'rand_trunc':
                i = np.random.default_rng(seed=idx).integers(low=0, high=audio.shape[-1] - self.chunklen)
                audio = audio[:, i:i+self.chunklen]
            else: raise ValueError(f"Unknown truncation method: {self.cfg.data.truncation}")

        #### Zero-shot Classification ####
        if self.dataset_name in ['ESC50']:
            raw_text = self.texts[audiofile.name]
            text = self.labels_dict[raw_text.replace('_', ' ')]
        #### Audio Retrieval ####
        # Clotho and AudioCaps dataset has 5 captions for each audio file
        elif self.dataset_name in ['Clotho', 'AudioCaps', 'sClotho', 'sAudioCaps', 'Freesound', 'sFreesound']:
            if self.dataset_name in ['Clotho', 'AudioCaps']:
                raw_text = self.texts[audiofile.name]
            elif self.dataset_name in ['sClotho', 'sAudioCaps']:
                metafile = str(audiofile).replace('/audio/', '/metadata/').replace('.flac', '.json')
                with open(metafile, 'r') as f:
                    metadata = json.load(f)
                raw_text = metadata['spatialized_caption']
            if self.dataset_type == 'train':
                raw_text = random.choice(raw_text)
                text = self.tokenize(raw_text)
            else: 
                text = [self.tokenize(t) for t in raw_text]
                text = {k: torch.stack([t[k] for t in text], dim=0) for k in text[0].keys()}
        else: raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        sample = {
            'audiofile': audiofile.stem,
            'audio': audio,
            'raw_text': raw_text,
            'text': text,
            'longer': longer
        }
        return sample
    


class sCLAPDataset(BaseDataset):
    """Dataset for retrieval task"""

    def __init__(self, cfg, dataset_name, splits=None, dataset_type='train'):
        super().__init__(cfg, dataset_name, splits, dataset_type)
        
        # dataset
        dataset = get_dataset(dataset_name, cfg)
        print(f"dataset_name: {dataset_name}")
        if dataset_name in ['sClotho', 'sAudioCaps', 'sFreesound', 
                            'sClotho_ColRIR', 'sAudioCaps_ColRIR',
                            'sClotho_ColRIR_New', 'sAudioCaps_ColRIR_New',]: # Used audio retrieval
            for split in splits:
                self.audiofiles += dataset.dataset[split]['audio']
                print(split, len(self.audiofiles))
        elif dataset_name in ['Clotho', 'AudioCaps'] and dataset_type == 'test':
            self.texts = {}
            for split in splits:
                self.audiofiles += dataset.dataset[split]['audio']
                self.texts.update(dataset.dataset[split]['text'])   
        else: 
            raise ValueError(f"Unknown dataset: {dataset_name}")
        direction_texts = ['The sound is coming from the east.',
                           'The sound is coming from the northeast.',
                           'The sound is coming from the north.',
                           'The sound is coming from the northwest.',
                           'The sound is coming from the west.',
                           'The sound is coming from the southwest.',
                           'The sound is coming from the south.',
                           'The sound is coming from the southeast.',]
        for t in direction_texts:
            text = self.tokenize(t)
            if self.text_embed is None: self.text_embed = {k: [] for k in text.keys()}
            self.text_embed['input_ids'].append(text['input_ids'])
            self.text_embed['attention_mask'].append(text['attention_mask'])
        self.text_embed = {k: torch.stack(v, dim=0) for k, v in self.text_embed.items()}
        self.direction_label_dict = {direction: i for i, direction in enumerate(direction_texts)}


    def __getitem__(self, idx):
        """
        Read waveform from the dataset
        """

        # process audio
        audiofile = self.audiofiles[idx]
        audio, sr = torchaudio.load(audiofile) # 24000 Hz for sClotho and sAudioCaps
        # only use the first channel while evaluating the semantic ability (from semantic branch only)
        if self.dataset_type == 'test' and self.dataset_name in ['Clotho', 'AudioCaps']:
            audio = audio[[0]]
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            azi = np.random.randint(-180, 180)
            ele = np.random.randint(-90, 90)
            w = audio
            x = np.cos(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
            y = np.sin(np.deg2rad(azi)) * np.cos(np.deg2rad(ele))
            z = np.sin(np.deg2rad(ele))
            audio = torch.concat((w, y * audio, z * audio, x * audio), axis=0)

        if audio.shape[-1] <= self.chunklen:
            longer = torch.tensor([False])
            if audio.shape[-1] < self.chunklen:
                audio = self.data_filling(audio)
            if self.cfg.data.truncation == 'fusion': 
                audio1 = audio[[0]].repeat(4, 1) 
            else: audio1 = audio[[0]]
            audio2 = audio
        else: 
            longer = torch.tensor([True])
            audio1 = audio[[0]]
            i = np.random.default_rng(seed=idx).integers(low=0, high=audio.shape[-1] - self.chunklen)
            if self.cfg.data.truncation == 'fusion':
                audio1 = self.data_truncation_fusion(audio1)
            elif self.cfg.data.truncation == 'rand_trunc':
                audio1 = audio1[:, i:i+self.chunklen]
            # audio2 = self.shrink_poly(audio[None]).squeeze(0)
            audio2 = audio[:, i:i+self.chunklen]

        #### Audio Retrieval ####
        # sClotho and sAudioCaps dataset has 5 captions for each audio file
        if self.dataset_name in ['sClotho', 'sAudioCaps', 'sFreesound', 
                                 'sClotho_ColRIR', 'sAudioCaps_ColRIR',
                                 'sClotho_ColRIR_New', 'sAudioCaps_ColRIR_New',]:
            metafile = str(audiofile).replace('/audio/', '/metadata/').replace('.flac', '.json')
            with open(metafile, 'r') as f:
                metadata = json.load(f)
            spatialized_caption = metadata['spatialized_caption']
            # spatialized_caption = ['north'] * 5 # default spatialized caption
            caption = metadata['caption']
            azi, ele = metadata['azi'], metadata['ele']
            azi_f = _to_angle(azi)
            ele_f = _to_angle(ele)
            azi = torch.deg2rad(torch.tensor(azi_f, dtype=torch.float32))
            ele = torch.deg2rad(torch.tensor(ele_f, dtype=torch.float32))
            x, y, z = torch.cos(ele) * torch.cos(azi), torch.cos(ele) * torch.sin(azi), torch.sin(ele)
            direction = metadata['direction']
            # direction = 'The sound is coming from the east.' # default direction
        # only use the first channel while evaluating the semantic ability (from semantic branch only)
        elif self.dataset_name in ['Clotho', 'AudioCaps'] and self.dataset_type == 'test':
            if -22.5 < azi <= 22.5: direction = 'south'
            elif 22.5 < azi <= 67.5: direction = 'southeast'
            elif 67.5 < azi <= 112.5: direction = 'east'
            elif 112.5 < azi <= 157.5: direction = 'northeast'
            elif -22.5 > azi >= -67.5: direction = 'southwest'
            elif -67.5 > azi >= -112.5: direction = 'west'
            elif -112.5 > azi >= -157.5: direction = 'northwest'
            else: direction = 'north'
            direction = f'The sound is coming from the {direction}.'
            caption = self.texts[audiofile.name]
            spatialized_caption = [f'The sound "{caption}" is coming from the {direction}.']
        else: raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        if self.dataset_type == 'train':
            i = random.randint(0, len(caption)-1)
            caption = caption[i]
            spatialized_caption = spatialized_caption[i]
            text = self.tokenize(caption)
            text_comb = self.tokenize(spatialized_caption)
        else:
            text = [self.tokenize(t) for t in caption]
            text_comb = [self.tokenize(t) for t in spatialized_caption]
            text = {k: torch.stack([t[k] for t in text], dim=0) for k in text[0].keys()}
            text_comb = {k: torch.stack([t[k] for t in text_comb], dim=0) for k in text_comb[0].keys()}

        sample = {
            'audiofile': audiofile.stem,
            'audio4sed': audio1, # for the CLAP audio-SED encoder
            'audio4doa': audio2, # for the CLAP audio-DOA encoder
            'spatialized_caption': spatialized_caption,
            'ori_caption': caption,
            'text_comb': text_comb,
            'text_sed': text,
            'cls_doa': self.direction_label_dict[direction],
            'cart_doa': torch.tensor([x, y, z]),
            'longer': longer,
        }
        return sample

def _to_angle(v):
    if isinstance(v, str):
        s = v.strip()
        if s == '':
            return 0.0
        try:
            return float(s)
        except ValueError:
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, (list, tuple)):
                    return float(obj[0])
                return float(obj)
            except Exception:
                raise ValueError(f"无效角度值: {v}")
    return float(v)

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
        self.metafiles = []
        # dataset
        dataset = get_dataset(dataset_name, cfg)
        print(f"dataset_name: {dataset_name}")
        if dataset_name in ['sClotho','stClotho', 'sAudioCaps', 'sFreesound', 
                            'sClotho_ColRIR', 'sAudioCaps_ColRIR',
                            'sClotho_ColRIR_New', 'sAudioCaps_ColRIR_New',]: # Used audio retrieval
            for split in splits:
                self.audiofiles += dataset.dataset[split]['audio']
                if 'metadata' in dataset.dataset[split]:
                    self.metafiles += dataset.dataset[split]['metadata']
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
        """Read waveform from the dataset."""

        def _process_one_audio(audio, sr, crop_i=None):
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)

            if audio.shape[-1] <= self.chunklen:
                longer = torch.tensor([False])
                if audio.shape[-1] < self.chunklen:
                    audio = self.data_filling(audio)
                if self.cfg.data.truncation == 'fusion':
                    audio1 = audio[[0]].repeat(4, 1)
                else:
                    audio1 = audio[[0]]
                audio2 = audio
            else:
                longer = torch.tensor([True])
                audio1 = audio[[0]]
                if crop_i is None:
                    crop_i = int(np.random.default_rng(seed=idx).integers(low=0, high=audio.shape[-1] - self.chunklen))
                if self.cfg.data.truncation == 'fusion':
                    audio1 = self.data_truncation_fusion(audio1)
                elif self.cfg.data.truncation == 'rand_trunc':
                    audio1 = audio1[:, crop_i:crop_i + self.chunklen]
                else:
                    raise ValueError(f"Unknown truncation method: {self.cfg.data.truncation}")
                audio2 = audio[:, crop_i:crop_i + self.chunklen]

            return audio1, audio2, longer

        def _parse_caption_and_doa(metafile_path):
            with open(metafile_path, 'r') as f:
                metadata = json.load(f)

            cart_doa = None

            if isinstance(metadata, dict) and 'audio_segments' in metadata:
                temporal_spatial_caption = metadata.get('temporal_spatial_caption', None)
                if temporal_spatial_caption is None:
                    raise KeyError(f"Missing temporal_spatial_caption in {metafile_path}")
                if not isinstance(temporal_spatial_caption, list) or len(temporal_spatial_caption) == 0:
                    raise KeyError(f"Invalid temporal_spatial_caption in {metafile_path}")

                segments = metadata.get('audio_segments', [])
                if not isinstance(segments, list) or len(segments) == 0:
                    raise KeyError(f"Invalid audio_segments in {metafile_path}")

                if len(segments) < 2:
                    raise KeyError(f"Need >=2 audio_segments in {metafile_path}")
                seg0_meta = segments[0].get('metadata', {})
                seg1_meta = segments[1].get('metadata', {})
                c0 = seg0_meta.get('caption', None)
                c1 = seg1_meta.get('caption', None)
                if not isinstance(c0, list) or not isinstance(c1, list) or len(c0) == 0 or len(c1) == 0:
                    raise KeyError(f"Missing/invalid caption list in audio_segments[0/1].metadata of {metafile_path}")

                k = min(len(c0), len(c1), len(temporal_spatial_caption))
                caption = []
                for i in range(k):
                    a = str(c0[i]).strip().rstrip('.!?')
                    b = str(c1[i]).strip()
                    out = f"{a}, then {b}"
                    out = out.strip()
                    if out and out[-1] not in '.!?':
                        out += '.'
                    caption.append(out)

                spatialized_caption = temporal_spatial_caption

                n_events = getattr(self.cfg.model, 'n_events', 2)
                if len(segments) < n_events:
                    raise KeyError(f"Need >= {n_events} audio_segments in {metafile_path}")

                cart_doa_list = []
                for j in range(n_events):
                    seg_j_meta = segments[j].get('metadata', {})
                    azi_j, ele_j = seg_j_meta.get('azi', None), seg_j_meta.get('ele', None)
                    direction_j = seg_j_meta.get('direction', None)
                    if azi_j is None or ele_j is None or direction_j is None:
                        raise KeyError(f"Missing azi/ele/direction in audio_segments[{j}].metadata of {metafile_path}")
                    azi_f = _to_angle(azi_j)
                    ele_f = _to_angle(ele_j)
                    azi_t = torch.deg2rad(torch.tensor(azi_f, dtype=torch.float32))
                    ele_t = torch.deg2rad(torch.tensor(ele_f, dtype=torch.float32))
                    xj = torch.cos(ele_t) * torch.cos(azi_t)
                    yj = torch.cos(ele_t) * torch.sin(azi_t)
                    zj = torch.sin(ele_t)
                    cart_doa_list.append(torch.stack([xj, yj, zj]))
                cart_doa = torch.stack(cart_doa_list, dim=0)

                seg_idx = np.random.randint(0, len(segments)) if self.dataset_type == 'train' else 0
                seg_meta = segments[seg_idx].get('metadata', {})
                azi, ele = seg_meta.get('azi', None), seg_meta.get('ele', None)
                direction = seg_meta.get('direction', None)
                if azi is None or ele is None or direction is None:
                    raise KeyError(f"Missing azi/ele/direction in audio_segments[{seg_idx}].metadata of {metafile_path}")
            else:
                spatialized_caption = metadata['spatialized_caption']
                caption = metadata['caption']
                azi, ele = metadata['azi'], metadata['ele']
                direction = metadata['direction']

            azi_f = _to_angle(azi)
            ele_f = _to_angle(ele)
            azi_t = torch.deg2rad(torch.tensor(azi_f, dtype=torch.float32))
            ele_t = torch.deg2rad(torch.tensor(ele_f, dtype=torch.float32))
            x = torch.cos(ele_t) * torch.cos(azi_t)
            y = torch.cos(ele_t) * torch.sin(azi_t)
            z = torch.sin(ele_t)

            if cart_doa is None:
                cart_doa = torch.stack([x, y, z], dim=0).unsqueeze(0)

            return caption, spatialized_caption, direction, cart_doa

        audioitem = self.audiofiles[idx]
        metaitem = self.metafiles[idx] if len(self.metafiles) > idx else None
        is_triplet = isinstance(audioitem, (tuple, list)) and len(audioitem) == 3

        # -------- audio --------
        if not is_triplet:
            audiofile = audioitem
            audio, sr = torchaudio.load(audiofile)

            # semantic-only eval branch for Clotho/AudioCaps
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

            audio1, audio2, longer = _process_one_audio(audio, sr)
        else:
            pos_path, neg_t_path, neg_s_path = audioitem
            audiofile = pos_path

            loaded = []
            lengths = []
            srs = []
            for p in (pos_path, neg_t_path, neg_s_path):
                a, sr = torchaudio.load(p)
                loaded.append(a)
                lengths.append(a.shape[-1])
                srs.append(sr)

            if any(L <= self.chunklen for L in lengths):
                crop_i = 0
            else:
                max_start = min(L - self.chunklen for L in lengths)
                crop_i = int(np.random.default_rng(seed=idx).integers(low=0, high=max_start + 1))

            audio1_list, audio2_list, longer_list = [], [], []
            for a, sr in zip(loaded, srs):
                a1, a2, lg = _process_one_audio(a, sr, crop_i=crop_i)
                audio1_list.append(a1)
                audio2_list.append(a2)
                longer_list.append(lg)

            audio1 = torch.stack(audio1_list, dim=0)
            audio2 = torch.stack(audio2_list, dim=0)
            longer = torch.stack(longer_list, dim=0)

        # -------- metadata/text --------
        if self.dataset_name in ['sClotho','stClotho', 'sAudioCaps', 'sFreesound',
                                 'sClotho_ColRIR', 'sAudioCaps_ColRIR',
                                 'sClotho_ColRIR_New', 'sAudioCaps_ColRIR_New',]:
            if not is_triplet:
                metafile = str(audiofile).replace('/audio/', '/metadata/').replace('.flac', '.json')
                if metaitem is not None:
                    metafile = str(metaitem)
                caption, spatialized_caption, direction, cart_doa = _parse_caption_and_doa(metafile)
            else:
                # Triplet: each audio reads its OWN metadata json.
                if isinstance(metaitem, (tuple, list)) and len(metaitem) == 3:
                    meta_paths = [str(p) for p in metaitem]
                else:
                    meta_paths = [
                        str(pos_path).replace('/audio/', '/metadata/').replace('.flac', '.json'),
                        str(neg_t_path).replace('/audio/', '/metadata/').replace('.flac', '.json'),
                        str(neg_s_path).replace('/audio/', '/metadata/').replace('.flac', '.json'),
                    ]

                parsed = [_parse_caption_and_doa(p) for p in meta_paths]
                captions = [p[0] for p in parsed]
                spatial_caps = [p[1] for p in parsed]
                directions = [p[2] for p in parsed]
                cart_doas = [p[3] for p in parsed]
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
            cart_doa = torch.zeros((1, 3), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if not is_triplet:
            if self.dataset_type == 'train':
                cap_i = random.randint(0, len(caption) - 1)
                caption = caption[cap_i]
                spatialized_caption = spatialized_caption[cap_i]
                text = self.tokenize(caption)
                text_comb = self.tokenize(spatialized_caption)
            else:
                text = [self.tokenize(t) for t in caption]
                text_comb = [self.tokenize(t) for t in spatialized_caption]
                text = {k: torch.stack([t[k] for t in text], dim=0) for k in text[0].keys()}
                text_comb = {k: torch.stack([t[k] for t in text_comb], dim=0) for k in text_comb[0].keys()}
        else:
            if self.dataset_type == 'train':
                min_len = min(len(c) for c in captions)
                min_len = min(min_len, *(len(sc) for sc in spatial_caps))
                if min_len <= 0:
                    raise KeyError("Empty caption list in triplet metadata")
                cap_i = random.randint(0, min_len - 1)
                caption = [c[cap_i] for c in captions]
                spatialized_caption = [sc[cap_i] for sc in spatial_caps]

                text_list = [self.tokenize(c) for c in caption]
                text_comb_list = [self.tokenize(c) for c in spatialized_caption]
                text = {k: torch.stack([t[k] for t in text_list], dim=0) for k in text_list[0].keys()}
                text_comb = {k: torch.stack([t[k] for t in text_comb_list], dim=0) for k in text_comb_list[0].keys()}
            else:
                k = min(len(c) for c in captions)
                k = min(k, *(len(sc) for sc in spatial_caps))
                captions_k = [c[:k] for c in captions]
                spatial_k = [sc[:k] for sc in spatial_caps]

                text_per = []
                text_comb_per = []
                for c_list, sc_list in zip(captions_k, spatial_k):
                    t = [self.tokenize(x) for x in c_list]
                    tc = [self.tokenize(x) for x in sc_list]
                    text_per.append({kk: torch.stack([x[kk] for x in t], dim=0) for kk in t[0].keys()})
                    text_comb_per.append({kk: torch.stack([x[kk] for x in tc], dim=0) for kk in tc[0].keys()})
                text = {kk: torch.stack([tp[kk] for tp in text_per], dim=0) for kk in text_per[0].keys()}
                text_comb = {kk: torch.stack([tp[kk] for tp in text_comb_per], dim=0) for kk in text_comb_per[0].keys()}
                caption = captions_k
                spatialized_caption = spatial_k

            direction = directions
            cart_doa = torch.stack(cart_doas, dim=0)

        sample = {
            'audiofile': audiofile.stem,
            'audio4sed': audio1,
            'audio4doa': audio2,
            'spatialized_caption': spatialized_caption,
            'ori_caption': caption,
            'text_comb': text_comb,
            'text_sed': text,
            'cls_doa': (torch.tensor([self.direction_label_dict[d] for d in direction], dtype=torch.long)
                        if isinstance(direction, (list, tuple)) else self.direction_label_dict[direction]),
            'cart_doa': cart_doa,
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from .component.model_utilities import MLPLayers
from .component.htsat import HTSAT_Swin_Transformer


class CLAP(nn.Module):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(CLAP, self).__init__()
        
        self.cfg = cfg
        self.enable_fusion = cfg.model.fusion.enable
        self.fusion_type = cfg.model.fusion.type
        self.audio_backbone = cfg.model.audio.backbone
        self.text_backbone = cfg.model.text.backbone
        
        if mlp_act == 'relu':
            self.mlp_act = nn.ReLU
        elif mlp_act == 'gelu':
            self.mlp_act = nn.GELU
        
        ####################### Audio Branch #######################
        self.audio_scalar = nn.BatchNorm2d(cfg.data.n_mels)
        if self.audio_backbone == 'HTSAT':
            self.audio_branch = HTSAT_Swin_Transformer(cfg, **cfg.model.audio.kwargs)
        else: raise NotImplementedError

        self.audio_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))

        ####################### Text Branch #######################
        if self.text_backbone == 'roberta':
            self.text_branch = RobertaModel.from_pretrained('roberta-base')
        else: raise NotImplementedError

        self.text_projection = nn.Sequential(
            nn.Linear(768, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))

        # ============================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        if cfg.ckpt_path is None:
            self.load_pretrained_weights(
                cfg.model.audio.ckpt_path, 
                cfg.model.text.ckpt_path)
    
    def encode_text(self, text):
        if self.text_backbone == 'roberta':
            text_output = self.text_branch(
                input_ids=text['input_ids'], 
                attention_mask=text['attention_mask']
                )['pooler_output']
        else: raise NotImplementedError
        return text_output
    
    def get_text_embedding(self, data):
        """Get the text embedding from the model

        """
        text_embeds = self.encode_text(data)
        text_embeds = self.text_projection(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def encode_audio(self, audio, longer_list=[]):
        return self.audio_branch(audio, longer_list)
    
    def get_audio_embedding(self, data, longer_list=[]):
        """Get the audio embedding from the model

        """
        data = self.audio_scalar(data.transpose(1, 3)).transpose(1, 3)
        audio_embeds = self.encode_audio(data, longer_list)['embedding']
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds
    
    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: torch.Tensor (batch_size, channel, frame, freq_bin)
            the mel spectrogram audio input
        text: torch.Tensor () // need to add
            the text token input
        """

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)

        return audio_embedding, text_embedding,

    def load_pretrained_weights(self, audio_path, text_path=None):
        """Load the pretrained weights for the audio and text encoder

        Parameters
        ----------
        audio_path: str
            the path to the audio encoder pretrained weights
        text_path: str
            the path to the text encoder pretrained weights
        """
        if audio_path is None:
            return

        all_keys = list(self.state_dict().keys())
        # Load the audio encoder (handle PyTorch 2.6+ weights_only default & LAION ckpt without 'state_dict')
        try:
            _raw_obj = torch.load(audio_path, map_location='cpu')
        except Exception:
            # retry with unsafe load only if user trusts source
            _raw_obj = torch.load(audio_path, map_location='cpu', weights_only=False)
        if isinstance(_raw_obj, dict) and 'state_dict' in _raw_obj:
            audio_ckpt = _raw_obj['state_dict']
        else:
            audio_ckpt = _raw_obj
        if 'HTSAT-fullset' in audio_path:
            # Load the HTS-AT model
            print('Loading HTS-AT model from {}'.format(audio_path))
            audio_ckpt = {k.replace('sed_model.', ''): v for k, v in audio_ckpt.items()}
            for key, value in self.audio_branch.state_dict().items():
                if 'mel_conv2d' in key: continue
                elif 'fusion_model' in key: continue
                if 'audio_branch.' + key in all_keys:
                    all_keys.remove('audio_branch.' + key)
                value.data.copy_(audio_ckpt[key])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys:
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(audio_ckpt['bn0.' + k])
        elif '630k-' in audio_path:
            print('Loading LAION-CLAP audio encoder from {}'.format(audio_path))
            ckpt = audio_ckpt
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            for key, value in self.state_dict().items():
                if key == 'logit_scale': 
                    value.data.copy_(ckpt['logit_scale_a'])
                elif 'audio_scalar' in key:
                    value.data.copy_(ckpt[key.replace('audio_scalar.', 'audio_branch.bn0.')])
                else: value.data.copy_(ckpt[key])
                all_keys.remove(key)
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path))     
                
        for key in all_keys:
            # if 'text_branch' in key: continue
            print('{} is not loaded.'.format(key))

        if text_path is None: return
        if '630k-' in text_path:
            print('Loading LAION-CLAP text encoder from {}'.format(text_path))
        else: ValueError('Unknown text encoder checkpoint: {}'.format(text_path))




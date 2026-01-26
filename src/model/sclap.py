import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from .component.model_utilities import MLPLayers
from .component.seld import EINV2_HTSAT
from .component.htsat import HTSAT_Swin_Transformer

torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class ModalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ModalityClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, 1)
        )
        
    def forward(self, x, alpha=1.0):
        x = GradientReversalFunction.apply(x, alpha)
        x = self.classifier(x)
        return x
    


class sCLAP(nn.Module):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP, self).__init__()

        self.mel_bins = cfg.data.n_mels
        self.sed_in_ch, self.doa_in_ch = 1, 7

        self.cfg = cfg
        self.n_events = getattr(cfg.model, 'n_events', 1)
        self.enable_fusion = cfg.model.fusion.enable
        self.fusion_type = cfg.model.fusion.type
        self.audio_backbone = cfg.model.audio.backbone
        self.text_backbone = cfg.model.text.backbone
        
        if mlp_act == 'relu':
            self.mlp_act = nn.ReLU
        elif mlp_act == 'gelu':
            self.mlp_act = nn.GELU

        ####################### Audio Branch #######################
        self.audio_scalar = nn.ModuleList(
            [nn.BatchNorm2d(self.mel_bins) for _ in range(self.doa_in_ch)])
        self.audio_branch = None

        self.fc_doa = nn.Sequential(
            nn.Linear(joint_embed_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, 3 * self.n_events),
            nn.Tanh()
        )

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

    def encode_text(self, text):
        if self.text_backbone == 'roberta':
            text_output = self.text_branch(
                input_ids=text['input_ids'], 
                attention_mask=text['attention_mask']
                )['pooler_output']
            text_output = self.text_projection(text_output)
            # text_output = F.normalize(text_output, dim=-1)
        else: raise NotImplementedError
        return text_output
    
    def get_text_embedding(self, data):
        """Get the text embedding from the model

        """
        
        text_comb_embeds = self.encode_text(data['text_comb'])
        text_sed_embeds = self.encode_text(data['text'])

        return [text_comb_embeds, text_sed_embeds]

    def encode_audio(self):
        raise NotImplementedError

    def get_audio_embedding(self):
        """Get the audio embedding from the model

        """
        raise NotImplementedError

    def load_pretrained_weights(self):
        raise NotImplementedError


class sCLAP_Single(sCLAP):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP_Single, self).__init__(cfg, joint_embed_dim, mlp_act)

        ####################### Audio Branch #######################
        if self.audio_backbone == 'HTSAT':
            self.audio_branch = HTSAT_Swin_Transformer(cfg, self.doa_in_ch, 
                                                       **cfg.model.audio.kwargs)
        else: raise NotImplementedError

        self.audio_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        # # Audio SED
        # self.audio_sed_projection = nn.Sequential(
        #     nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))
        # # Audio DOA
        # self.audio_doa_projection = nn.Sequential(
        #     nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))

        if cfg.ckpt_path is None:
            audio_ckpt_path = cfg.model.audio.ckpt_path
            if isinstance(audio_ckpt_path, list):
                audio_ckpt_path = audio_ckpt_path[0]
            text_ckpt_path = cfg.model.text.ckpt_path
            self.load_pretrained_weights(audio_ckpt_path, text_ckpt_path)
    
    def encode_audio(self, audio, longer_list=[]):
        return self.audio_branch(audio, longer_list)
    
    def get_audio_embedding(self, data, longer_list=[]):
        """Get the audio embedding from the model

        """
        audio = data['audio4doa']
        # Compute scalar
        audio = audio.transpose(1, 3)
        for nch in range(audio.shape[-1]):
            audio[..., [nch]] = self.audio_scalar[nch](audio[..., [nch]])
        audio = audio.transpose(1, 3)

        audio = self.encode_audio(audio, longer_list)
        audio_embeds = self.audio_projection(audio['embedding'])
        # audio_embeds = F.normalize(audio_embeds, dim=-1)

        return [audio_embeds, audio_embeds, audio_embeds]

        # audio_sed_embeds = self.audio_sed_projection(audio['embedding'])
        # audio_sed_embeds = F.normalize(audio_sed_embeds, dim=-1)
        # audio_doa_embeds = self.audio_doa_projection(audio['embedding'])
        # audio_doa_embeds = F.normalize(audio_doa_embeds, dim=-1)
        
        # return [audio_embeds, audio_sed_embeds, audio_doa_embeds]
    
    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the sCLAP"""

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)
        doa = self.fc_doa(audio_embedding[2])
        if doa.dim() == 2:
            b = doa.shape[0]
            doa = doa.view(b, self.n_events, 3)

        return [audio_embedding, text_embedding, doa]
    
    def load_pretrained_weights(self, audio_path, text_path=None):
        """Load the pretrained weights for the audio and text encoder"""

        if audio_path is None:
            return
        all_keys = list(self.state_dict().keys())
        ckpt = torch.load(audio_path, map_location='cpu')['state_dict']
        if 'ACCDOA' in audio_path or 'SEDDOA' in audio_path: # Single Branch Model {ACCDOA, mACCDOA, SEDDOA}
            print('Loading audio encoder from {}'.format(audio_path))
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for k, v in self.audio_branch.state_dict().items():
                if any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                if 'audio_branch.' + k in all_keys: 
                    all_keys.remove('audio_branch.' + k) 
                v.data.copy_(ckpt['encoder.' + k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])
        elif 'HTSAT-fullset' in audio_path: # HTSAT Model
            print('Loading HTSAT model from {}'.format(audio_path))
            ckpt = {k.replace('sed_model.', ''): v for k, v in ckpt.items()}
            for k, v in self.audio_branch.state_dict().items():
                if any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                elif k == 'patch_embed.proj.weight':
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    paras = ckpt[k].repeat(1, self.doa_in_ch, 1, 1) / self.doa_in_ch
                    v.data.copy_(paras)
                else:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    v.data.copy_(ckpt[k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['bn0.' + k[2:]])
        elif '630k-' in audio_path:
            print('Loading LAION-CLAP audio encoder from {}'.format(audio_path))
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            for key, value in self.state_dict().items():
                if key == 'logit_scale': 
                    value.data.copy_(ckpt['logit_scale_a'])
                elif key == 'audio_branch.patch_embed.proj.weight':
                    paras = ckpt[key].repeat(1, self.doa_in_ch, 1, 1) / self.doa_in_ch
                    value.data.copy_(paras)
                elif 'audio_scalar' in key: continue
                elif 'doa' in key: continue
                else: value.data.copy_(ckpt[key])
                all_keys.remove(key)
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['audio_branch.bn0.' + k[2:]])
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path))     
                
        for key in all_keys:
            # if 'text_branch' in key: continue
            print(f'{key} not loaded.')

        if text_path is None: return
        if '630k-' in text_path:
            print('Loading LAION-CLAP text encoder from {}'.format(text_path))
        else: ValueError('Unknown text encoder checkpoint: {}'.format(text_path))


#Temporal embedding encoder
class TemporalAudioEncoder(nn.Module):

    def __init__(self, embedding_dim=512, num_temporal_steps=2, num_heads=8, dim_feedforward=2048, num_layers=2):
        super().__init__()
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_temporal_steps, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, audio_embeddings):
        # audio_embeddings: (B, T, D)
        B, T, D = audio_embeddings.size()
        temporal_pos_embedding = self.temporal_pos_embedding[:, :T, :]  # (1, T, D)
        x = audio_embeddings + temporal_pos_embedding  # (B, T, D)

        x = self.transformer_encoder(x)  # (B, T, D)
        x = self.ln(x)  # (B, T, D)

        x = x.mean(dim=1)  # (B, D)
        temporal_embeds= self.mlp(x)  # (B, D)

        return temporal_embeds      

class sCLAP_Dual(sCLAP):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP_Dual, self).__init__(cfg, joint_embed_dim, mlp_act)

        ####################### Audio Branch #######################
        if self.audio_backbone == 'HTSAT':
            self.audio_branch = EINV2_HTSAT(cfg, self.sed_in_ch, self.doa_in_ch)
        else: raise NotImplementedError
        # Audio SED
        self.audio_sed_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        # Audio DOA
        self.audio_doa_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        
        #Conv for temporal audio
        self.audio_temporal_conv = nn.Sequential(
            nn.Conv2d(cfg.model.audio.output_dim, cfg.model.audio.output_dim, kernel_size=(self.audio_branch.sed_encoder.SF, 1)),
            nn.BatchNorm2d(cfg.model.audio.output_dim),
            self.mlp_act(),
        )

       
        #Audio Temporal
        self.audio_temporal_encoder = TemporalAudioEncoder(
            embedding_dim=joint_embed_dim,
            num_temporal_steps=2,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=2
        )

         #Audio Temporal Projection 
        self.audio_temporal_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        
        #Final Audio Projection
        self.final_audio_projection= nn.Sequential(
            nn.Linear(joint_embed_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        
        #Modality Classifier
        self.modality_classifier = ModalityClassifier(joint_embed_dim)

        # self.audio_projection = nn.Sequential(
        #     # nn.Linear(cfg.model.audio.output_dim*2, joint_embed_dim),
        #     nn.Linear(joint_embed_dim * 2, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))

        # ============================================================
        self.weights = nn.Parameter(torch.ones([joint_embed_dim]))

        self.temporal_alpha = nn.Parameter(torch.full((joint_embed_dim,), 0.1))

        if cfg.ckpt_path is None: 
            self.load_pretrained_weights(cfg.model.audio.ckpt_path[0], 
                                         cfg.model.audio.ckpt_path[1], 
                                         cfg.model.text.ckpt_path)
        for stitch in self.audio_branch.stitch1:
            stitch.weight.data[:, 0, 0].fill_(1)
            stitch.weight.data[:, 0, 1].zero_()
            stitch.weight.data[:, 1, 0].zero_()
            stitch.weight.data[:, 1, 1].fill_(1)
        self.audio_branch.stitch1.requires_grad_(True)



    def encode_audio(self, audio1, audio2, longer_list=[]):
        return self.audio_branch(audio1, audio2, longer_list)
    
    def get_audio_embedding(self, data, longer_list=[]):
        """Get the audio embedding from the model

        """
        audio1, audio2 = data['audio4sed'], data['audio4doa']
        # Compute audio4sed scalar
        audio1 = self.audio_scalar[0](audio1.transpose(1, 3)).transpose(1, 3)
        # Compute audio4doa scalar
        audio2 = audio2.transpose(1, 3)
        for nch in range(audio2.shape[-1]):
            audio2[..., [nch]] = self.audio_scalar[nch](audio2[..., [nch]])
        audio2 = audio2.transpose(1, 3)

        audio_output = self.encode_audio(audio1, audio2, longer_list)

        audio_sed_embeds = self.audio_sed_projection(audio_output['sed_embedding'])
        audio_doa_embeds = self.audio_doa_projection(audio_output['doa_embedding'])
        audio_embeds = (audio_sed_embeds + self.weights * audio_doa_embeds)/2

        sed_feature_maps = audio_output['sed_feature_maps']  # (B, C, F, T)
        doa_feature_maps = audio_output['doa_feature_maps']  # (B, C, F, T)

        B, C, F, T = sed_feature_maps.shape

        sed_temporal_feat = self.audio_temporal_conv(sed_feature_maps).squeeze(2)
        doa_temporal_feat = self.audio_temporal_conv(doa_feature_maps).squeeze(2)

        T_sed = sed_temporal_feat.shape[-1]
        T_doa = doa_temporal_feat.shape[-1]


        sed_chunck1 = sed_feature_maps[..., :T_sed//2].mean(dim=(-1))
        sed_chunck2 = sed_feature_maps[..., T_sed//2:].mean(dim=(-1))

        doa_chunck1 = doa_feature_maps[..., :T_doa//2].mean(dim=(-1))
        doa_chunck2 = doa_feature_maps[..., T_doa//2:].mean(dim=(-1))
        chunck1_embeds = self.audio_temporal_projection(sed_chunck1) + \
                        self.weights * self.audio_temporal_projection(doa_chunck1)
        chunck2_embeds = self.audio_temporal_projection(sed_chunck2) + \
                        self.weights * self.audio_temporal_projection(doa_chunck2)

        temporal_seq = torch.stack([chunck1_embeds, chunck2_embeds], dim=1)  # (B, 2, D)
        audio_temporal_embeds = self.audio_temporal_encoder(temporal_seq) 

        audio_triplet_embeds = audio_embeds + self.temporal_alpha.unsqueeze(0) * self.final_audio_projection(audio_temporal_embeds)
        
        
        # audio_embeds = self.audio_projection(
        #     torch.cat([audio_sed_embeds, audio_doa_embeds], dim=-1))

        # audio_embeds = F.normalize(audio_embeds, dim=-1)
        # audio_sed_embeds = F.normalize(audio_sed_embeds, dim=-1)
        # audio_doa_embeds = F.normalize(audio_doa_embeds, dim=-1)
        
        return [audio_embeds, audio_sed_embeds, audio_doa_embeds, audio_temporal_embeds, audio_triplet_embeds]

    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the sCLAP"""

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)

        audio_embedding = [F.normalize(x, dim=-1) for x in audio_embedding]
        text_embedding = [F.normalize(x, dim=-1) for x in text_embedding]
        
        doa = self.fc_doa(audio_embedding[-1])
        if doa.dim() == 2:
            b = doa.shape[0]
            doa = doa.view(b, self.n_events, 3)

        return [audio_embedding, text_embedding, doa]
    
    def load_pretrained_weights(self, audio_path1, audio_path2, text_path):
        """Load the pretrained weights for the audio and text encoder

        Parameters
        ----------
        audio_path: str
            the path to the audio encoder pretrained weights
        text_path: str
            the path to the text encoder pretrained weights
        seld_path: str
            the path to the PSELDNets pretrained weights
        """
        if audio_path1 is None and audio_path2 is None:
            return

        all_keys = list(self.state_dict().keys())
        # Load pseldnets-EINV2 first 
        if audio_path1 and 'EINV2' in audio_path1:
            ckpt = torch.load(audio_path1, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            print('Loading PSELDNets pretrained weights from ', audio_path1)
            for k, v in self.audio_branch.state_dict().items():
                if k == 'sed_encoder.patch_embed.proj.weight':
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k)
                    paras = ckpt[k][:, :v.shape[1]] * 4
                    v.data.copy_(paras)
                elif any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                else:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    v.data.copy_(ckpt[k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])
        elif audio_path2 and 'ACCDOA' in audio_path2:
            ckpt = torch.load(audio_path2, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            print('Loading PSELDNets pretrained weights from ', audio_path2)
            for k, v in self.audio_branch.state_dict().items():
                if 'sed_encoder' in k: continue
                elif any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                elif 'doa_encoder' in k:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k)
                    v.data.copy_(ckpt[k[4:]])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])

        # Load the audio encoder from clap
        if audio_path1 and '630k-' in audio_path1:
            print('Loading LAION-CLAP audio encoder from {}'.format(audio_path1))
            ckpt = torch.load(audio_path1, map_location='cpu')['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            self.logit_scale.data.copy_(ckpt['logit_scale_a'])
            all_keys.remove('logit_scale')
            # Reload the audio encoder from LAION-CLAP
            for k, v in self.audio_branch.sed_encoder.state_dict().items():
                if 'audio_branch.sed_encoder.' + k in all_keys: 
                    all_keys.remove('audio_branch.sed_encoder.' + k)
                v.data.copy_(ckpt['audio_branch.' + k])
            # Load 'audio_projection' from LAION-CLAP into 'audio_sed_projection', 'audio_doa_projection'
            for k, v in self.audio_sed_projection.state_dict().items():
                if 'audio_sed_projection.' + k in all_keys: 
                    all_keys.remove('audio_sed_projection.' + k)
                v.data.copy_(ckpt['audio_projection.' + k])
            for k, v in self.audio_doa_projection.state_dict().items():
                if 'audio_doa_projection.' + k in all_keys: 
                    all_keys.remove('audio_doa_projection.' + k)
                v.data.copy_(ckpt['audio_projection.' + k])
            # Load the text encoder from LAION-CLAP
            for k, v in self.state_dict().items():
                if k in ckpt and 'audio_projection' not in k:
                    if k in all_keys: all_keys.remove(k)
                    v.data.copy_(ckpt[k])
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path1))

        for key in all_keys:
            # if 'text_branch' in key: continue
            print(f'{key} not loaded.')

        if text_path is None: return
        if audio_path1 and '630k-' in text_path:
            print('Loading LAION-CLAP text encoder from {}'.format(text_path))
        else: ValueError('Unknown text encoder checkpoint: {}'.format(text_path))

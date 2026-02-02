import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from rotary_embedding_torch import RotaryEmbedding

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



# RoPE Attenrion
class RotaryPositionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x):
        B, T, C = x.shape
   
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
      
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class RoPETransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotaryPositionAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    


#Temporal embedding encoder
class TemporalAudioEncoder(nn.Module):

    def __init__(self, embedding_dim=512, num_temporal_steps=2, num_heads=8, dim_feedforward=2048, num_layers=2):
        super().__init__()
        # self.temporal_pos_embedding = nn.Parameter(torch.randn(1, num_temporal_steps, embedding_dim))

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embedding_dim, 
        #     nhead=num_heads, 
        #     dim_feedforward=dim_feedforward,
        #     batch_first=True,
        #     dropout=0.1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layers = nn.ModuleList([
            RoPETransformerLayer(
                dim=embedding_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.1)for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, audio_embeddings):
        # audio_embeddings: (B, T, D)
        # B, T, D = audio_embeddings.size()
        # # temporal_pos_embedding = self.temporal_pos_embedding[:, :T, :]  # (1, T, D)
        # # x = audio_embeddings + temporal_pos_embedding  # (B, T, D)
        x = audio_embeddings
        for layer in self.layers:
            x = layer(x)
        # x = self.transformer_encoder(x)  # (B, T, D)
        x = self.ln(x)  # (B, T, D)

        x = x.mean(dim=1)  # (B, D)
        temporal_embeds= self.mlp(x)  # (B, D)

        return temporal_embeds      
    
 #Bi-directional Cross-Attention Module 
 #Adapted from https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py   
 
class CrossAttention(nn.Module):
    """Cross-Attention between latents and input tokens -- returning the refined latents and tokens as tuple """
    def __init__(self, dim_sed, dim_doa, dim_attn, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.proj_sed_in = nn.Linear(dim_sed, dim_attn * 2, bias=qkv_bias)  # 'in-projection' for latents
        self.proj_doa_in = nn.Linear(dim_doa, dim_attn * 2, bias=qkv_bias)  # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        
        self.proj_sed_out = nn.Linear(dim_attn, dim_sed)             # 'out-projection' for latents
        self.proj_drop_sed = nn.Dropout(proj_drop)
        self.proj_doa_out = nn.Linear(dim_attn, dim_doa)             # 'out-projection' for patches/tokens
        self.proj_drop_doa = nn.Dropout(proj_drop)

    def forward(self, x_sed, x_doa):
        B, N_sed, _ = x_sed.shape
        _, N_doa, _ = x_doa.shape
        rv_sed = self.proj_sed_in(x_sed).reshape(B, N_sed, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_sed, v_sed = rv_sed.unbind(0)
        rv_doa = self.proj_doa_in(x_doa).reshape(B, N_doa, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_doa, v_doa = rv_doa.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_sed @ r_doa.transpose(-2, -1)) * self.scale  # query from latent, key from patches
        attn_T = attn.transpose(-2, -1)  # bidirectional attention, associated with the values from the query q

        attn_sed2doa = attn.softmax(dim=-1)  # softmax along patch token dimension
        attn_doa2sed = attn_T.softmax(dim=-1)  # softmax along latent token dimension
        attn_sed2doa = self.attn_drop(attn_sed2doa)
        attn_doa2sed = self.attn_dropT(attn_doa2sed)

        # Retrieve information form the patch tokens via latent query:
        x_sed_fused = (attn_sed2doa @ v_doa).transpose(1, 2).reshape(B, N_sed, self.dim_attn)
        x_sed_fused = self.proj_sed_out(x_sed_fused)
        x_sed_fused = self.proj_drop_sed(x_sed_fused)

        # Likewise, store information from the latents in the patch tokens via transposed attention:
        x_doa_fused = (attn_doa2sed @ v_sed).transpose(1, 2).reshape(B, N_doa, self.dim_attn)
        x_doa_fused = self.proj_doa_out(x_doa_fused)
        x_doa_fused = self.proj_drop_doa(x_doa_fused)

        return x_sed_fused, x_doa_fused
    
    
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
        
        #Conv for doa
        doa_feat_dim = self.audio_branch.doa_encoder.num_features 
        self.audio_doa_conv = nn.Sequential(
            nn.Conv2d(doa_feat_dim, 3, kernel_size=(self.audio_branch.doa_encoder.SF, 1)),
        )
        self.doa_act =nn.Tanh()
    
       
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
        
        # Bi-directional Cross-Attention Module
        self.cross_attention = CrossAttention(
            dim_sed=cfg.model.audio.output_dim,
            dim_doa=cfg.model.audio.output_dim,
            dim_attn=cfg.model.audio.output_dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.1,
            proj_drop=0.1
        )

        # self.audio_projection = nn.Sequential(
        #     # nn.Linear(cfg.model.audio.output_dim*2, joint_embed_dim),
        #     nn.Linear(joint_embed_dim * 2, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))

        # ============================================================
        self.weights = nn.Parameter(torch.ones([joint_embed_dim]))

        self.temporal_alpha = nn.Parameter(torch.tensor(1e-4))

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
        audio_embeds = audio_sed_embeds + self.weights * audio_doa_embeds

        sed_feature_maps = audio_output['sed_feature_maps']  # (B, C, F, T)
        doa_feature_maps = audio_output['doa_feature_maps']  # (B, C, F, T)

        B, C, F, T = sed_feature_maps.shape

        sed_temporal_feat = self.audio_temporal_conv(sed_feature_maps).squeeze(2)
        doa_temporal_feat = self.audio_temporal_conv(doa_feature_maps).squeeze(2)
        
        sed_in = sed_temporal_feat.permute(0, 2, 1) 
        doa_in = doa_temporal_feat.permute(0, 2, 1)  
        
        sed_delta, _ = self.cross_attention(sed_in, doa_in)
        
        fused_sed = sed_in + sed_delta
        fused_sed = self.audio_temporal_projection(fused_sed)
        audio_temporal_embeds = self.audio_temporal_encoder(fused_sed) 

        audio_triplet_embeds = audio_embeds + self.temporal_alpha * self.final_audio_projection(audio_temporal_embeds)
        
        
        # audio_embeds = self.audio_projection(
        #     torch.cat([audio_sed_embeds, audio_doa_embeds], dim=-1))

        # audio_embeds = F.normalize(audio_embeds, dim=-1)
        # audio_sed_embeds = F.normalize(audio_sed_embeds, dim=-1)
        # audio_doa_embeds = F.normalize(audio_doa_embeds, dim=-1)
        
        return [audio_embeds, audio_sed_embeds, audio_doa_embeds,audio_output['doa_feature_maps'],audio_temporal_embeds, audio_triplet_embeds]

    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the sCLAP"""

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)
        
        audio_embedding_norm = []
        for i, x in enumerate(audio_embedding):
            if i !=3: #doa feature maps
                audio_embedding_norm.append(F.normalize(x, dim=-1))
            else:
                audio_embedding_norm.append(x)
        
        audio_embedding = audio_embedding_norm

        text_embedding = [F.normalize(x, dim=-1) for x in text_embedding]
        
        doa_pred = self.audio_doa_conv(audio_embedding[3]).squeeze(2)  # (B, 3, T)
        doa_pred = self.doa_act(doa_pred)
        
        if self.n_events > 1:
            T = doa_pred.shape[-1]
            doa_event1 = doa_pred[..., :T//2].mean(dim=-1)
            doa_event2 = doa_pred[..., T//2:].mean(dim=-1)
            doa = torch.stack([doa_event1, doa_event2], dim=1)
        else:
            doa = doa_pred.mean(dim=-1)  # (B, 3)
        

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

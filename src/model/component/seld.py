from copy import deepcopy
import torch.nn as nn

from .htsat import HTSAT_Swin_Transformer
from .model_utilities import CrossStitch
from utils.utilities import get_pylogger

log = get_pylogger(__name__)


class EINV2_HTSAT(nn.Module):
    def __init__(self, cfg, sed_in_ch=1, doa_in_ch=7):
        super().__init__()
        
        kwargs = cfg.model.audio.kwargs

        # encoder
        self.sed_encoder = HTSAT_Swin_Transformer(cfg, sed_in_ch, **kwargs)
        cfg = deepcopy(cfg)
        cfg.model.fusion = {}
        self.doa_encoder = HTSAT_Swin_Transformer(cfg, doa_in_ch, **kwargs)
        
        # soft-parameter sharing
        num_feats = [kwargs['embed_dim'] * (2 ** i_layer) 
                     for i_layer in range(len(kwargs['depths']))]
        self.stitch1 = nn.ModuleList([CrossStitch(num_feat) for num_feat in num_feats])


    def forward(self, audio1, audio2, longer_list):
        """
        x: waveform, (batch_size, num_channels, time_frames, mel_bins)
        """

        # Rewrite the forward function of the encoders
        x_sed = self.sed_encoder.forward_patch(audio1, longer_list)
        x_doa = self.doa_encoder.forward_patch(audio2)
        for sed_layer, doa_layer, stitch in zip(self.sed_encoder.layers, 
                                                self.doa_encoder.layers, 
                                                self.stitch1):
            x_sed, x_doa = stitch(x_sed, x_doa)
            x_sed = sed_layer(x_sed)[0]
            x_doa = doa_layer(x_doa)[0]
        x_sed = self.sed_encoder.forward_reshape(x_sed)
        x_doa = self.doa_encoder.forward_reshape(x_doa)

        x_sed_latent = nn.functional.adaptive_avg_pool2d(x_sed, (1, 1)).squeeze((-1, -2))
        x_doa_latent = nn.functional.adaptive_avg_pool2d(x_doa, (1, 1)).squeeze((-1, -2))

        return {
            'sed_embedding': x_sed_latent,
            'doa_embedding': x_doa_latent,
            'sed_feature_maps': x_sed,
            'doa_feature_maps': x_doa,
        }
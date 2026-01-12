from numpy import pad
import torch
import torch.nn as nn

from .component.htsat import HTSAT_Swin_Transformer


class HTSAT_DOA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mel_bins = cfg.data.n_mels
        self.sed_in_ch, self.doa_in_ch = 1, 7
        self.cfg = cfg
        self.n_events = getattr(cfg.model, 'n_events', 2)
        
        ####################### Audio Branch #######################
        self.audio_scalar = nn.ModuleList(
            [nn.BatchNorm2d(self.mel_bins) for _ in range(self.doa_in_ch)])
        self.audio_branch = HTSAT_Swin_Transformer(cfg, self.doa_in_ch, 
                                                    **cfg.model.audio.kwargs)
        self.tscam_conv = nn.Conv2d(
            in_channels=self.audio_branch.num_features,
            out_channels=3 * self.n_events,
            kernel_size=(self.audio_branch.SF, self.audio_branch.ST),
        )
        self.final_act = nn.Tanh()
        # self.fc_doa = nn.Sequential(
        #     nn.Linear(cfg.model.audio.output_dim, 3),
        #     nn.Tanh()
        # )

        if cfg.ckpt_path is None:
            audio_ckpt_path = cfg.model.audio.ckpt_path
            if isinstance(audio_ckpt_path, list):
                audio_ckpt_path = audio_ckpt_path[0]
            self.load_pretrained_weights(audio_ckpt_path)
    
    def forward(self, data, longer_list=[]):
        """ Forward audio and text into the sCLAP

        """
        audio = data['audio4doa']
        # Compute scalar
        audio = audio.transpose(1, 3)
        for nch in range(audio.shape[-1]):
            audio[..., [nch]] = self.audio_scalar[nch](audio[..., [nch]])
        audio = audio.transpose(1, 3)

        audio = self.audio_branch(audio, longer_list)
        
        # return self.fc_doa(audio['embedding'])
        output = self.tscam_conv(audio['feature_maps']).squeeze()
        output = self.final_act(output)

        if self.n_events > 1:
            output = output.view(output.shape[0], self.n_events, 3)
            
        return output

    
    def load_pretrained_weights(self, audio_path):
        """Load the pretrained weights for the audio"""

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
        elif 'EINV2' in audio_path: # EINV2 Model
            print('Loading audio encoder model from {}'.format(audio_path))
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for k, v in self.audio_branch.state_dict().items():
                if any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                if 'audio_branch.' + k in all_keys: 
                    all_keys.remove('audio_branch.' + k) 
                v.data.copy_(ckpt['doa_encoder.' + k])
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
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path))     
                
        for key in all_keys:
            # if 'text_branch' in key: continue
            print(f'{key} not loaded.')


import numpy as np
import torch
from torch.nn import functional as F

from utils.config import get_afextractor
from torchmetrics import MeanMetric
from utils.utilities import get_pylogger
from .component.model_module import BaseModelModule
from evaluate.eval_zeroshot import evaluate_zero_shot


class CLAPModelModule(BaseModelModule):

    logging = get_pylogger(__name__)

    def __init__(self, cfg, steps, label_embed=None):
        super().__init__(cfg, steps, label_embed)

        from loss.loss_clap import CLAPLoss

        self.af_extractor = get_afextractor(cfg)
        self.loss = CLAPLoss(mlp_loss=cfg.model.mlp_loss)

        self.train_loss.update({'total_loss': MeanMetric()})
    
    def setup(self, stage):

        from model.clap import CLAP

        self.net = CLAP(self.cfg)
        if stage == 'fit' and self.cfg.compile:
            self.logging.info('Compiling model')
            self.net = torch.compile(self.net)
        if stage == 'test':
            pass
        self.logging.info("Number of parameters of the net: " + 
                          f"{sum(p.numel() for p in self.net.parameters())}")
        self.logging.info("Number of parameters of the audio encoder: " + 
                          f"{sum(p.numel() for p in self.net.audio_branch.parameters())}")
        self.logging.info("Number of parameters of the text encoder: " +
                            f"{sum(p.numel() for p in self.net.text_branch.parameters())}")

    def forward(self, audio, text, longer):
        if audio is None and text is None:
            raise ValueError('Both audio and text cannot be None')
        if audio is None:
            return self.net.get_text_embedding(text)
        
        audio = self.af_extractor(audio)
        longer_list = torch.tensor([])
        if self.cfg.model.get('fusion', {}).get('enable'):
            longer_list = torch.where(longer)[0]
            if len(longer_list) == 0 and self.training:
                longer_list = torch.tensor([np.random.choice(len(audio))])

        if text is None:
            return self.net.get_audio_embedding(audio, longer_list)
        
        if text['input_ids'].dim() == 3: # Only for Clotho and AudioCaps dataset in non-training stage
            # Clotho and AudioCaps dataset has 5 captions for each audio file
            # original text shape: (batch, n_captions, n_tokens)
            # we reshape it to (batch * n_captions, n_tokens)
            # text['input_ids'] = text['input_ids'].flatten(0, 1)
            # text['attention_mask'] = text['attention_mask'].flatten(0, 1)
            text = {k: v.flatten(0, 1) for k, v in text.items()}

        return self.net(audio, text, longer_list)          

    ############################## Training functions ##############################

    def training_step(self, batch_sample, batch_idx):
        audio_features, text_features = self.forward(batch_sample['audio'],
                                                      batch_sample['text'],
                                                      batch_sample['longer'])
        total_loss = self.loss(audio_features, text_features,
                               self.net.logit_scale)
        for key, loss in total_loss.items():
            self.train_loss[key].update(loss)
        return total_loss['total_loss']

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # NOTE: we clamp to 4.6052 = ln(100), as in the original paper.
        self.net.logit_scale.clamp_(0, np.log(100))
        if self.cfg.model.mlp_loss:
            raise NotImplementedError

    ############################## Validation functions ##############################
    def validation_step(self, batch_sample, batch_idx, dataloader_idx=0):
        if self.last_dataloader_idx != dataloader_idx:
            dataset_name = self.valid_dataset_names[self.last_dataloader_idx]
            self.compute_metrics(dataset_name)
            self.reset_system_output()
            self.last_dataloader_idx = dataloader_idx
        audio_features, text_features = self.forward(batch_sample['audio'],
                                                     batch_sample['text'],
                                                     batch_sample['longer'])
        self.system_output['all_audio_features'].append(audio_features)
        self.system_output['all_text_features'].append(text_features)
    
    def on_validation_epoch_end(self):
        self.compute_metrics(self.valid_dataset_names[-1])
        self.reset_system_output()
        self.last_dataloader_idx = 0

    ############################## Test functions ##############################
    def on_test_epoch_start(self):
        self.reset_system_output()
        if self.cfg.task == 'zero-shot-classification':
            if 'ground_truth' not in self.system_output:
                # [n_samples], only used for zero-shot classification
                self.system_output['ground_truth'] = []
            for k, v in self.label_embed.items():
                self.label_embed[k] = v.to(self.device)
            self.system_output['all_text_features'] = self.forward(None, self.label_embed, None)

    def test_step(self, batch_sample, batch_idx):
        if self.cfg.task == 'zero-shot-classification':
            audio_features = self.forward(batch_sample['audio'], 
                                          None, 
                                          batch_sample['longer'])
            self.system_output['ground_truth'].append(batch_sample['text'])
            self.system_output['all_audio_features'].append(audio_features)
        else: raise NotImplementedError
    
    def on_test_epoch_end(self):
        if self.cfg.task == 'zero-shot-classification':
            metrics = evaluate_zero_shot(
                self.logging, self.system_output, 
                self.test_dataset_names[-1])
            self.log_metrics(metrics, self.test_dataset_names[-1])
            self.reset_system_output()
        else: raise NotImplementedError
        
    def reset_system_output(self):
        self.system_output = {
            'all_audio_features': [], # [n_samples, n_dim]
            'all_text_features': [], # [n_samples, n_dim] or [n_classes, n_dim]
        }


class sCLAPModelModule(BaseModelModule):

    logging = get_pylogger(__name__)

    def __init__(self, cfg, steps, label_embed=None):
        super().__init__(cfg, steps, label_embed)

        from loss.loss_sclap import sCLAPLoss

        self.af_extractor1 = get_afextractor(cfg)
        self.af_extractor2 = get_afextractor(cfg, audio_feature='logmelIV')
        self.loss = sCLAPLoss(mlp_loss=cfg.model.mlp_loss, 
                              loss_weights=cfg.model.loss_weights)

        self.train_loss.update(
            {'total_loss': MeanMetric(), 'loss_doa': MeanMetric(),
            'loss_logit_semantic': MeanMetric(), 
            # 'loss_logit_doa': MeanMetric(),
            'loss_logit_spatial_semantic': MeanMetric(),
            'loss_logit_temporal': MeanMetric(),
            'loss_logit_spatial': MeanMetric(),
            'loss_logit_ts': MeanMetric(),
            'loss_modality': MeanMetric()
            }
        )
    
    def setup(self, stage):

        from model.sclap import sCLAP_Single, sCLAP_Dual
        
        if self.cfg.data.truncation != 'fusion' and self.cfg.model.get('fusion', {}).get('enable'):
            raise ValueError('The fusion model is only allowed for truncation-fusion')

        if self.cfg.model.audio.seld_method in ['ACCDOA', 'mACCDOA', 'SEDDOA']:
            assert self.cfg.model.get('fusion', {}).get('enable') == False, \
                'Fusion model is not allowed for single branch model'
            self.net = sCLAP_Single(self.cfg)
        elif self.cfg.model.audio.seld_method in ['EINV2']:
            self.net = sCLAP_Dual(self.cfg)

        if stage == 'fit' and self.cfg.compile:
            self.logging.info('Compiling model')
            self.net = torch.compile(self.net)
        if stage == 'test':
            pass
        self.logging.info("Number of parameters of the net: " + 
                          f"{sum(p.numel() for p in self.net.parameters())}")
        self.logging.info("Number of parameters of the audio encoder: " + 
                          f"{sum(p.numel() for p in self.net.audio_branch.parameters())}")
        self.logging.info("Number of parameters of the text encoder: " +
                            f"{sum(p.numel() for p in self.net.text_branch.parameters())}")

    def forward(self, audio, text, longer, normalize=True):
        if audio is None and text is None:
            raise ValueError('Both audio and text cannot be None')
        if audio is None:
            text_features = self.net.get_text_embedding(text)
            if normalize: 
                for idx in range(len(text_features)):
                    text_features[idx] = F.normalize(text_features[idx], dim=-1)
            return text_features
        
        audio['audio4sed'] = self.af_extractor1(audio['audio4sed'])
        audio['audio4doa'] = self.af_extractor2(audio['audio4doa'])
        longer_list = torch.tensor([])
        if self.cfg.model.get('fusion', {}).get('enable'):
            longer_list = torch.where(longer)[0]
            if len(longer_list) == 0 and self.training:
                longer_list = torch.tensor([np.random.choice(len(audio))])

        if text is None:
            audio_features = self.net.get_audio_embedding(audio, longer_list)
            if normalize:
                for idx in range(len(audio_features)):
                    audio_features[idx] = F.normalize(audio_features[idx], dim=-1)
            return audio_features
        
        for k in text.keys():
            if text[k]['input_ids'].dim() == 3:
                text[k] = {k1: v1.flatten(0, 1) for k1, v1 in text[k].items()}

        audio_features, text_features, doa = self.net(audio, text, longer_list)
        if normalize:
            for idx in range(len(audio_features)):
                audio_features[idx] = F.normalize(audio_features[idx], dim=-1)
            for idx in range(len(text_features)):
                text_features[idx] = F.normalize(text_features[idx], dim=-1)
        return audio_features, text_features, doa

    ############################## Training functions ##############################

    def training_step(self, batch_sample, batch_idx):
        is_triplet = (batch_sample['audio4sed'].dim() == 4)  # (B,3,C,T)
        if  batch_sample['audio4sed'].dim() == 4:  # (B,3,C,T)
            # Triplet batches from stClotho
            # - audio: flatten to candidates (B*3,...)
            # - text_sed: keep ONLY positives as anchors (B,...) for temporal hard-negative mining
            # - text_comb: keep flattened (B*3,...) so spatial negatives can be trained as ordinary samples
            batch_sample['audio4sed'] = batch_sample['audio4sed'].flatten(0, 1)
            batch_sample['audio4doa'] = batch_sample['audio4doa'].flatten(0, 1)
            batch_sample['longer'] = batch_sample['longer'].flatten(0, 1)
            batch_sample['cart_doa'] = batch_sample['cart_doa'].flatten(0, 1)

            text_sed = {k: v.flatten(0,1) for k, v in batch_sample['text_sed'].items()}
            text_comb = {k: v.flatten(0, 1) for k, v in batch_sample['text_comb'].items()}
        else:
            text_sed = batch_sample['text_sed']
            text_comb = batch_sample['text_comb']

        audio = {'audio4sed': batch_sample['audio4sed'], 
                 'audio4doa': batch_sample['audio4doa']}
        text = {'text': text_sed,
            'text_comb': text_comb}
        longer = batch_sample['longer']
        audio_features, text_features, doa = self.forward(audio, text, longer, normalize=False)
        # text_feature_doa = self.encode_direction_text()
        # cls_doa_gt = batch_sample['cls_doa']
        audio_emb = audio_features[-1] #audio_triplet_emb
        text_emb = text_features[0] #text_comb_emb

        modality_input = torch.cat([audio_emb, text_emb], dim=0)
        batch_size = audio_emb.size(0)
        labels_audio = torch.zeros(batch_size, 1, device=self.device)
        labels_text = torch.ones(batch_size, 1, device=self.device)
        modality_labels = torch.cat([labels_audio, labels_text], dim=0)
        modality_preds = self.net.modality_classifier(modality_input)
        modality_preds = torch.clamp(modality_preds, min=-50, max=50)
        modality_loss = F.binary_cross_entropy_with_logits(modality_preds, modality_labels)
        loss_weights = self.cfg.model.loss_weights
        if isinstance (loss_weights, list):
            w_modality = loss_weights[5] if len(loss_weights) > 5 else 0.05
        else:
            w_modality = 0.05

        modality_ramp_epochs = 4
        if modality_ramp_epochs <= 0:
            w_modality_eff = w_modality
        else:
            if self.current_epoch < 3:
                w_modality_eff = 0.0
            else:
                progress = float(self.current_epoch - 2) / float(modality_ramp_epochs)
                progress = max(0.0, min(1.0, progress))
                w_modality_eff = w_modality * progress    

        if self.trainer.world_size > 1:
            audio_features_temp, text_features_temp = [], []
            for audio_feature in audio_features:
                audio_feature = self.all_gather(audio_feature, sync_grads=True).flatten(0, 1)
                audio_features_temp.append(audio_feature)
            audio_features = audio_features_temp
            for text_feature in text_features:
                text_feature = self.all_gather(text_feature, sync_grads=True).flatten(0, 1)
                text_features_temp.append(text_feature)
            text_features = text_features_temp
            doa = self.all_gather(doa, sync_grads=True).flatten(0, 1)
            batch_sample['cart_doa'] = self.all_gather(
                batch_sample['cart_doa'], sync_grads=True
                ).flatten(0, 1)
            
        # text_features.append(text_feature_doa)
        total_loss = self.loss(audio_features, text_features, 
                               self.net.logit_scale,
                               [doa, batch_sample['cart_doa']],
                               epoch_it=self.current_epoch, is_triplet=is_triplet)
        total_loss['total_loss'] = total_loss['total_loss'] + w_modality_eff * modality_loss
        total_loss['loss_modality'] = modality_loss

        for key, loss in total_loss.items():
            self.train_loss[key].update(loss)
        # return total_loss['loss_doa']
        return total_loss['total_loss']

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # NOTE: we clamp to 4.6052 = ln(100), as in the original paper.
        self.net.logit_scale.clamp_(0, np.log(100))
        if self.cfg.model.mlp_loss:
            raise NotImplementedError

    ############################## Validation functions ##############################
    def on_validation_start(self):
        self.text_feature_doa = self.encode_direction_text()
        if self.cfg.get('edit') == 'modify':
            direction_texts = [
                'The sound is coming from the east.',
                'The sound is coming from the northeast.',
                'The sound is coming from the north.',
                'The sound is coming from the northwest.',
                'The sound is coming from the west.',
                'The sound is coming from the southwest.',
                'The sound is coming from the south.',
                'The sound is coming from the southeast.',
            ]
            self.direction_label_dict = {direction: i for i, direction in enumerate(direction_texts)}
            import json
            self.instructions = {}
            with open('datasets/spatial_audio_text/Clotho/metadata/test.json', 'r') as f:
                self.instructions.update(json.load(f))
            with open('datasets/spatial_audio_text/AudioCaps/metadata/test.json', 'r') as f:
                self.instructions.update(json.load(f))

    def validation_step(self, batch_sample, batch_idx, dataloader_idx=0):
        # Triplet batches from stClotho: (B,3,C,T) -> (B*3,C,T)
        if batch_sample['audio4sed'].dim() == 4:
            batch_sample['audio4sed'] = batch_sample['audio4sed'].flatten(0, 1)
            batch_sample['audio4doa'] = batch_sample['audio4doa'].flatten(0, 1)
            batch_sample['longer'] = batch_sample['longer'].flatten(0, 1)
            if 'cart_doa' in batch_sample:
                batch_sample['cart_doa'] = batch_sample['cart_doa'].flatten(0, 1)

            for key in ['text_sed', 'text_comb']:
                if key in batch_sample:
                    for k in batch_sample[key].keys():
                        # e.g. (B,3,L) or (B,3,K,L) -> (B*3,L) or (B*3,K,L)
                        if batch_sample[key][k].dim() >= 3:
                            batch_sample[key][k] = batch_sample[key][k].flatten(0, 1)

        EDIT = self.cfg.get('edit', False)
        if self.last_dataloader_idx != dataloader_idx:
            dataset_name = self.valid_dataset_names[self.last_dataloader_idx]
            self.compute_metrics(dataset_name, task='spatial_retrieval')
            self.last_dataloader_idx = dataloader_idx
            self.reset_system_output()
        audio = {'audio4sed': batch_sample['audio4sed'], 
                 'audio4doa': batch_sample['audio4doa']}
        text = {'text': batch_sample['text_sed'], 
                'text_comb': batch_sample['text_comb']}
        longer = batch_sample['longer']
        audio_features, text_features, doa = self.forward(audio, text, longer, normalize=not EDIT)
        
        if EDIT: # edit the doa_audio_feature using doa_text_feature
            audio_features = self.edit_features(audio_features, batch_sample, EDIT)
            for idx in range(len(audio_features)):
                audio_features[idx] = F.normalize(audio_features[idx], dim=-1)
            for idx in range(len(text_features)):
                text_features[idx] = F.normalize(text_features[idx], dim=-1)
        
        self.system_output['all_audio_features'].append(audio_features[-1])
        # self.system_output['sed_audio_features'].append(audio_features[1])
        # self.system_output['doa_audio_features'].append(audio_features[2])
        self.system_output['all_text_features'].append(text_features[0])
        # self.system_output['sed_text_features'].append(text_features[1])
        self.system_output['gt_doa'].append(batch_sample['cart_doa'])
        self.system_output['pred_doa'].append(doa)
    
    def on_validation_epoch_end(self):
        self.compute_metrics(self.valid_dataset_names[-1], 
                             task='spatial_retrieval')
        self.reset_system_output()
        self.last_dataloader_idx = 0

    ############################## Test functions ##############################
    def on_test_epoch_start(self):
        self.reset_system_output()
        if self.cfg.task == 'zero-shot-classification (Direction)': 
            self.system_output['doa_text_features'] = self.encode_direction_text()
            if 'doa_audio_features' not in self.system_output:
                self.system_output['doa_audio_features'] = []
        elif self.cfg.task == 'semantic_retrieval': pass
        else: raise NotImplementedError            

    def test_step(self, batch_sample, batch_idx):
        # Triplet batches from stClotho: (B,3,C,T) -> (B*3,C,T)
        if batch_sample['audio4sed'].dim() == 4:
            batch_sample['audio4sed'] = batch_sample['audio4sed'].flatten(0, 1)
            batch_sample['audio4doa'] = batch_sample['audio4doa'].flatten(0, 1)
            batch_sample['longer'] = batch_sample['longer'].flatten(0, 1)
            if 'cart_doa' in batch_sample:
                batch_sample['cart_doa'] = batch_sample['cart_doa'].flatten(0, 1)

            for key in ['text_sed', 'text_comb']:
                if key in batch_sample:
                    for k in batch_sample[key].keys():
                        if batch_sample[key][k].dim() >= 3:
                            batch_sample[key][k] = batch_sample[key][k].flatten(0, 1)

        audio = {'audio4sed': batch_sample['audio4sed'], 
                'audio4doa': batch_sample['audio4doa']}
        longer = batch_sample['longer']
        if self.cfg.task == 'zero-shot-classification (Direction)': 
            audio_features = self.forward(audio, None, longer)
            self.system_output['doa_audio_features'].append(audio_features[self.cfg.doa_feature_type])
            self.system_output['gt_doa'].append(batch_sample['cls_doa'])
        elif self.cfg.task == 'semantic_retrieval':
            text = {'text': batch_sample['text_sed'], 
                    'text_comb': batch_sample['text_comb']}
            audio_features, text_features, doa = self.forward(audio, text, longer)
            # NOTE: Semantic retrieval instead of spatial semantic retrieval
            self.system_output['all_audio_features'].append(audio_features[1])
            self.system_output['all_text_features'].append(text_features[1])
        else: raise NotImplementedError
    
    def on_test_epoch_end(self):
        if self.cfg.task == 'zero-shot-classification (Direction)': 
            metrics = evaluate_zero_shot(
                self.logging, self.system_output, 
                self.test_dataset_names[-1], audio_key='doa_audio_features', 
                text_key='doa_text_features', gt_key='gt_doa')
            self.log_metrics(metrics, self.test_dataset_names[-1])
            self.reset_system_output()
        elif self.cfg.task == 'semantic_retrieval':
            del self.system_output['gt_doa'], self.system_output['pred_doa']
            self.compute_metrics(self.test_dataset_names[-1], 
                                 task='retrieval')
            self.reset_system_output()
        else: raise NotImplementedError
        
    ############################## OTHERS ##############################
    def edit_features(self, audio_features, batch_sample, EDIT):
        text_feature_doa = self.text_feature_doa
        cls_doa = batch_sample['cls_doa']
        audio_feature_comb, audio_feature_sed, audio_feature_doa = audio_features
        amplitude = (audio_feature_doa ** 2).sum(dim=-1).sqrt().unsqueeze(-1)#.mean()

        if EDIT == 'modify': 
            filenames = batch_sample['audiofile']
            for idx, filename in enumerate(filenames):
                instruction = self.instructions[filename]
                new_dir_idx = self.direction_label_dict[instruction['new_dir']]
                new_feature_doa = text_feature_doa[new_dir_idx] * amplitude[idx]
                audio_feature_comb[idx] = audio_feature_sed[idx] + self.net.weights * new_feature_doa
        else:
            new_feature_doa = text_feature_doa[cls_doa] * amplitude
            audio_feature_comb = audio_feature_sed + self.net.weights * new_feature_doa
        audio_features = [audio_feature_comb, audio_feature_sed, audio_feature_doa]
        
        return audio_features

    def encode_direction_text(self):
        for k, v in self.label_embed.items():
            self.label_embed[k] = v.to(self.device)
        doa_text_embedding = self.net.encode_text(self.label_embed)
        return F.normalize(doa_text_embedding, dim=-1)

    def reset_system_output(self):
        self.system_output = {
            'all_audio_features': [], # [n_samples, n_dim] contain both semantic and spatial information
            # 'sed_audio_features': [], # [n_samples, n_dim] only contain semantic information
            # 'doa_audio_features': [], # [n_samples, n_dim] only contain spatial information
            'all_text_features': [], # [n_samples, n_dim] or [n_classes, n_dim] contain both semantic and spatial information
            # 'sed_text_features': [], # [n_samples, n_dim] or [n_classes, n_dim] only contain semantic information
            'pred_doa': [], 
            'gt_doa': []
        }


class DOAModelModule(BaseModelModule):

    logging = get_pylogger(__name__)

    def __init__(self, cfg, steps, label_embed=None):
        super().__init__(cfg, steps, label_embed)

        self.af_extractor = get_afextractor(cfg, audio_feature='logmelIV')

        self.train_loss.update(
            {'total_loss': MeanMetric(), 'loss_doa': MeanMetric()}
        )
    
    def setup(self, stage):

        from model.doa import HTSAT_DOA
        
        if self.cfg.data.truncation != 'fusion' and self.cfg.model.get('fusion', {}).get('enable'):
            raise ValueError('The fusion model is only allowed for truncation-fusion')

        self.net = HTSAT_DOA(self.cfg)
        

        if stage == 'fit' and self.cfg.compile:
            self.logging.info('Compiling model')
            self.net = torch.compile(self.net)
        if stage == 'test':
            pass
        self.logging.info("Number of parameters of the net: " + 
                          f"{sum(p.numel() for p in self.net.parameters())}")

    def forward(self, audio):
        audio['audio4doa'] = self.af_extractor(audio['audio4doa'])

        return self.net(audio)

    ############################## Training functions ##############################

    def training_step(self, batch_sample, batch_idx):
        audio = {'audio4sed': batch_sample['audio4sed'], 
                 'audio4doa': batch_sample['audio4doa']}
        doa = self.forward(audio)
            
        # total_loss = 1 - F.cosine_similarity(doa, batch_sample['cart_doa'], dim=-1).mean()
        total_loss = F.mse_loss(doa, batch_sample['cart_doa'])
        loss_doa = total_loss
        self.train_loss['loss_doa'].update(loss_doa)
        self.train_loss['total_loss'].update(total_loss)
        return total_loss

    ############################## Validation functions ##############################

    def validation_step(self, batch_sample, batch_idx, dataloader_idx=0):
        if self.last_dataloader_idx != dataloader_idx:
            dataset_name = self.valid_dataset_names[self.last_dataloader_idx]
            self.compute_metrics(dataset_name, task='spatial_retrieval')
            self.last_dataloader_idx = dataloader_idx
            self.reset_system_output()
        audio = {'audio4sed': batch_sample['audio4sed'], 
                 'audio4doa': batch_sample['audio4doa']}
        doa = self.forward(audio)
        
        self.system_output['gt_doa'].append(batch_sample['cart_doa'])
        self.system_output['pred_doa'].append(doa)
    
    def on_validation_epoch_end(self):
        self.compute_metrics(self.valid_dataset_names[-1], 
                             task='spatial_retrieval')
        self.reset_system_output()
        self.last_dataloader_idx = 0

    def reset_system_output(self):
        self.system_output = {
            'pred_doa': [], 
            'gt_doa': []
        }
    
    def compute_metrics(self, dataset_name, task='retrieval'):
        if 'pred_doa' in self.system_output.keys():
            for key in ['pred_doa', 'gt_doa']:
                self.system_output[key] = torch.cat(self.system_output[key], dim=0)

        cos_sim = F.cosine_similarity(
            self.system_output['pred_doa'], 
            self.system_output['gt_doa'], dim=-1
        )
        loss_doa = F.mse_loss(
            self.system_output['pred_doa'], 
            self.system_output['gt_doa']
        ).mean().item()
        self.logging.info(f"Loss_DOA for {dataset_name}: {loss_doa:.4f}\n")
        localization_error = torch.acos(cos_sim) * 180 / torch.pi
        localization_error = localization_error.mean().item()
        self.logging.info(f"Localization_error for {dataset_name}: {localization_error:.1f}\n")

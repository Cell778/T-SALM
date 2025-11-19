import numpy as np
import lightning as L
from torch import optim, nn
import torch

from utils.utilities import get_pylogger
from evaluate.eval_retrieval import evaluate


class BaseModelModule(L.LightningModule):

    logging = get_pylogger(__name__)

    def __init__(self, cfg, steps, label_embed=None):
        super(BaseModelModule, self).__init__()
        
        self.cfg = cfg
        self.steps = steps
        self.label_embed = label_embed
        self.net = None

        self.last_dataloader_idx = 0
        self.valid_dataset_names = list(cfg.data.valid_dataset.keys())
        self.test_dataset_names = list(cfg.data.test_dataset.keys())
        self.reset_system_output()
        self.train_loss = nn.ModuleDict()

    def forward(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer_params = self.cfg['model']['optimizer']
        lr_scheduler_params = self.cfg['model']['lr_scheduler']

        params = self.parameters()     
        optimizer = get_optimizer(params, optimizer_params['method'], 
                                  **optimizer_params['kwargs'])
        if lr_scheduler_params['method'] == 'cosinelr':
            warmup_steps = lr_scheduler_params['warmup_epochs'] * self.steps['num_steps_per_epoch']
            lr_lambda = lambda step: cosine_lr(step, warmup_steps, self.steps['max_steps'])
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,}
        elif lr_scheduler_params['method'] == 'steplr':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                step_size=lr_scheduler_params['step_size'], gamma=lr_scheduler_params['gamma'])
            scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,}
        else: raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    
    def log_losses(self, loss, set_type):
        out_str = set_type + ": "
        for key, value in loss.items():
            out_str += '{}: {:.4f}, '.format(key, value.compute())
            self.log(f'{set_type}/{key}', value.compute(), logger=True, 
                     on_epoch=True, on_step=False, sync_dist=True)
        self.logging.info(out_str)
    
    def compute_metrics(self, dataset_name, task='retrieval'):
        # Skip metric computation if no samples were collected (e.g. empty dataset)
        if (
            'all_audio_features' not in self.system_output
            or 'all_text_features' not in self.system_output
            or len(self.system_output['all_audio_features']) == 0
            or len(self.system_output['all_text_features']) == 0
        ):
            self.logging.info(f"[compute_metrics] Skip empty validation dataset: {dataset_name}")
            return
        self.system_output['all_audio_features'] = torch.cat(self.system_output['all_audio_features'], dim=0)
        self.system_output['all_text_features'] = torch.cat(self.system_output['all_text_features'], dim=0)
        if 'pred_doa' in self.system_output.keys(): # sCLAP
            for key in ['pred_doa', 'gt_doa']:
            # for key in ['pred_doa', 'gt_doa', 'sed_audio_features', 'doa_audio_features', 'sed_text_features']:
                self.system_output[key] = torch.cat(self.system_output[key], dim=0)

        if self.trainer.world_size > 1:
            # raise NotImplementedError
            self.gather_outputs(dataset_name)

        new_idx = None
        if self.cfg.get('edit') == 'modify':
            new_idx = self.instructions[f'new_idx_{dataset_name}']

        metrics = evaluate(self.logging, self.system_output, 
                           self.net.logit_scale, dataset_name, task, new_idx)
        self.log_metrics(metrics, dataset_name)

    def gather_outputs(self, dataset_name):
        self.system_output = self.all_gather(self.system_output)
        for key in self.system_output.keys():
            self.system_output[key] = self.system_output[key].flatten(0, 1)
            if 'text' in key: self.system_output[key] = self.system_output[key][:self.steps['val_samples'][dataset_name]*5]
            else: self.system_output[key] = self.system_output[key][:self.steps['val_samples'][dataset_name]]

    def log_metrics(self, value_dict, dataset_name):
        out_str = 'valid {}: \n'.format(dataset_name)
        for key, value in value_dict.items():
            out_str += key.upper() + ': '
            for k, v in value.items():
                if k == 'num_samples' or 'rank' in k:
                    out_str += '{}: {}\t'.format(k, int(v))
                else: out_str += '{}: {:.4f}\t'.format(k, v)
                self.log(f'val_{dataset_name}/{key}/{k}', v, logger=True, 
                        on_epoch=True, on_step=False, sync_dist=True, 
                        add_dataloader_idx=False)
            out_str += '\n'
        self.logging.info(out_str)
    
    def on_load_checkpoint(self, checkpoint):
        if self.cfg.compile: return
        keys_list = list(checkpoint['state_dict'].keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]  

    def training_step(self, batch_sample, batch_idx):
        raise NotImplementedError
    
    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]['lr']
        max_epochs = self.cfg.trainer.max_epochs
        self.log_losses(self.train_loss, set_type='train')
        self.log('lr', lr)
        self.logging.info(f"Epoch/Total Epoch: {self.current_epoch+1}/{max_epochs}, LR: {lr}")
        self.logging.info("-------------------------------------------"
                 + "---------------------------------------")
    
    def on_train_epoch_start(self):
        for key in self.train_loss.keys():
            self.train_loss[key].reset()

    def validation_step(self, batch_sample, batch_idx):
        raise NotImplementedError

    def test_step(self, batch_sample, batch_idx):
        raise NotImplementedError
    
    def reset_system_output(self):
        self.system_output = {}

def get_optimizer(params, optimizer_name, lr, betas, eps, momentum):
    if isinstance(betas, list): 
        betas = tuple(betas)

    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            params, lr=lr, betas=betas, eps=eps
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            params, lr=lr, momentum=momentum
        )
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            params, lr=lr, betas=betas, eps=eps
        )
    else:
        raise ValueError("optimizer name is not correct")
    return optimizer


def cosine_lr(step, warmup_step, steps):
    if step < warmup_step:
        return (step + 1) / warmup_step
    else:
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_step) / (steps - warmup_step)))
    
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from utils.utilities import get_pylogger

log = get_pylogger(__name__)


class DataModule(L.LightningDataModule):

    from data.data import CLAPDataset, sCLAPDataset
    UserDataset = {
        'retrieval': CLAPDataset,
        'zero-shot-classification': CLAPDataset, # only for test stage
        'spatial-retrieval': sCLAPDataset,
        'semantic_retrieval': sCLAPDataset, # only for test stage
        'zero-shot-classification (Direction)': sCLAPDataset, # only for test stage
    }

    def __init__(self, cfg, stage='fit'):
        super().__init__()
        self.cfg = cfg
        self.seed = cfg.seed
        self.batch_size = cfg.model.batch_size
        self.steps = {'max_steps': -1, 'val_samples': {}}
        self.text_embed = None

        if stage in ['fit', 'train', 'valid']:
            self.train_set, self.val_set = [], []

            for dataset_name, splits in cfg.data.train_dataset.items():
                train_set = self.UserDataset[cfg.task](cfg, dataset_name, splits, 'train')
                self.train_set.append(train_set)
                log.info(f"Training clip number of {dataset_name} is: {len(train_set)}")
                if self.text_embed is None: self.text_embed = train_set.text_embed
            self.train_set = ConcatDataset(self.train_set)
            log.info(f"Training clip number is: {len(self.train_set)}")

            for dataset_name, splits in cfg.data.valid_dataset.items():
                val_set = self.UserDataset[cfg.task](cfg, dataset_name, splits, 'valid')
                self.val_set.append(val_set)
                log.info(f"Validation clip number of {dataset_name} is: {len(val_set)}")
                self.steps['val_samples'][dataset_name] = len(val_set)
                if self.text_embed is None: self.text_embed = val_set.text_embed
            self.steps['num_steps_per_epoch'] = np.ceil(len(self.train_set) / self.batch_size)
            self.steps['max_steps'] = self.steps['num_steps_per_epoch'] * cfg.trainer.max_epochs

        elif stage == 'test':
            # self.test_set = []
            assert len(cfg.data.test_dataset.keys()) == 1, "Only one test dataset is allowed"
            for dataset_name, splits in cfg.data.test_dataset.items():
                self.test_set = self.UserDataset[cfg.task](cfg, dataset_name, splits, 'test')
                log.info(f"Testing clip number of {self.test_set.dataset_name} is: {len(self.test_set)}")
                if self.text_embed is None: self.text_embed = self.test_set.text_embed
    
    def train_dataloader(self):
        
        return DataLoader(dataset=self.train_set, 
                          shuffle=True,
                          batch_size=self.batch_size, 
                          pin_memory=True,
                          num_workers=self.cfg.num_workers, 
                          generator=torch.Generator().manual_seed(self.seed))
    
    def val_dataloader(self):
        
        return [DataLoader(dataset=val_set, 
                           batch_size=self.batch_size,
                           shuffle=False, 
                           num_workers=self.cfg.num_workers//4,
                           generator=torch.Generator().manual_seed(self.seed),
                           pin_memory=True
                           ) for val_set in self.val_set]
    
    def test_dataloader(self):

        return  DataLoader(dataset=self.test_set, 
                           batch_size=self.batch_size,
                           shuffle=False, 
                           num_workers=self.cfg.num_workers,
                           generator=torch.Generator().manual_seed(self.seed),
                           pin_memory=True)
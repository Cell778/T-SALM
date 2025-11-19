# Reference: https://github.com/ashleve/lightning-hydra-template

from typing import List

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.utilities import (extras, get_pylogger, instantiate_loggers, log_hyperparameters)

log = get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig):
    """ Train or test the model.
    training or testing.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.datamodule, cfg, cfg.mode)
    text_embed = datamodule.text_embed

    log.info(f"Instantiating model <{cfg.modelmodule._target_}> ...")
    model = hydra.utils.instantiate(cfg.modelmodule, cfg, 
                                    datamodule.steps, text_embed)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    
    if cfg.mode == 'test':
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")) # type: ignore
    elif cfg.mode == 'valid':
        log.info("Starting validation!")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    
    main()
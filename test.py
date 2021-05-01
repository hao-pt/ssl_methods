import os
import torch
import time

from config import Config

from dataset import create_test_loader
from model_factory import get_model
from meanteacher import Tester

from model_utils import save_checkpoint, load_checkpoint

if __name__ == "__main__":
    cfg = Config()
    cfg.device = torch.device("cuda" if cfg.device_ids != "cpu" else "cpu")

    # dataset
    eval_loader = create_test_loader(cfg.data_dir, cfg)
    
    # create model
    model = get_model(cfg.model_arch, pretrained=cfg.pretrained)
    ema_model = get_model(cfg.model_arch, pretrained=cfg.pretrained, ema=True)


    # resume training / load trained weights
    last_epoch = 0
    if cfg.resume:
        model, ema_model, optimizer, last_epoch = load_checkpoint(model, ema_model, cfg.resume)
    
    # create trainer
    tester = Tester(cfg, model, ema_model)
    tester._set_device(cfg.device) 

    results = tester(eval_loader)

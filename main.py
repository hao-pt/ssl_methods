import torch

from config import get_config

from model_factory import get_model
from meanteacher import Trainer

if __name__ == "__main__":
    cfg = get_config()
    cfg.device = torch.device("cuda" if cfg.device_ids != "cpu" else "cpu")

    # dataset
    
    # create model
    model = get_model(cfg.model_name)
    ema_model = get_model(cfg.model_name, ema=True)

    # create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        cfg.lr, 
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov)
    
    trainer = Trainer(cfg, model, ema_model, optimizer)
    trainer._set_device() 
    trainer._create_ema_updater()

    pass
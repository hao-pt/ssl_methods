import os
import torch
import time

from config import Config

from dataset import create_data_loaders
from model_factory import get_model
from meanteacher import Trainer

from model_utils import save_checkpoint, load_checkpoint

if __name__ == "__main__":
    cfg = Config()
    cfg.device = torch.device("cuda" if cfg.device_ids != "cpu" else "cpu")

    # dataset
    train_loader, val_loader = create_data_loaders(cfg.data_dir, cfg)
    
    # create model
    model = get_model(cfg.model_arch, pretrained=cfg.pretrained)
    ema_model = get_model(cfg.model_arch, pretrained=cfg.pretrained, ema=True)

    # create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        cfg.lr, 
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov)

    # resume training / load trained weights
    last_epoch = 0
    if cfg.resume:
        model, ema_model, optimizer, last_epoch = load_checkpoint(model, ema_model, cfg.resume, optimizer)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    rundir = f'{cfg.log_dir}/meanteacher/{time_str}'
    os.makedirs(rundir, exist_ok=True)
    cfg.save(rundir) # save configs

    # create trainer
    trainer = Trainer(cfg, model, ema_model, optimizer)
    trainer._set_device() 
    trainer._create_ema_updater()
    trainer._create_summary_writer(rundir)

    is_best, best_acc = False, 0.0
    ckpt_dir = f"{cfg.weight_dir}/{cfg.dataset}/meanteacher"
    os.makedirs(ckpt_dir, exist_ok=True)
    

    for epoch in range(last_epoch, cfg.epochs):
        trainer.train(epoch, train_loader)

        if epoch % cfg.eval_interval == 0:
            # get top-1 accuracy of student and teacher
            acc, ema_acc = trainer.val(epoch, val_loader)
            
            if ema_acc > best_acc:
                best_acc = ema_acc
                is_best = True
            else:
                is_best = False

        save_checkpoint(ckpt_dir, {
            "epoch": epoch,
            "arch": cfg.model_arch,
            "state_dict": trainer.model.state_dict(),
            "ema_state_dict": trainer.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc}, is_best, epoch)

    
import os
import torch
import shutil

def load_checkpoint(model, ema_model, ckpt_path, optimizer=None):
    """
    Load model checkpoint
    """
    state = torch.load(ckpt_path)

    last_epoch = state["epoch"]
    model.load_state_dict(state["state_dict"])
    ema_model.load_state_dict(state["ema_state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer"])

    return model, ema_model, optimizer, last_epoch

def save_checkpoint(dirpath, state, is_best, epoch):
    filename = 'last_model.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best_model.ckpt')
    torch.save(state, checkpoint_path)
    # save best model if any
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


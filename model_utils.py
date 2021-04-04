import os
import torch
import shutil

def load_model():
    pass

def save_checkpoint(dirpath, state, is_best, epoch):
    filename = 'last_model.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best_model.ckpt')
    torch.save(state, checkpoint_path)
    # save best model if any
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


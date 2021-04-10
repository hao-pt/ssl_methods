import numpy as np
from numpy.core.numeric import correlate
import torch

def sigmoid_rampup(current_t, rampup_length=100):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    current_t =  np.clip(current_t, 0, rampup_length) 

    return float(np.exp(-5.0*(1-current_t/rampup_length)**2))

def linear_rampup(current_t, rampup_length):
    if current_t >= rampup_length:
        return 1

    return current_t / rampup_length

def cosine_rampdown(current_t, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current_t <= rampdown_length
    return float(.5 * (np.cos(np.pi * current_t / rampdown_length) + 1))
    
def consistency_rampup_weight(weight, epoch, rampup_length):
    """
    Computing consistency weight during rampup phase since output distribution of model varies alot 
    in early stage of training
    """
    return weight*sigmoid_rampup(epoch, rampup_length)

def lr_schedule(optimizer, epoch, cfg, cur_step, total_steps):
    """
    Learning rate scheduler
    - linear rampup 
    - cosine rampdown
    """
    lr = cfg.lr
    num_steps += cur_step/total_steps

    # linear ramp-up for handling large batch size https://arxiv.org/abs/1706.02677
    lr = linear_rampup(num_steps, cfg.lr_rampup_length) * (lr - cfg.initial_lr) + cfg.initial_lr
    
    if cfg.lr_rampdown_length:
        lr *= cosine_rampdown(num_steps, cfg.lr_rampdown_length)

    # set lr for all params
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def accuracy(preds, targets, topk=(1,)):
    # get topk results
    _, topk_preds = torch.topk(preds, max(topk), dim=1, largest=True, sorted=True)
    topk_preds = topk_preds.t() # (topk, #batches)
    correct = topk_preds.eq(targets.view(1,-1))
    
    # get labeled batch size
    labeled_batch_size = max(targets.ne(-1).sum(), 1e-8)
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k*100/labeled_batch_size)

    return res
    
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


if __name__ == "__main__":
    targets = torch.tensor([[1], [2], [-1], [3], [2], [-1]])
    preds = torch.rand((6,5))
    print(preds)
    print(accuracy(preds, targets, topk=(1,5)))
    
    
    
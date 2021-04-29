import os
import torch
import time
import random
import numpy as np

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from config import Config

from dataset import create_data_loaders
from model_factory import get_model
from meanteacher import Trainer

from model_utils import save_checkpoint, load_checkpoint

def setup(distributed):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed
    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 8  # 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank

def train(cfg):
    # ############################################################
    # if cfg.use_ddp:
    #     rank = cfg.node_rank * cfg.gpus + device	                          
    #     # init distributed backend
    #     dist.init_process_group(                                   
    #         backend='nccl',                                         
    #         init_method='env://',                                   
    #         world_size=cfg.world_size,                              
    #         rank=rank                                               
    #     )                                                          
    #     # torch.distributed.init_process_group(backend="nccl")
    #     ############################################################
    #     cfg.local_rank = rank
    #     torch.cuda.set_device(device)
    #     print("{}/{} process initialized.\n".format(rank+1, cfg.world_size))
    # else:
    #     rank = device

    device, local_rank = setup(distributed=cfg.use_ddp)
    torch.cuda.set_device(device)
    if cfg.use_ddp:
        torch.set_num_threads(1) # #cpu_threads / #process_per_node

    cfg.local_rank = local_rank
    is_main_gpu = not cfg.use_ddp or int(os.environ.get('RANK')) == 0

    # dataset
    train_loader, val_loader = create_data_loaders(cfg.data_dir, cfg)

    if cfg.use_ddp:
        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
    
    # create model
    model = get_model(cfg.model_arch, pretrained=cfg.pretrained)
    ema_model = get_model(cfg.model_arch, pretrained=cfg.pretrained, ema=True)

    # resume training / load trained weights
    last_epoch = 0
    if cfg.resume:
        model, ema_model, optimizer, last_epoch = load_checkpoint(model, ema_model, cfg.resume, optimizer)
    
    if is_main_gpu:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        rundir = f'{cfg.log_dir}/meanteacher/{time_str}'
        print("Make dir", rundir)
        os.makedirs(rundir, exist_ok=True)
        cfg.save(rundir) # save configs

    # create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        cfg.lr, 
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov)

    # create trainer
    trainer = Trainer(cfg, model, ema_model, optimizer)
    trainer._set_device(device) 
    trainer._create_ema_updater()
    if is_main_gpu:
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

        if is_main_gpu:
            save_checkpoint(ckpt_dir, {
                "epoch": epoch,
                "arch": cfg.model_arch,
                "state_dict": trainer.model.state_dict(),
                "ema_state_dict": trainer.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc}, is_best, epoch)

    if cfg.use_ddp:
        torch.distributed.destroy_process_group()
        

if __name__ == "__main__":
    cfg = Config()
    cfg.device = torch.device("cuda" if cfg.device_ids != "cpu" else "cpu")

    ### Distribute a process for every GPU
    # if cfg.use_ddp:
    #     cfg.world_size = cfg.gpus * cfg.nodes # compute total of #processes to run = #gpus

    #     os.environ['MASTER_ADDR'] = '127.0.0.1' # IP to look for process 0, so all process can sync up             #
    #     os.environ['MASTER_PORT'] = '29500' # '8888' # port
    #     # run train(i, cfg) where i goes from 0 to cfg.gpus-1
    #     mp.spawn(train, nprocs=cfg.gpus, args=(cfg,), join=True) 
    # else:
    #     train(0, cfg)
    train(cfg)

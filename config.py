import argparse
from ast import parse

def create_args():
    parser = argparse.ArgumentParser(description="Semi supervised learning for methods")

    data_group = parser.add_argument_group(title='Data group')
    data_group.add_argument("--data_dir", help="Data directory")


    facility_group = parser.add_argument_group(title='Facility group')
    facility_group.add_argument("--device_ids", default="0",
        help="Device to use such as 0, 1, 2, ... or cpu")
    facility_group.add_argument("--workers", type=int, default=4,
        help="Number of data loaders")

    hyper_param_group = parser.add_argument_group(title='Hyperameter group')
    hyper_param_group.add_argument("--epochs", type=int, default=100,
        help="#Epochs")
    hyper_param_group.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate")
    hyper_param_group.add_argument("--batch_size", type=float, default=32,
        help="Batch size")
    hyper_param_group.add_argument("--labeled_batch_size", type=float, default=16,
        help="Labeled batch size")
    hyper_param_group.add_argument("--unp_weight", type=float, default=10,
        help="Unsupervised weight")
    hyper_param_group.add_argument("--ramup_length", type=float, default=30,
        help="Length of ramup period")
    hyper_param_group.add_argument("--weight_decay", type=float, default=1e-4,
        help="Weight decay")
    hyper_param_group.add_argument("--nesterov", type=bool, default=False,
        help="Use Nesterov in SGD")
    hyper_param_group.add_argument("--eval_interval", type=int, default=5,
        help="Evaluation interval on val set")
    hyper_param_group.add_argument("--print_interval", type=int, default=5,
        help="Print interval for training")
    
    unp_group = parser.add_argument_group(title='Unsupervised group')
    unp_group.add_argument("--exclude_unlabeled", type=bool, default=false,
        help="Exclude unlabeled data")

    return parser.parse_args()


def get_config():
    cfg = create_args()
    cfg.UNLABELED = -1 # mark targe for unlabeled data

    return cfg 


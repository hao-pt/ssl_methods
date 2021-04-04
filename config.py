import argparse
import json
import jsonpickle

NUM_CLASSES = {"cifar10": 10,
    "imagenet": 1000}
class Config:
    def __init__(self):
        self.create_args()
        self.NO_LABEL = -1 # mark targe for unlabeled data
        self.num_classes = NUM_CLASSES[self.dataset]

        self.chunk_sizes = [self.master_batch_size]
        rest_batch_size = self.batch_size - self.master_batch_size
        num_gpus = len(self.device_ids)
        for i in range(num_gpus - 1):
            slave_chunk_size = rest_batch_size // (num_gpus - 1)
            if i < rest_batch_size % (num_gpus - 1):
                slave_chunk_size += 1
            self.chunk_sizes.append(slave_chunk_size)
        print('Training chunk_sizes:', self.chunk_sizes)

    def create_args(self):
        parser = argparse.ArgumentParser(description="Semi supervised learning for methods")

        data_group = parser.add_argument_group(title='Data group')
        data_group.add_argument("--data_dir", default="datadir",
            help="Data directory")
        data_group.add_argument("--dataset", default="cifar10",
            help="Name of dataset such as: cifar10")
        data_group.add_argument("--weight_dir", default="weights",
            help="Weight directory")
        data_group.add_argument("--log_dir", default="logs",
            help="Log directory")
        data_group.add_argument("--labels", default="data/cifar10_labels/4000_balanced_labels/00.txt",
            help="File lists labeled images")
        data_group.add_argument("--exclude_unlabeled", type=bool, default=False,
            help="Exclude unlabeled data")

        model_group = parser.add_argument_group(title='Model group')
        model_group.add_argument("--model_arch", default="shake_resnet26", choices=["wide_resnet50_2", 
            "resnet18", "shake_resnet26"], help="Model architectures")
        model_group.add_argument("--pretrained", type=bool, default=False, 
            help="Use ImageNet pretrained")

        facility_group = parser.add_argument_group(title='Facility group')
        facility_group.add_argument("--device_ids", default="-1",
            help="Device to use such as 0, 1, 2, ... or -1 (cpu)")
        facility_group.add_argument("--workers", type=int, default=4,
            help="Number of data loaders")
        facility_group.add_argument("--master_batch_size", type=int, default=-1,
            help="Batch size of main gpu oftens 0")

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
        hyper_param_group.add_argument("--ema_decay", type=float, default=0.999,
            help="Ema decay rate")
        hyper_param_group.add_argument("--use_num_updates", action="store_true",
            help="Use number of updates to compute ema decay")
        hyper_param_group.add_argument('--rampup_decay', type=float, default=0.99,
                             help='Ema decay during rampup phase')
        hyper_param_group.add_argument('--rampup_steps', type=int, default=20000,
                             help='Change ema_decay to 0.999 if step > rampup_steps else 0.99')
        

        return parser.parse_args(namespace=self)

    def save(self, rundir):
        with open(f"{rundir}/configs.json", "wt") as f:
            cfg_dict = self.__dict__.copy()
            # Convert non-serializable objects to str
            for key in cfg_dict:
                if not isinstance(cfg_dict, str):
                    cfg_dict[key] = str(cfg_dict[key])
                
            json.dump(cfg_dict, f, indent=4)
        
    def load(self, config_file):
        with open(config_file, "rt") as f:
            self = json.load(f)

if __name__ == "__main__":
    cfg = Config()
    # print(cfg.data_dir)
    config_file = r"/home/hp/Desktop/ssl_methods/logs/meanteacher/2021-04-04-18-36/configs.json"
    cfg.load(config_file)
    



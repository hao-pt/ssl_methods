## Setup
```
conda create -n ssl python=3.7
conda activate ssl
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
```

## Download data:
Cifar:
```
python download_cifar10.py --data_dir datadir
```
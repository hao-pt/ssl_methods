This is unofficial implementation of MeanTeacher that purposes to reproduce results of Semi-Supervised Learning method for Image Classification problem.

## Setup
```
conda create -n ssl python=3.7
conda activate ssl
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch

pip install -r requirements.txt
```

## Download data:
Cifar:
```
python data/download_cifar10.py --data_dir datadir
```

## Run:
Train on CIFAR10
```
python main.py \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/00.txt \
    --model_arch shake_resnet26 \
    --unp_weight 100 \
    --rampup_length 5 \
    --labeled_batch_size 62 \
    --batch_size 256 \
    --epochs 180
```
Evalate on CIFAR10
```
python test.py --data_dir datadir/cifar10/ \
    --batch_size 20 \
    --resume weights/cifar10/meanteacher/best_model.ckpt \
    --test_set val \
    --device_ids 0
```

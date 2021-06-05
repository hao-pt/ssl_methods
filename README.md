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
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/$trial.txt \
    --model_arch shake_resnet26 \
    --weight_decay 2e-4 \
    --unp_weight 100 \
    --rampup_length 5 \
    --batch_size 512 \
    --labeled_batch_size 124 \
    --epochs 180 \
    --ema_decay 0.97 \
    --nesterov True \
    --lr_rampdown_length 210 \
    --device_ids 0python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/$trial.txt \
    --model_arch shake_resnet26 \
    --weight_decay 2e-4 \
    --unp_weight 100 \
    --rampup_length 5 \
    --batch_size 512 \
    --labeled_batch_size 124 \
    --epochs 180 \
    --ema_decay 0.97 \
    --nesterov True \
    --lr_rampdown_length 210 \
    --device_ids 0python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/00.txt \
    --model_arch shake_resnet26 \
    --unp_weight 100 \
    --rampup_length 5 \
    --labeled_batch_size 62 \
    --batch_size 256 \
    --epochs 180 \
    --device_ids 0
```
Distributed training on CIFAR10 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/$trial.txt \
    --model_arch shake_resnet26 \
    --weight_decay 2e-4 \
    --unp_weight 100 \
    --rampup_length 5 \
    --batch_size 512 \
    --labeled_batch_size 124 \
    --epochs 180 \
    --ema_decay 0.97 \
    --nesterov True \
    --lr_rampdown_length 210 \
    --device_ids 0python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/$trial.txt \
    --model_arch shake_resnet26 \
    --weight_decay 2e-4 \
    --unp_weight 100 \
    --rampup_length 5 \
    --batch_size 512 \
    --labeled_batch_size 124 \
    --epochs 180 \
    --ema_decay 0.97 \
    --nesterov True \
    --lr_rampdown_length 210 \
    --device_ids 0python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/00.txt \
    --model_arch shake_resnet26 \
    --unp_weight 100 \
    --rampup_length 5 \
    --labeled_batch_size 62 \
    --batch_size 256 \
    --epochs 180 \
    --use_ddp
```

Evalate on CIFAR10
```
python test.py --data_dir datadir/cifar10/ \
    --batch_size 20 \
    --resume weights/cifar10/meanteacher/best_model.ckpt \
    --test_set val \
    --device_ids 0
```

# Reproduce results:
Do 10 runs on 1000-labeled CIFAR10:
```bash
bash cifar10_1k_experiments.sh
```

Do 10 runs on 4000-labeled CIFAR10:
```bash
bash cifar10_4k_experiments.sh
```

# Results:
Average results over 10 runs on CIFAR10 with 4k labeled images:

<table style="width:100%">
  <tr>
    <th></th>
    <th>student_top1</th>
    <th>teacher_top1</th>
    <th>student_top5</th>
    <th>teacher_top5</th>
  </tr>
  <tr>
    <td><b>my acc</b></td>
    <td>91.12 &plusmn .54 </td>
    <td>91.59 &plusmn .36 </td>
    <td>99.63 &plusmn .07 </td>
    <td>99.72 &plusmn .05 </td>
  </tr>
  <tr>
    <td><b>my err</b></td>
    <td>8.88 &plusmn .54 </td>
    <td>8.41 &plusmn .36 </td>
    <td>.37 &plusmn .07 </td>
    <td>.28 &plusmn .05 </td>
  </tr>
  <tr>
    <td><b>paper err</b></td>
    <td>-</td>
    <td>6.28 &plusmn .15</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

# Reference:
[1] Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." arXiv preprint arXiv:1703.01780 (2017). [[paper]](https://arxiv.org/pdf/1703.01780.pdf)[[repo]](https://github.com/CuriousAI/mean-teacher/tree/master/pytorch)

[2] [pytorch_ema](https://github.com/fadel/pytorch_ema.git)
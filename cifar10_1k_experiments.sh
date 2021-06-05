#!/bin/bash
mkdir -p /mnt/vinai/meanteacher/weights/cifar10_1k
# Basic while loop
counter=10
while [ $counter -lt 20 ]
do
    echo Trial $counter
    # train
    trial=$(printf "%0*d" 2 $counter)
    
    python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/1000_balanced_labels/$trial.txt \
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
    --device_ids 0

    # test
    python test.py --data_dir datadir/cifar10/ \
    --batch_size 20 \
    --resume weights/cifar10/meanteacher$trial/best_model.ckpt \
    --test_set test \
    --device_ids 0

    # cp trained weights
    cp -r weights/cifar10/meanteacher$trial /mnt/vinai/meanteacher/weights/cifar10_1k

    ((counter++))
done
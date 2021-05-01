#!/bin/bash
# Basic while loop
counter=0
while [ $counter -lt 20 ]
do
    echo Trial $counter
    # train
    trial=$(printf "%0*d" 2 $counter)
    
    python main.py \
    --data_dir datadir/cifar10/ \
    --dataset cifar10 \
    --labels data/cifar10_labels/4000_balanced_labels/$trial.txt \
    --model_arch shake_resnet26 \
    --unp_weight 100 \
    --rampup_length 5 \
    --labeled_batch_size 62 \
    --batch_size 256 \
    --epochs 180 \
    --device_ids 0

    # test
    python test.py --data_dir datadir/cifar10/ \
    --batch_size 20 \
    --resume weights/cifar10/meanteacher$trial/best_model.ckpt \
    --test_set test \
    --device_ids 0

    ((counter++))
done
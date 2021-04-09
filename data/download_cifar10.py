# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import os
import pickle
import sys

from tqdm import tqdm
import argparse

import matplotlib.image
import numpy as np
from torchvision.datasets import CIFAR10


parser = argparse.ArgumentParser("Prepare cifar10")
parser.add_argument("--data_dir", default="datadir", 
    help="Data directory for storing data")
args = parser.parse_args()

data_dir = os.path.abspath(args.data_dir)
test_dir = os.path.abspath(os.path.join(data_dir, 'test'))
train_dir = os.path.abspath(os.path.join(data_dir, 'train+val'))

print("Downloading CIFAR-10")
cifar10 = CIFAR10(data_dir, download=True)

def load_file(file_name):
    with open(os.path.join(data_dir, cifar10.base_folder, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")


def unpack_data_file(source_file_name, target_dir, start_idx):
    print("Unpacking {} to {}".format(source_file_name, target_dir))
    data = load_file(source_file_name)
    for idx, (image_data, label_idx) in tqdm(enumerate(zip(data['data'], data['labels'])), total=len(data['data'])):
        subdir = os.path.join(target_dir, label_names[label_idx])
        name = "{}_{}.png".format(start_idx + idx, label_names[label_idx]) # generate filename
        os.makedirs(subdir, exist_ok=True)
        image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
        matplotlib.image.imsave(os.path.join(subdir, name), image)
    return len(data['data'])

# get classes
label_names = load_file('batches.meta')['label_names']
print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

# get test images
print("Get test images")
start_idx = 0
for source_file_path, _ in cifar10.test_list:
    start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

# get train-val images
print("Get train-val images")
start_idx = 0
for source_file_path, _ in cifar10.train_list:
    start_idx += unpack_data_file(source_file_path, train_dir, start_idx)

# Create symbolic links for train images and val images
output_dir = f"{data_dir}/cifar10" 
splits = ["train", "val"]
split_pattern = os.path.abspath("data/cifar10_%s.txt")
for split in splits:
    src_dir = train_dir
    dst_dir = f"{output_dir}/{split}"

    with open(split_pattern%split) as f:
        filenames = f.read().split("\n")

        print(src_dir, dst_dir)
        print("Create symbolic links for %s images"%split)

        for filename in filenames:
            class_name = os.path.splitext(filename)[0].split("_")[-1]
            sub_folder = "%s/%s"
            os.makedirs(sub_folder%(dst_dir, class_name), exist_ok=True)

            try:
                print("Link %s to %s" %(f"{sub_folder%(src_dir, class_name)}/{filename}",
                    f"{sub_folder%(dst_dir, class_name)}/{filename}"))
                os.symlink(f"{sub_folder%(src_dir, class_name)}/{filename}", 
                    f"{sub_folder%(dst_dir, class_name)}/{filename}")
            except:
                print(f"{sub_folder%(dst_dir, class_name)}/{filename} exits")


print("Create symbolic links for test images")
src_dir = test_dir
dst_dir = f"{output_dir}/test"
for root, folders, files in os.walk(test_dir):
    for filename in filenames:
        class_name = os.path.splitext(filename)[0].split("_")[-1]
        sub_folder = "%s/%s"
        os.makedirs(sub_folder%(dst_dir, class_name), exist_ok=True)

        try:
            print("Link %s to %s" %(f"{sub_folder%(src_dir, class_name)}/{filename}",
                f"{sub_folder%(dst_dir, class_name)}/{filename}"))
            os.symlink(f"{sub_folder%(src_dir, class_name)}/{filename}", 
                f"{sub_folder%(dst_dir, class_name)}/{filename}")
        except:
            print(f"{sub_folder%(dst_dir, class_name)}/{filename} exits")

# split_file = "/home/hp/Desktop/ssl_methods/data/link_cifar10_train.sh"
# with open(split_file) as f:
#     data = f.read().split("\n")[9:]
#     file_names = []
#     for line in data:
#         filename =  line.split(" ")[-1].split("/")[-1]
#         file_names.append(filename)

# output_split_file = "data/cifar10_train.txt"
# with open(output_split_file, "wt") as f:
#     for filename in file_names:
#         f.write(filename+"\n")
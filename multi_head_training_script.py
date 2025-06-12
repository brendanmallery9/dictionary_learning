import os
import sys
import math
import time
import random
import pickle
import csv
import datetime
import gc

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
import torchvision.utils as utils

import cvxpy as cp
import qpth

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from joblib import Parallel, delayed

from OT_utils import *
from models.multi_head_dir.multi_head_model import *
from models.multi_head_dir.multi_head_training import *
from visualizations import *
from generate_data import *
#from self_supervised_training import *
from multi_head_utils_mps import *

import argparse


def save_args(args, loss,directory):
    os.makedirs(directory, exist_ok=True)
    args_file = os.path.join(directory, "training_args.csv")
    with open(args_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Argument', 'Value'])
        for arg, val in vars(args).items():
            writer.writerow([arg, val])
        writer.writerow(['Loss',loss])

def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--outer_epoch_schedule', type=int, nargs='+', default=30,
                        help='List of outer epochs per schedule phase, e.g., --outer_epoch_schedule 10 20')
    parser.add_argument('--inner_epochs_1', type=int, default=5)
    parser.add_argument('--inner_epochs_2', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_increment', type=int, default=100000)
    parser.add_argument('--dtype', type=str, default='MNIST')
    parser.add_argument('--filename', type=str, default=f'training_runs/{date_str}_runs')
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--variance_threshold_scaling', type=float, default=1.2)
    # Add parameters for reg_schedule if you want to customize from CLI
    parser.add_argument('--reg_base_value', type=float, default=0.0)
    parser.add_argument('--reg_scale_factor', type=float, default=1.0)
    parser.add_argument('--reg_min_i', type=int, default=0)
    
    parser.add_argument('--base_points', type=str, default='datasets/vary_base_tensor_data/0_4_8_LBCM_base_points.pt')

    parser.add_argument('--mapping_path', type=str, default='datasets/vary_base_tensor_data/0_4_8_LBCM_mappings.pt')

    parser.add_argument('--base_logic',type=str, default='VaryBase', choices=['VaryBase','FixedBase'])
    parser.add_argument('--sparsity_reg',type=float,default=0.01)
    parser.add_argument('--warm_start_length',type=int,default=0)
    parser.add_argument('--comment', type=str, default='', help='Optional comment for the training run')

    return parser.parse_args()

def main():
    start_time = time.time()  # Start timing

    args = parse_args()

    QP_reg_schedule = reg_schedule(base_value=args.reg_base_value,
                                   scale_factor=args.reg_scale_factor,
                                   min_i=args.reg_min_i)

    dim = 2
    no_heads = 10
    dropout_prob = 0
    dtype = args.dtype
    model_class = SixLayer_Net_Multihead_BatchNorm
    model_init_args = (dim, no_heads, dropout_prob)
    model_state_dict=None
    #model_state_dict = 'training_runs/2025-06-09_runs/Trial_1749501917.9238/multi_head_model_24000.pt'

    try:
        if args.base_logic == 'VaryBase':
            base_point_tensor = torch.load(args.base_points)
            mapping_tensor = torch.load(args.mapping_path)
            dataset = PairedTensors(base_data=base_point_tensor, mapping_data=mapping_tensor, dtype=dtype)
        else:
            base_point_tensor = torch.load(args.base_points).unsqueeze(0)
            mapping_tensor = torch.load(args.mapping_path)
            coefficients_tensor = torch.ones(len(mapping_tensor), no_heads)
            coefficients_tensor = coefficients_tensor / coefficients_tensor.sum(dim=1, keepdim=True)
            dataset = MappingCoeffTensors(
                base_point_tensor=base_point_tensor,
                mapping_data=mapping_tensor,
                dtype=dtype,
                coefficients_data=coefficients_tensor)

    except Exception as e:
        print(f"Error loading mapping tensor from {args.mapping_path}: {e}")
        return

    inner_epochs_pair = (args.inner_epochs_1, args.inner_epochs_2)
    model = model_class(*model_init_args)
    if model_state_dict is not None:
        model.load_state_dict(torch.load(model_state_dict, map_location="mps"))
    model = model.to("mps")

    full_dir_name, inner_loss = multi_head_train_cuda(
        model=model,
        dataloader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True),
        outer_epoch_schedule=args.outer_epoch_schedule,
        inner_epochs_pair=inner_epochs_pair,
        QP_reg_schedule=QP_reg_schedule,
        batch_size=args.batch_size,
        save_increment=args.save_increment,
        dtype=dtype,
        filename=args.filename,
        lambda_reg=args.lambda_reg,
        variance_threshold_scaling=args.variance_threshold_scaling,
        lr=args.lr,
        base_logic=args.base_logic,
        sparsity_reg=args.sparsity_reg,
        device=torch.device("mps"),
        warm_start_length=args.warm_start_length)

    end_time = time.time()  # End timing
    total_time = end_time - start_time

    print(f"Total training time: {total_time:.2f} seconds")

    # Save total time to file
    with open(os.path.join(full_dir_name, "training_time.txt"), "w") as f:
        f.write(f"Total training time (seconds): {total_time:.2f}\n")

    save_args(args, inner_loss, full_dir_name)

    # Save comment to comments.csv if provided
    if args.comment:
        comments_file = "comments.csv"
        file_exists = os.path.exists(comments_file)

        with open(comments_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Filename', 'Comment'])  # Header row
            writer.writerow([full_dir_name, args.comment])


if __name__ == "__main__":
    main()

#EXAMPLE USAGE:
'''
python3 multi_head_training_script.py \
  --batch_size 256 \
  --outer_epoch_schedule 40 0 0 0 0 0 0 0 30\
  --inner_epochs_1 1 \
  --inner_epochs_2 5 \
  --lr 2e-4 \
  --save_increment 1000 \
  --dtype MNIST \
  --lambda_reg 0.1 \
  --variance_threshold_scaling 1.2 \
  --base_points 'datasets/fixed_base_tensor_data/base_point_tensor.pt' \
  --mapping_path 'datasets/fixed_base_tensor_data/pure_013468_mappings.pt' \
  --base_logic 'FixedBase' \
  --sparsity_reg 50 \
  --warm_start_length 2\
  --comment 'lowering inner epoch 1'
'''

'''
python3 multi_head_training_script.py \
  --batch_size 256 \
  --outer_epochs 10 \
  --inner_epochs_1 5 \
  --inner_epochs_2 10 \
  --lr 1e-4 \
  --save_increment 10000 \
  --dtype MNIST \
  --base_points 'datasets/vary_base_tensor_data/0_4_8_LBCM_base_points.pt' \
  --mapping_path 'datasets/vary_base_tensor_data/0_4_8_LBCM_mappings.pt' \
  --base_logic 'VaryBase' \
'''
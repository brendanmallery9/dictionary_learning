
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
import scipy as sp
from scipy import stats, linalg, optimize
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

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
from self_supervised_training import *
from self_supervised_cuda_utils import *

import argparse


def save_args(args, directory):
    os.makedirs(directory, exist_ok=True)
    args_file = os.path.join(directory, "training_args.csv")
    with open(args_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Argument', 'Value'])
        for arg, val in vars(args).items():
            writer.writerow([arg, val])

def parse_args():
    parser = argparse.ArgumentParser(description="Run DDP training.")
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')

    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--outer_epochs', type=int, default=30)
    parser.add_argument('--inner_epochs_1', type=int, default=5)
    parser.add_argument('--inner_epochs_2', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_increment', type=int, default=100000)
    parser.add_argument('--dtype', type=str, default='MNIST')
    parser.add_argument('--filename', type=str, default=f'training_runs/{date_str}_runs')
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--variance_threshold', type=float, default=100.0)

    # Add parameters for reg_schedule if you want to customize from CLI
    parser.add_argument('--reg_base_value', type=float, default=0.0)
    parser.add_argument('--reg_scale_factor', type=float, default=1.0)
    parser.add_argument('--reg_min_i', type=int, default=0)

    parser.add_argument('--mapping_path', type=str, default='datasets/tensor_data/LBCM_0_4_8_mappings.pt')

    return parser.parse_args()

def main():
    args = parse_args()

    QP_reg_schedule = reg_schedule(base_value=args.reg_base_value,
                                  scale_factor=args.reg_scale_factor,
                                  min_i=args.reg_min_i)

    dim = 2
    no_heads = 5
    dropout_prob = 0.5

    dtype = args.dtype  # Use CLI argument
    model_class = FatFourLayer_Net_Multihead
    model_init_args = (dim, no_heads, dropout_prob)
    model_state_dict = None

    # Load mapping tensor with error handling
    try:
        mapping_tensor = torch.load(args.mapping_path)
        base_point_tensor=torch.load('datasets/tensor_data/base_point_tensor.pt')
    except Exception as e:
        print(f"Error loading mapping tensor from {args.mapping_path}: {e}")
        return
    inner_epochs_pair = (args.inner_epochs_1, args.inner_epochs_2)

    run_ddp_training(
        model_class=model_class,
        model_state_dict=model_state_dict,
        model_init_args=model_init_args,
        mapping_tensor=mapping_tensor,
        base_point_tensor=base_point_tensor,
        world_size=args.world_size,
        batch_size=args.batch_size,
        outer_epochs=args.outer_epochs,
        inner_epochs_pair=inner_epochs_pair,
        QP_reg_schedule=QP_reg_schedule,
        save_increment=args.save_increment,
        dtype=dtype,
        filename=args.filename,
        lr=args.lr,
        variance_threshold=args.variance_threshold,
        lambda_reg=args.lambda_reg
    )
    save_args(args, args.filename)

if __name__ == "__main__":
    main()

#EXAMPLE USAGE:
'''
python3 self_supervised_cuda_script.py \
  --world_size 4 \
  --batch_size 256 \
  --outer_epochs 30 \
  --inner_epochs_1 5 \
  --inner_epochs_2 10 \
  --lr 1e-5 \
  --save_increment 100000 \
  --dtype MNIST \
  --mapping_path datasets/tensor_data/LBCM_0_4_8_mappings.pt
'''
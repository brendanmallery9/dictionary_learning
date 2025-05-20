import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from scipy import stats, linalg
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fsolve, least_squares, minimize, curve_fit
import cvxpy as cp
import tensorflow as tf
import scipy as sp
from OT_utils import *
from joblib import Parallel, delayed
import pickle
import csv
import datetime
from models.multi_head_dir.multi_head_model import *
from models.multi_head_dir.multi_head_training import *
from visualizations import *
import gc
from self_supervised_training import *
from generate_data import *
import math


dim=2
no_heads=5
dropout_prob=0.5

model=FatFourLayer_Net_Multihead(d=dim,no_heads=no_heads,dropout_prob=dropout_prob)
#model.load_state_dict(torch.load('.......'))

dtype='.....'
#ex: dtype='one_two_three_cvx_randomref_(8000,200)'

data_directory='....'
#ex: 'datasets/MNIST_data/0_1_2_convex_hull.npy'

embedded_data_list=np.load('....',allow_pickle=True)
#ex: embedded_data_list=np.load(data_directory,allow_pickle=True)
semi_supervised_list=[]
for i in embedded_data_list:
    semi_supervised_list.append(semi_supervised_embedded_data(i,{}))

date_str = datetime.datetime.now().strftime('%Y-%m-%d')

#outer_epoch schedule should be a list of integers,with the same length as no_heads
outer_epoch_schedule=[30,20,0,0,20]
inner_epochs=[5,10]

#QP_reg schedule adds regularization to the QP matrix if desired
#The regularization is a matrix of the form a_i Id, where a_i is a scalar depending on iteration i
#a_0=base_value. If this is zero, then no regularization is applied
#min_i is the minimal i before a_0 starts decreasing
#for all i > min_i, a_i=a_{i-1}*scale_factor
#ex: QP_reg_schedule=reg_schedule(base_value=0.0000005, scale_factor=1,min_i=0)
QP_reg_schedule=reg_schedule(base_value=0, scale_factor=1,min_i=0)
lambda_reg=0
batch_size=256
filter_ratio=.3
rescale_gens=False
#filename='{}_{}_self_supervised_lambda_reg_{}'.format(date_str,dtype,lambda_reg)
filename='training_runs/{}_{}_self_supervised_test'.format(date_str,dtype,lambda_reg)

model,semi_supervised_list,basis_list,paths=semi_supervised_pipeline_v4(model,embedded_data_list,outer_epoch_schedule,inner_epochs,filter_ratio,rescale_gens,batch_size,QP_reg_schedule,dtype,filename,)

#model=FatFourLayer_Net_Multihead(d=2,no_heads=5,dropout_prob=0)
#semi_supervised_plot_atoms(dir, model, 1000, 2,True)
#semi_supervised_plot_all(dir, model, 2000, 2,True)

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
import scipy as sp
from OT_utils import *
from joblib import Parallel, delayed
import pickle
import csv
from models.multi_head_dir.multi_head_model import *
from models.multi_head_dir.multi_head_training import *
from visualizations import *

#EXAMPLE USAGE:
#generator_dirs=['datasets/MNIST_data/0/','datasets/MNIST_data/4/','datasets/MNIST_data/8/']
#embedded_data_list_random_reference=generate_convex_combination_random_reference(400,generator_dirs,2,3000)

def center_tensor(data_tensor,shift):
    mean = data_tensor.mean(dim=0, keepdim=True)  # shape [1, d]
    centered = data_tensor - mean + shift  # broadcasted subtraction
    return centered

def center_array(data_array,shift):
    mean = np.mean(data_array, axis=0, keepdims=True)  # shape (1, d)
    centered = data_array - mean + shift # broadcast subtraction over rows
    return centered

def generate_centered_convex_combination_tensor(base_measure,generators):
    m=len(generators)
    random_coeff=sample_from_simplex(m-1)
    scaled_map_array=[]
    for j in range(m):
       # gen=measure(center_array(generators[j].points),generators[j].masses)
        gen=measure(generators[j].points,generators[j].masses)
        mapping=wass_map(base_measure,gen,'emd')
        scaled_mapping=np.array(random_coeff[j]*mapping)
        scaled_map_array.append(scaled_mapping)
    combined_map=np.sum(scaled_map_array,axis=0)
    base_measure=measure(torch.tensor(base_measure.points,dtype=torch.float32),torch.tensor(base_measure.masses,dtype=torch.float32))
    return embedded_data(torch.tensor(combined_map,dtype=torch.float32),0,[],base_measure,random_coeff)

def select_random_mnist(base_supp_size,generator_dirs,dim,n,base_shift_list):
    #base_supp_size: integer
    #generator_dirs: list of strings
    #dim: integer
    #n: integer
    #base_shift_list: list of vectors
    embedded_data_list_of_lists=[]
    random_numbers_list=[]
    for i in range(n):
        random_numbers=random.sample(range(1,3000),len(generator_dirs))
        random_numbers_list.append(random_numbers)
    for k in range(len(base_shift_list)):
        k_data=[]
        for i in range(n):
            if i%100==0:
                print('{}/{}'.format(i,n))
            base_measure=measure(center_array(np.random.rand(base_supp_size,dim),base_shift_list[k])
                                    ,np.ones(base_supp_size)/base_supp_size)
            #base_measure=measure(np.random.rand(base_supp_size,dim)
            #                       ,np.ones(base_supp_size)/base_supp_size)
            random_numbers=random_numbers_list[i]
            for j in range(len(generator_dirs)):
                reference_data=np.load('{}/{}.pkl'.format(generator_dirs[j],random_numbers[j]), allow_pickle=True).data
                reference_measure=image_to_empirical(reference_data)
                mapping=wass_map(base_measure,reference_measure,'emd')
                coeff=torch.zeros(len(generator_dirs))
                coeff[j]=1
                j_data=embedded_data(torch.tensor(mapping,dtype=torch.float32),generator_dirs[j],[],base_measure,coeff)
                k_data.append(j_data)
        embedded_data_list_of_lists.append(k_data)
    return embedded_data_list_of_lists

#EXAMPLE USAGE:
#generator_dirs=['datasets/MNIST_data/0/','datasets/MNIST_data/4/','datasets/MNIST_data/8/']
#base_shift_list=[np.array([0,0]),np.array([0.5,0.5])]
#embedded_data=select_random_mnist(400,generator_dirs,2,2000,base_shift_list)

def generate_centered_convex_combination_random_reference_tensor(base_supp_size,generator_dirs,dim,n,base_shift):
    embedded_data_list=[]
    for i in range(n):
        if i%100==0:
            print('{}/{}'.format(i,n))
        random_numbers=random.sample(range(1,3000),3)
        base_measure=measure(center_array(np.random.rand(base_supp_size,dim),base_shift)
                                ,np.ones(base_supp_size)/base_supp_size)
        #base_measure=measure(np.random.rand(base_supp_size,dim)
         #                       ,np.ones(base_supp_size)/base_supp_size)
        reference_measures=[]
        for j in range(len(generator_dirs)):
            reference_data=np.load('{}/{}.pkl'.format(generator_dirs[j],random_numbers[j]), allow_pickle=True).data
            reference_measure=image_to_empirical(reference_data)
            reference_measures.append(reference_measure)
        embedded_data_list.append(generate_centered_convex_combination_tensor(base_measure,reference_measures))
    return embedded_data_list
#EXAMPLE USAGE:
#generator_dirs=['datasets/MNIST_data/0/','datasets/MNIST_data/4/','datasets/MNIST_data/8/']
#embedded_data_list_random_reference=generate_convex_combination_random_reference(400,generator_dirs,2,3000)

def generate_centered_convex_combination_tensor(base_measure,generators):
    m=len(generators)
    random_coeff=sample_from_simplex(m-1)
    scaled_map_array=[]
    for j in range(m):
       # gen=measure(center_array(generators[j].points),generators[j].masses)
        gen=measure(generators[j].points,generators[j].masses)
        mapping=wass_map(base_measure,gen,'emd')
        scaled_mapping=np.array(random_coeff[j]*mapping)
        scaled_map_array.append(scaled_mapping)
    combined_map=np.sum(scaled_map_array,axis=0)
    base_measure=measure(torch.tensor(base_measure.points,dtype=torch.float32),torch.tensor(base_measure.masses,dtype=torch.float32))
    return embedded_data(torch.tensor(combined_map,dtype=torch.float32),0,[],base_measure,random_coeff)


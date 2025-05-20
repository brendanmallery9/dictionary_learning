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

class semi_supervised_embedded_data:
    def __init__(self,embedded_data,basis_dict):
        #embedded_data: embedded_data object
        #basis_dict: dictionary of learned maps, indexed by output head number
        self.embedded_data=embedded_data
        self.basis_dict=basis_dict

#a reduced version of semi_supervised_data, which does not reference embedded_data structure
class light_semi_supervised_data:
    def __init__(self,base_points,mapping,basis_dict):
        #embedded_data: embedded_data object
        #basis_dict: dictionary of learned maps, indexed by output head number
        self.base_points=base_points
        self.mapping=mapping
        self.basis_dict=basis_dict

def project_vectors(mapping_vec, basis_vec):
    #mapping_vec: (base_supp_size,d)
    #basis_vec: (base_supp_size,d)
    if not isinstance(mapping_vec, torch.Tensor):
        mapping_vec = torch.tensor(mapping_vec, dtype=torch.float32)

    dot_products = (mapping_vec * basis_vec).sum(dim=1) #shape: (base_supp_size)
    v_norm_sq = (basis_vec ** 2).sum(dim=1) #shape: (base_supp_size)
    coeffs = (dot_products / v_norm_sq).unsqueeze(1) #shape: (base_supp_size,1)
    projections = coeffs * basis_vec #shape: (N,d)
    return projections

def filter_by_proj(semi_supervised_data, basis_idx, filter_ratio):
    #basis_idx: integer in class_nos
    #filter_ratio: percentage of length of semi_supervised_data we should remove
    #e.g. filter_ratio=0 --> no filtering happens
    proj_dict = {}
    with torch.no_grad():
        for data in semi_supervised_data:
            basis_vec = data.basis_dict[basis_idx]  # (N, d)
            mapping_vec = torch.tensor(data.embedded_data.mapping)            # (N, d)

            proj = project_vectors(mapping_vec, basis_vec)  # (N, d)

            # Normalize
            proj = proj / proj.norm(dim=1, keepdim=True) # (N,d)
            mapping_norm = mapping_vec / mapping_vec.norm(dim=1, keepdim=True) #(N,d)

            # Compute residual and its norm
            residual =mapping_norm - proj #(N,d)
            norm_residual = residual.norm() #scalar
            proj_dict[data] = norm_residual.item()

        proj_dict = dict(sorted(proj_dict.items(), key=lambda item: item[1], reverse=True))
        filter_range=len(semi_supervised_data)-math.floor(len(semi_supervised_data)*filter_ratio)
        filtered_data = list(proj_dict)[:filter_range]
    return filtered_data

def tensor_ss_data(data_list):
    if type(data_list[0])==semi_supervised_embedded_data:
        tensorized_data=[light_semi_supervised_data(
            base_points=torch.as_tensor(data.embedded_data.base.points,dtype=torch.float32),
            mapping=torch.as_tensor(data.embedded_data.mapping,dtype=torch.float32),
            basis_dict={
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in data.basis_dict.items()})
                for data in data_list]
    if type(data_list[0])==light_semi_supervised_data:
            tensorized_data=[light_semi_supervised_data(
                base_points=torch.as_tensor(data.base_points,dtype=torch.float32),
                mapping=torch.as_tensor(data.mapping,dtype=torch.float32),
                basis_dict={
                    k: torch.as_tensor(v, dtype=torch.float32)
                    for k, v in data.basis_dict.items()})
                    for data in data_list]
    return tensorized_data

def compute_maps(list_of_base_supports,target_points):
    def process_wass_maps(base_points,target_points):
        base_measure=measure(base_points,np.ones(len(base_points))/len(base_points))
        target_measure=measure(target_points,np.ones(len(target_points))/len(target_points))
        mapping=wass_map(base_measure,target_measure,'emd')
        return mapping
    map_list=Parallel(n_jobs=-1)(delayed(process_wass_maps)(list_of_base_supports[i], target_points) for i in range(len(list_of_base_supports)))
    return map_list


def compute_OT(map_list_1,map_list_2):
    if len(map_list_1) != len(map_list_2):
        raise ValueError("The two lists must have the same length.")
    def process_OT(map_1,map_2):
        map_1_masses=np.ones(len(map_1))*(1/len(map_1))
        map_2_masses=np.ones(len(map_2))*(1/len(map_2))
        distance_mat=np.linalg.norm(map_1[:, None] - map_2[None, :], axis=2)
        return ot.emd2(map_1_masses, map_2_masses, distance_mat)
    results=Parallel(n_jobs=-1)(delayed(process_OT)(map_list_1[i], map_list_2[i]) for i in range(len(map_list_1)))
    return results



def find_nearest_generator(model,no_gens,data_list,forbidden_heads,method):
    #maps should be centered for rescaling to make sense
    #method: OT or L2
    model.eval()
    base_point_list=[]
    true_map_list=[]
    model_map_list=[]
    #true_map_list is a list of .mapping data
    #model_map_list is the outputs of the model applied 
    for i in range(len(data_list)):
        if type(data_list[i])==embedded_data:
            base_point_i=torch.as_tensor(data_list[i].base.points).float()
        else:
            base_point_i=torch.as_tensor(data_list[i].base_points).float()
        base_point_list.append(base_point_i)
        model_map_all_heads=model(base_point_i)
        i_model_dict={}
        for j in range(no_gens):
            model_map_j=model_map_all_heads[:,j,:]
            i_model_dict[j]=model_map_j.detach().numpy()
        model_map_list.append(i_model_dict)
        true_map_i=data_list[i].mapping
        true_map_list.append(np.array(true_map_i))

    dist_dict={}

    for j in range(no_gens):
        if j not in forbidden_heads:
            print('Finding closest generators to head no:',j)
           # if rescaled==True:
           #     var_list=[]
           #     for map in true_map_list:
           #         centered=map-np.mean(map,axis=0,keepdims=True)
           #         variance = np.sum(centered ** 2)
           #         var_list.append(variance)
           #     scale=np.mean(variance)
           #     j_map_list=[np.array(rescale_tensor_to_fixed_variance(model_map_list[i][j],scale)) for i in range(len(model_map_list))]
            j_map_list=[model_map_list[i][j] for i in range(len(model_map_list))]
            if method=='OT':
                OT_list=compute_OT(j_map_list,true_map_list)
                dist_dict[j]=OT_list
            if method=='L2':
                L2_list=compute_L2(j_map_list,true_map_list,base_point_list)
                dist_dict[j]=L2_list
    large_dist_dict={}
    for i in range(len(data_list)):
        i_dist_dict={}
        for j in range(no_gens):
            if j not in forbidden_heads:
                i_dist_dict[j]=dist_dict[j][i]
        large_dist_dict[i]=i_dist_dict
    best_gens={}
    for j in range(no_gens):
        if j not in forbidden_heads:
            sorted_by_j = dict(sorted(large_dist_dict.items(), key=lambda item: item[1][j]))
            first_key, first_value = next(iter(sorted_by_j.items()))
            best_gens[j]=[first_key,first_value[j]]
    return best_gens


def find_best_model_idx(model,no_gens,data_list,forbidden_heads,method):
    #maps should be centered for rescaling to make sense
    #method: OT or L2
    model.eval()
    base_point_list=[]
    true_map_list=[]
    model_map_list=[]
    #true_map_list is a list of .mapping data
    #model_map_list is the outputs of the model applied 
    for i in range(len(data_list)):
        if type(data_list[i])==embedded_data:
            base_point_i=torch.as_tensor(data_list[i].base.points).float()
        else:
            base_point_i=torch.as_tensor(data_list[i].base_points).float()
        base_point_list.append(base_point_i)
        model_map_all_heads=model(base_point_i)
        i_model_dict={}
        for j in range(no_gens):
            model_map_j=model_map_all_heads[:,j,:]
            i_model_dict[j]=model_map_j.detach().numpy()
        model_map_list.append(i_model_dict)
        true_map_i=data_list[i].mapping
        true_map_list.append(np.array(true_map_i))

    dist_dict={}

    for j in range(no_gens):
        if j not in forbidden_heads:
            print('Finding closest generators to head no:',j)
            j_map_list=[model_map_list[i][j] for i in range(len(model_map_list))]
            if method=='OT':
                OT_list=compute_OT(j_map_list,true_map_list)
                dist_dict[j]=OT_list
            if method=='L2':
                L2_list=compute_L2(j_map_list,true_map_list,base_point_list)
                dist_dict[j]=L2_list
    min_dists={}
    for j in range(no_gens):
        if j not in forbidden_heads:
            min_dists[j]=min(dist_dict[j])
    min_key = min(min_dists, key=min_dists.get)
    return min_dists,min_key

def learn_generators(embedded_data_list,model,atom_search_pattern,forbidden_heads,method):
    #rescaled: boolean
    basis_dict={}
    score_dict={}
    if atom_search_pattern=='None':
        #print('model, basis_dict')
        return basis_dict
    if atom_search_pattern=='Nearest_Neighbor':
        no_gens=model.no_heads
        #if rescaled==True:
        #    gen_dict=find_nearest_generator_rescaled(model,no_gens,embedded_data_list,forbidden_heads)
        #else:
        gen_dict=find_nearest_generator(model,no_gens,embedded_data_list,forbidden_heads,method)
        print(gen_dict)
        for i in range(no_gens):
            if i not in forbidden_heads:
                basis_dict[i]=embedded_data_list[gen_dict[i][0]]
                score_dict[i]=gen_dict[i][1]
        return basis_dict,score_dict
    
def compute_L2(map_list_1, map_list_2, base_point_list):
    if len(map_list_1) != len(map_list_2) or len(map_list_1) != len(base_point_list):
        raise ValueError("All input lists must have the same length.")

    # Convert all entries to tensors if they are not already
    map_list_1 = [torch.as_tensor(x) for x in map_list_1]
    map_list_2 = [torch.as_tensor(x) for x in map_list_2]
    base_point_list = [torch.as_tensor(x) for x in base_point_list]

    results = []
    for i in range(len(map_list_1)):
        avg_dot = torch.einsum(
            'nd,nd->n',
            map_list_1[i] - base_point_list[i],
            map_list_2[i] - base_point_list[i]
        ).mean()
        results.append(avg_dot)
    return results

#update_semi_supervised_embedding updates a list of semi_supervised_embeddings by adding the entry {map_index: target_mapping} to the basis_dicts
def update_semi_supervised_embedding(data_list,target_mapping,map_index):
    #embedded_data_list: list of embedded_data objects
    #target_mapping: target mapping
    if type(data_list[0])==semi_supervised_embedded_data:
        base_supp_list=[np.array(data_list[i].embedded_data.base.points) for i in range(len(data_list))]
    if type(data_list[0])==light_semi_supervised_data:
        base_supp_list=[np.array(data_list[i].base_points) for i in range(len(data_list))]
    print(np.shape(target_mapping))
    target_mapping = target_mapping.detach().numpy()
    print(target_mapping.shape)
    map_list=compute_maps(base_supp_list,target_mapping)
    for i in range(len(data_list)):
        data_list[i].basis_dict[map_index]=torch.as_tensor(map_list[i])
    return data_list


def rescale_tensor_to_fixed_variance(points, target_variance):
    points=torch.as_tensor(points)
    original_center = points.mean(dim=0, keepdim=True)
    centered = points - original_center
    current_variance = torch.sum(centered ** 2)
    if current_variance > 0:
        scale = torch.sqrt(torch.tensor(target_variance, dtype=points.dtype, device=points.device) / current_variance)
        rescaled = centered * scale
    else:
        rescaled = centered  
    final_points = rescaled + original_center
    return final_points
#EXAMPLE USAGE:
#Given embedded_data_list, generates semi_supervised_embeddings with basis_dicts {0:target_mapping_1, 2:target_mapping_2}

#semi_supervised_embeddings=[]
#for i in range(len(embedded_data_list)):
#    semi_supervised_embeddings.append(semi_supervised_embedded_data(embedded_data_list[i],{}))

#target_mapping_1=...
#target_mapping_2=...
#semi_supervised_embeddings=update_semi_supervised_embedding(semi_supervised_embeddings,target_mapping_1,0)
#semi_supervised_embeddings=update_semi_supervised_embedding(semi_supervised_embeddings,target_mapping_2,2)

def semi_supervised_solve_for_coefficients(model,base_points,mapping,supervised_class_nos,basis_dict,QP_reg,device):
    #base_points: (supp_size,dim)
    #mapping: (supp_size,dim)
    #supervised_class_nos: list of supervised class numbers
    #basis_dict: dictionary of learned maps, indexed by output head number
    base_points=torch.as_tensor(base_points)
    mapping=mapping.cpu().numpy()
    output=model(base_points).detach().cpu().numpy() #output: (supp_size,no_heads,dim)
    no_heads=output.shape[1]
    supp_size=base_points.shape[0]
    map_array=[]
    for i in range(no_heads):
        if i in supervised_class_nos:
            i_basis=basis_dict[i].cpu().numpy()
            tensor_map=i_basis-mapping #tensor_map: (supp_size,dim)
            map_array.append(tensor_map)
        else:
            tensor_map=output[:,i,:]-mapping
            map_array.append(tensor_map)
    #build and solve QP
    QP_mat=build_QP(map_array,supp_size)+QP_reg*np.eye(len(map_array))
    coefficients,min_value=solve_QP(QP_mat)
    return coefficients,min_value

def batch_solve_for_coefficients_qpth(model,batch_base_points,batch_mapping,supervised_class_nos,batch_basis,QP_reg,device):
    #batch_base_points: (batch_size,supp_size,dim)
    #batch_mapping: (batch_size,supp_size,dim)
    #supervised_class_nos: list of supervised class numbers
    #batch_basis: (batch_size,supp_size,no_heads,dim)
    with torch.no_grad():
        output=model(batch_base_points) #output: (batch_size,no_heads,supp_size,dim)
    batch_size=output.shape[0]
    no_heads=output.shape[1]
    supp_size=batch_base_points.shape[2]
    displacement_tensor=[]
    for i in range(no_heads):
        if i in supervised_class_nos:
            i_basis=batch_basis[:,:,i,:] #shape (batch_size,supp_size,dim)
            tensor_map=i_basis-batch_mapping #tensor_map: (batch_size,supp_size,dim)
            displacement_tensor.append(tensor_map)
        else:
            tensor_map=output[:,i,:,:]-batch_mapping
            displacement_tensor.append(tensor_map)
    displacement_tensor=torch.stack(displacement_tensor,dim=1) #displacement_tensor: (batch_size,no_heads,supp_size,dim)
    QP=build_QP_v2(displacement_tensor)+QP_reg*torch.eye(no_heads).expand(batch_size, no_heads, no_heads)
    QP=QP.to('cpu')

    lin_term = torch.zeros(batch_size, no_heads, device='cpu') #QP linear term = 0

    #nonnegativity constraints
    nonneg = -torch.eye(no_heads, device='cpu').unsqueeze(0).expand(batch_size, no_heads, no_heads)
    zero_vec = torch.zeros(batch_size, no_heads, device='cpu')

    #normalization constraints
    norm = torch.ones(batch_size, 1, no_heads, device='cpu')
    ones_vec = torch.ones(batch_size, 1, device='cpu')

    solver = qpth.qp.QPFunction()

    #calls solver(QP,p,nonneg,zero_vec,norm,ones_vec)
    #solves min_v v^T QP v with v\geq 0 and \sum v=1

    v = solver(QP, lin_term, nonneg, zero_vec, norm, ones_vec)  # shape: (b, n)
    return v

def batch_solve_for_coefficients_cvxpy(model, batch_base_points, batch_mapping, supervised_class_nos, batch_basis, QP_reg, device):
    # batch_base_points: (batch_size, supp_size, dim)
    # batch_mapping: (batch_size, supp_size, dim)
    # supervised_class_nos: list of supervised class numbers
    # batch_basis: (batch_size, supp_size, no_heads, dim)

    with torch.no_grad():
        output = model(batch_base_points)  # (batch_size, no_heads, supp_size, dim)

    batch_size = output.shape[0]
    no_heads = output.shape[1]
    supp_size = batch_base_points.shape[1]

    displacement_tensor = []
    for i in range(no_heads):
        if i in supervised_class_nos:
            i_basis = batch_basis[:, :, i, :]  # (batch_size, supp_size, dim)
            tensor_map = i_basis - batch_mapping
        else:
            tensor_map = output[:, i, :, :] - batch_mapping
        displacement_tensor.append(tensor_map)

    # (batch_size, no_heads, supp_size, dim)
    displacement_tensor = torch.stack(displacement_tensor, dim=1)

    # Build QP tensor: (batch_size, no_heads, no_heads)
    QP = build_QP_v2(displacement_tensor)
    QP += QP_reg * torch.eye(no_heads).expand(batch_size, no_heads, no_heads)
    QP = QP.to('cpu')

    v_solutions = []

    for b in range(batch_size):
        Q = QP[b].numpy()  # Do not enforce symmetry

        v = cp.Variable(no_heads)
        objective = cp.Minimize(0.5 * cp.quad_form(v, Q))
        constraints = [v >= 0, cp.sum(v) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if v.value is None:
            v_solutions.append(np.ones(no_heads) / no_heads)
        else:
            v_solutions.append(v.value)

    # Convert back to torch tensor on the correct device
    v_tensor = torch.tensor(np.array(v_solutions), dtype=torch.float32).to(device)  # (batch_size, no_heads)
    return v_tensor

def positive_soft_thresholding(x, tau):
    x_thresh = torch.clamp(x - tau, min=0.0)
    return x_thresh

def compute_stepsize(base_point_tensor,model,supervised_class_nos,supervised_basis):
    #base_point_tensor: (batch_size,supp_size,dim)
    #supervised_class_nos: (n,supp_size,no_heads,dim)
    n=base_point_tensor.shape[0]
    no_heads=model.no_heads
    eig_total=0
    for i in range(n):

        image = model(base_point_tensor[i]).to('cpu')  # (supp_size, no_heads, dim)
        #replace heads with supervised bases
        for j in range(no_heads):
            if j in supervised_class_nos:
                image[:,j,:]=supervised_basis[i,:,j,:].to('cpu') #(supp_size,dim)                
        image_image = torch.einsum('ikd,jkd->ijk', image, image) # (no_heads, no_heads, supp_size)
        image_image = image_image.mean(dim=-1)# (no_heads, no_heads)
        max_eig=torch.linalg.eigvalsh(image_image)[-1]
        eig_total+=max_eig
    eig_total=eig_total/n
    return 1/eig_total**2

def avg_squash(
    x,radius,center,eps: float = 1e-8) -> torch.Tensor:
    #x: (batch_size,supp_size,dim)
    #radius: desired radius of ball
    #center: (dim)
    center_exp = center.view(1, 1, -1) #(1,1,dim)-->(batch_size, supp_size, dim)
    offset = x - center_exp  # (batch_size, supp_size, dim)
    point_norms = offset.norm(dim=-1)  #(batch_size, supp_size)
    mean_norms = point_norms.mean(dim=1, keepdim=True) #(batch_size,1)
    mean_norms = mean_norms.unsqueeze(-1) # bring it to (batch_size, 1, 1) for broadcasting over points & dims
    scaled = radius * offset / (1.0 + mean_norms + eps) # (batch_size, supp_size, dim)
    return center_exp + scaled

def squash_per_point(
    x, radius, center, eps: float = 1e-8) -> torch.Tensor:
    # x: (batch_size, supp_size, dim)
    # radius: desired radius of ball
    # center: (dim)
    center_exp = center.view(1, 1, -1)  # (1, 1, dim) for broadcasting
    offset = x - center_exp             # (batch_size, supp_size, dim)
    point_norms = offset.norm(dim=-1, keepdim=True)  # (batch_size, supp_size, 1)
    # Apply your smooth squashing formula pointwise
    scaled = radius * offset / (1.0 + point_norms + eps)  # (batch_size, supp_size, dim)
    
    return center_exp + scaled


def softmax_squash(x, radius, center, alpha=10.0, eps=1e-8):
    center_exp = center.view(1, 1, -1)
    offset = x - center_exp
    point_norms = offset.norm(dim=-1)  # (batch_size, supp_size)

    # Softmax-weighted max (logsumexp approximation)
    soft_max_norm = (1.0 / alpha) * torch.logsumexp(alpha * point_norms, dim=1, keepdim=True)
    soft_max_norm = soft_max_norm.unsqueeze(-1)  # (batch_size, 1, 1)

    scaled = radius * offset / (soft_max_norm + eps)
    return center_exp + scaled


def batch_LASSO(base_point_tensor,mapping_tensor,model,supervised_class_nos,supervised_basis,stepsize,lambda_reg,device,iterations):
    #base_point_tensor: (batch_size,supp_size,dim)
    #mapping_tensor: (batch_size,supp_size,dim)
    #model: no_heads headed neural net
    #supervised_class_nos: list of integers
    #supervised_basis: dictionary indexed by supervised_class_nos
    #stepsize: scalar

    n=base_point_tensor.shape[0] #batch_size

    no_heads=model.no_heads

    image_image_big = torch.zeros(n, no_heads, no_heads, device=device)
    image_data_big = torch.zeros(n, no_heads, 1, device=device)
    for i in range(n):
        image = model(base_point_tensor[i]).to(device)  # (supp_size, no_heads, dim)
        #replace heads with supervised bases
                    #semi_supervised_basis: tensor of shape (batch_size,support_size,no_heads,dim)

        for j in range(no_heads):
            if j in supervised_class_nos:
                image[:,j,:]=supervised_basis[i,:,j,:].to(device) #(supp_size,dim)
        data = mapping_tensor[i] # (supp_size,dim)

        image_image = torch.einsum('ijd,ikd->ijk', image, image) # (supp_size,no_heads, no_heads)
        image_image = image_image.mean(dim=0)# (no_heads, no_heads)
        image_data = torch.einsum('ikd,id->k', image, data) / image.shape[1] # (no_heads)

        #Store tensors in big tensors
        image_image_big[i] = image_image # image_image_big: (batch_size, no_heads, no_heads)
        image_data_big[i,:,0] = image_data # image_data_big: (batch_size, no_heads, 1)
    
    weights=torch.zeros(n,no_heads,1,device=device) #weights: (no_heads,batch_size)

    for i in range(iterations):

        residual=image_image_big @ weights - image_data_big #residual: (no_heads)
        grad=image_image_big.transpose(1,2)@residual
        weights=weights-stepsize*grad
        weights = positive_soft_thresholding(weights, stepsize * lambda_reg)
    return weights


def batch_FISTA(base_point_tensor, mapping_tensor, model, supervised_class_nos, supervised_basis,
                stepsize, lambda_reg, device, iterations, tolerance=1e-5):
    n = base_point_tensor.shape[0]
    no_heads = model.no_heads

    image_image_big = torch.zeros(n, no_heads, no_heads, device=device)
    image_data_big = torch.zeros(n, no_heads, 1, device=device)

    for i in range(n):
        image = model(base_point_tensor[i]).to(device)  # (supp_size, no_heads, dim)

        for j in range(no_heads):
            if j in supervised_class_nos:
                image[:, j, :] = supervised_basis[i, :, j, :].to(device)  # (supp_size, dim)

        data = mapping_tensor[i]  # (supp_size, dim)

        image_image = torch.einsum('ijd,ikd->ijk', image, image).mean(dim=0)  # (no_heads, no_heads)
        image_data = torch.einsum('ikd,id->k', image, data) / image.shape[1]  # (no_heads,)

        image_image_big[i] = image_image
        image_data_big[i, :, 0] = image_data

    weights = torch.zeros(n, no_heads, 1, device=device)
    y = weights.clone()
    t = 1.0

    for i in range(iterations):
        weights_old = weights.clone()

        residual = image_image_big @ y - image_data_big
        grad = image_image_big.transpose(1, 2) @ residual

        new_weights = y - stepsize * grad
        new_weights = positive_soft_thresholding(new_weights, stepsize * lambda_reg)

        t_next = 0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)
        y = new_weights + ((t - 1) / t_next) * (new_weights - weights)

        # Check convergence
        diff = new_weights - weights_old
        avg_diff = diff.mean(dim=0)  # average over the first dimension (batch size)
        rel_change = avg_diff.norm() / (weights_old.mean(dim=0).norm() + 1e-8)
        if rel_change.item() < tolerance:
            print(f"FISTA converged at iteration {i}")
            break

        weights = new_weights
        t = t_next

    return weights

def optimization_step(displacement_tensor,coefficient_tensor,optimizer,lambda_reg,device):
    #batches here should be comprised of different embeddings
    # displacement_tensor: (batch_size,supp_size,no_heads,d)
    # coefficient_tensor: (batch_size, no_heads) → coefficients for each head
    # mapping_tensor: (batch_size, supp_size,dim) → batch of original maps applied to points in R^d
    #lambda_reg: scalar

    #outer_product_tensor: (batch_size, supp_size, no_heads,no_heads)
    outer_product_tensor = displacement_tensor @ displacement_tensor.transpose(-1, -2) 
    #mean_displacement: sums over supp_size, returning (batch_size,no_heads,dim)
    mean_displacement = displacement_tensor.sum(dim=1)
    optimizer.zero_grad()
    #result has shape (batch_size,supp_size)
    result=torch.einsum('ik,ijkl,il->ij',coefficient_tensor,outer_product_tensor,coefficient_tensor).to(device)
    main_loss=result.mean().to(device)
    reg_term=lambda_reg*torch.norm(mean_displacement).to(device)
    loss=main_loss+reg_term
    loss.backward()
    optimizer.step()
    return loss

def optimization_step_2(i_semi_supervised_basis, mapping_tensor, coefficients_tensor, optimizer, device):
    # i_semi_supervised_basis: (batch_size, supp_size, no_heads, d) — trainable
    # mapping_tensor: (batch_size, supp_size, d) — target
    # coefficients_tensor: (batch_size, no_heads) — fixed

    i_semi_supervised_basis = i_semi_supervised_basis.to(device)
    mapping_tensor = mapping_tensor.to(device)
    coefficients_tensor = coefficients_tensor.to(device)

    optimizer.zero_grad()

    # Compute weighted sum: [i, j, k, l] * [i, k] -> [i, j, l]
    weighted_sum = torch.einsum('ijkl,ik->ijl', i_semi_supervised_basis, coefficients_tensor)

    # L2^2 loss
    loss = torch.mean(torch.sum((weighted_sum - mapping_tensor) ** 2, dim=(1, 2)))

    loss.backward()
    optimizer.step()

    return loss.item()


def semi_supervised_multi_head_train_v3(model,semi_supervised_data,outer_epochs,inner_epochs_pair,QP_reg_schedule,batch_size,supervised_class_nos,save_increment,atom_reg,dtype,filename,squash_factor,lr):
    start_time=time.time()
    model.train()
    #model: multi head neural net 
    #dataset: list of semi_supervised_embedded_data objects
    #outer_epochs: int
    #inner epochs_pair: (int_1,int_2), 
    # total epochs = outer_epochs*int_1*int_2
    #QP_reg_schedule: reg_schedule object
    #batch_size: int
    #dim: int
    #save_increment: int
    #squash factor: int. if squash_factor < 0, no squashing. if squash_factor > 0, constraints solutions to ball of radius squash_factor
    #lr: float

    if len(supervised_class_nos)>0:
        for head_idx in supervised_class_nos:
            print(f"Freezing head {head_idx}")
            for param in model.heads[head_idx].parameters():
                param.requires_grad = False
            # Set these specific heads to evaluation mode
            model.heads[head_idx].eval()
        model.freeze_heads(supervised_class_nos)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    tensorized_data=tensor_ss_data(semi_supervised_data)

    #calculate variance of data
    norm_vec = []
    mean_vec = []
    for i in tensorized_data:
        mapping = i.mapping  # assume shape (n, d)
        # Compute centered L2 norm
        centered = mapping - mapping.mean(dim=0, keepdim=True)
        norm = centered.norm(dim=1).mean()
        norm_vec.append(norm)
        # Compute mean of the means of vectors (scalar)
        mean_vec.append(mapping.mean(dim=0, keepdim=True))
    # Final statistics
    avg_norm = torch.stack(norm_vec).mean().to(device)
    avg_mean = torch.stack(mean_vec).mean(dim=0).to(device)

    dataset=CustomDataset(tensorized_data,dtype)

    base_supp_size=dataset[0].base_points.shape[0]

    no_batches=int(np.ceil(len(dataset)/batch_size))
    no_heads=model.no_heads
    optimizer = optim.Adam(model.parameters(), lr=lr)
    time_now=time.time()
    os.makedirs('{}/Trial_{:.4f}'.format(filename,time_now), exist_ok=True)  
    counter=0
    inner_loss=0

    for outer_epoch in range(outer_epochs):
        dataset.permute()
        for batch in range(no_batches):
            start=time.time()
            batch_data = dataset[batch_size * batch : batch_size * (batch + 1)]

            #base_point_tensor has shape (batch_size,support_size,dim)
            #mapping_tensor has shape (batch_size,support_size,dim)

            base_point_tensor = torch.stack([d.base_points for d in batch_data]).to(device)
            mapping_tensor = torch.stack([d.mapping for d in batch_data]).to(device)

            #semi_supervised_basis: tensor of shape (batch_size,support_size,no_heads,dim)
            semi_supervised_basis=torch.zeros((len(batch_data),base_point_tensor.shape[1],no_heads,base_point_tensor.shape[2]),dtype=torch.float32,device=device)
            #populates semi_supervised_basis with maps from the supervised generators
            for j in range(no_heads):
                if j in supervised_class_nos:
                    temp_j = torch.stack([d.basis_dict[j] for d in batch_data])
                    semi_supervised_basis[:, :, j, :] = temp_j.to(device)
    #        semi_supervised_basis=torch.tensor(semi_supervised_basis).to(device)
            
            for i in range(inner_epochs_pair[0]):
                inner_loss=0

                #input_reshaped flattens base_point_tensor to shape (batch_size*support_size,dim)
                input_reshaped = base_point_tensor.reshape(-1, base_point_tensor.shape[-1])
                #output_reshaped: tensor of shape (batch_size*support_size,no_heads,dim)
                #output_reshaped = model(input_reshaped).to(device)
                QP_reg=QP_reg_schedule(outer_epoch)
                
                coefficients_tensor=batch_solve_for_coefficients_cvxpy(model,base_point_tensor,mapping_tensor,supervised_class_nos,semi_supervised_basis,QP_reg,device)
                #v2 routine
               # for j in range(len(base_point_tensor)):
                    #iterates over the batch 
                #    coefficients,_=semi_supervised_solve_for_coefficients(model,base_point_tensor[j],mapping_tensor[j],
                                                               #           supervised_class_nos,batch_data[j].basis_dict,QP_reg,device)
                 #   coefficients_list.append(coefficients)
                #coefficients_tensor: tensor of shape (batch_size,no_heads)
                coefficients_tensor=torch.tensor(coefficients_tensor,dtype=torch.float32,device=device)
                for j in range(inner_epochs_pair[1]):
                    output_reshaped = model(input_reshaped).to(device) #output_reshaped: tensor of shape (batch_size*support_size,no_heads,dim)
                    #reshapes output_tensor to (batch_size, support_size,no_heads,dim)
                    output_tensor = output_reshaped.view(base_point_tensor.shape[0], base_point_tensor.shape[1], output_reshaped.shape[1],output_reshaped.shape[2])
                    j_semi_supervised_basis = semi_supervised_basis.clone()
                    for j in range(no_heads):
                        if j not in supervised_class_nos:
                            #adds the output entries to the unsupervised class numbers
                            if squash_factor>=0:
                                j_semi_supervised_basis[:,:,j,:]=squash_per_point(output_tensor[:, :, j, :],squash_factor*avg_norm,avg_mean,1e-8)
                            else:
                                j_semi_supervised_basis[:,:,j,:]=output_tensor[:, :, j, :]
                    #displacement_tensor: (batch_size,supp_size,no_heads,d)
                    displacement_tensor=(j_semi_supervised_basis-mapping_tensor.unsqueeze(2)).to(device)
                    loss=optimization_step(displacement_tensor,coefficients_tensor,optimizer,atom_reg,device)
                    inner_loss+=loss.item()
                    if counter%save_increment==0:
                        torch.save(model.state_dict(), '{}/Trial_{:.4f}/multi_head_model_{}.pt'.format(filename,time_now,counter))
                        print(f"Model saved at counter {counter}")
                    counter=counter+1
            inner_epoch_count=inner_epochs_pair[0]*inner_epochs_pair[1]
            print(f'Epoch [{outer_epoch+1}/{outer_epochs}], Batch [{batch+1}/{no_batches}], counter {counter}, Avg Inner Loss: {inner_loss/inner_epoch_count:.8f}')
            end=time.time()
            print(end-start)

    end_time=time.time()
    save_trial('self_supervised.csv',
               dtype=dtype,
               index=time_now,
               end_loss=inner_loss,
               train_time=end_time-start_time,
               architecture=model.__class__.__name__,
               no_heads=model.no_heads,
               base_supp_size=base_supp_size,
               batch_size=batch_size,
               outer_epochs=outer_epochs,
               inner_epochs_pair=inner_epochs_pair,
               counter=counter,
               QP_base_scale_min=(QP_reg_schedule.base_value,QP_reg_schedule.scale_factor,QP_reg_schedule.min_i),
                learning_rate =lr,supervised_basis=supervised_class_nos, no_data_points=len(semi_supervised_data))
            
    torch.save(model.state_dict(), '{}/Trial_{:.4f}/final_multi_head_model_{}.pt'.format(filename,time_now,counter))
    return '{}/Trial_{:.4f}/final_multi_head_model_{}.pt'.format(filename,time_now,counter)



'''
def semi_supervised_pipeline_v3(model,embedded_data_list,
                                outer_epoch_schedule,inner_epochs_pair,
                                filter_ratio,
                                squash_factor,
                                batch_size,QP_reg_schedule,
                                dtype,filename,lr):
    #model
    #embedded_data_list: list of embedded_data objects
    #outer_epoch_schedule: list of integers (should be length=number of heads)
    #inner_epochs pair: list of two integers
    #dtype: string
    #filename:string
    basis_list=[]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    supervised_class_no_list=[]
    semi_supervised_list=[]
    model_no_heads=model.no_heads
    for i in embedded_data_list:
        semi_supervised_list.append(semi_supervised_embedded_data(i,{}))
    paths=[]
    os.makedirs(filename, exist_ok=True)

    if len(outer_epoch_schedule)>model_no_heads:
        raise RuntimeError('Error: outer_epoch_schedule has more than no_heads elements')
    
    if outer_epoch_schedule[0]>0:
        output_path=semi_supervised_multi_head_train_v3(model,semi_supervised_list,outer_epochs=outer_epoch_schedule[0],inner_epochs_pair=inner_epochs_pair,
                                         QP_reg_schedule=QP_reg_schedule,batch_size=batch_size,supervised_class_nos=supervised_class_no_list,save_increment=100000,atom_reg=0,dtype=dtype,filename=filename,squash_factor=squash_factor,lr=lr)

        paths.append(output_path)
    else:
        time_now = time.time()
        trial_dir = '{}/Trial_{:.4f}'.format(filename, time_now)
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{trial_dir}/multi_head_model_0.pt')
        paths.append('{}/multi_head_model_{}.pt'.format(trial_dir,0))

    for i in range(len(outer_epoch_schedule[1:])+1):
        torch.cuda.empty_cache()
        gc.collect()
        model.to('cpu')
        method='OT'
        new_basis,score_dict=learn_generators(embedded_data_list,model,'Nearest_Neighbor',supervised_class_no_list,method=method)
        basis_list.append(new_basis)
        filtered_score_dict = {k: v for k, v in score_dict.items() if k not in supervised_class_no_list}
        min_index = min(filtered_score_dict, key=filtered_score_dict.get)
        print('Supervising head_no:',min_index)
        semi_supervised_list=update_semi_supervised_embedding(semi_supervised_list,new_basis[min_index].mapping,min_index)
        supervised_class_no_list.append(min_index)
        print('Supervised head_no list:',supervised_class_no_list)
        model.to(device)

        if model_no_heads==len(supervised_class_no_list):
            print('All classes supervised')
            break
        else:  
            if outer_epoch_schedule[1:][i]>0:
                print('Thinning data set. Removing {} elements'.format(filter_ratio*len(semi_supervised_list)))
                semi_supervised_list=filter_by_proj(semi_supervised_list,min_index,filter_ratio)
            print('Data set size:',len(semi_supervised_list))
            output_path=semi_supervised_multi_head_train_v3(model,semi_supervised_list,outer_epochs=outer_epoch_schedule[1:][i],inner_epochs_pair=inner_epochs_pair,
                                    QP_reg_schedule=QP_reg_schedule,batch_size=batch_size,supervised_class_nos=supervised_class_no_list,save_increment=100000,atom_reg=0,dtype=dtype,filename=filename,squash_factor=squash_factor,lr=lr)

            paths.append(output_path)
    
    output_dir=f'{filename}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'paths.npy'), paths, allow_pickle=True)
    np.save(os.path.join(output_dir, 'basis_list.npy'), basis_list, allow_pickle=True)
    np.save(os.path.join(output_dir, 'supervised_classes.npy'), supervised_class_no_list, allow_pickle=True)
    return model,semi_supervised_list,basis_list,paths
'''

def semi_supervised_pipeline_v4(model,embedded_data_list,
                                outer_epoch_schedule,inner_epochs_pair,
                                filter_ratio,
                                squash_factor,
                                batch_size,QP_reg_schedule,
                                dtype,filename,lr,basis_update):
    #model
    #embedded_data_list: list of embedded_data objects
    #outer_epoch_schedule: list of integers (should be length=number of heads)
    #inner_epochs pair: list of two integers
    #dtype: string
    #filename:string
    #basis_update: 'in_data' or 'out_data'
    basis_list=[]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    supervised_class_no_list=[]
    semi_supervised_list=[]
    model_no_heads=model.no_heads
    for i in embedded_data_list:
        semi_supervised_list.append(semi_supervised_embedded_data(i,{}))
    paths=[]
    os.makedirs(filename, exist_ok=True)

    if len(outer_epoch_schedule)>model_no_heads:
        raise RuntimeError('Error: outer_epoch_schedule has more than no_heads elements')
    
    if outer_epoch_schedule[0]>0:
        output_path=semi_supervised_multi_head_train_v3(model,semi_supervised_list,outer_epochs=outer_epoch_schedule[0],inner_epochs_pair=inner_epochs_pair,
                                         QP_reg_schedule=QP_reg_schedule,batch_size=batch_size,supervised_class_nos=supervised_class_no_list,save_increment=100000,atom_reg=0,dtype=dtype,filename=filename,squash_factor=squash_factor,lr=lr)

        paths.append(output_path)
    else:
        time_now = time.time()
        trial_dir = '{}/Trial_{:.4f}'.format(filename, time_now)
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{trial_dir}/multi_head_model_0.pt')
        paths.append('{}/multi_head_model_{}.pt'.format(trial_dir,0))

    for i in range(len(outer_epoch_schedule[1:])+1):
        torch.cuda.empty_cache()
        gc.collect()
        model.to('cpu')
        if basis_update=='in_data':
            method='OT'
            new_basis,score_dict=learn_generators(embedded_data_list,model,'Nearest_Neighbor',supervised_class_no_list,method=method)
            basis_list.append(new_basis)
            filtered_score_dict = {k: v for k, v in score_dict.items() if k not in supervised_class_no_list}
            min_index = min(filtered_score_dict, key=filtered_score_dict.get)
            print('Supervising head_no:',min_index)
            semi_supervised_list=update_semi_supervised_embedding(semi_supervised_list,new_basis[min_index].mapping,min_index)
            supervised_class_no_list.append(min_index)
            print('Supervised head_no list:',supervised_class_no_list)
        if basis_update=='out_data':
            method='OT'
            score_dict,min_index=find_best_model_idx(model,model_no_heads,embedded_data_list,supervised_class_no_list,method)
            print('Supervising head_no:',min_index)
            base_supp=torch.tensor(embedded_data_list[0].base.points,dtype=torch.float32)
            new_gen=model(base_supp)[:,min_index,:]
            semi_supervised_list=update_semi_supervised_embedding(semi_supervised_list,new_gen,min_index)
            supervised_class_no_list.append(min_index)
            print('Supervised head_no list:',supervised_class_no_list)
            
        model.to(device)

        if model_no_heads==len(supervised_class_no_list):
            print('All classes supervised')
            break
        else:  
            if outer_epoch_schedule[1:][i]>0:
                print('Thinning data set. Removing {} elements'.format(filter_ratio*len(semi_supervised_list)))
                semi_supervised_list=filter_by_proj(semi_supervised_list,min_index,filter_ratio)
            print('Data set size:',len(semi_supervised_list))
            output_path=semi_supervised_multi_head_train_v3(model,semi_supervised_list,outer_epochs=outer_epoch_schedule[1:][i],inner_epochs_pair=inner_epochs_pair,
                                    QP_reg_schedule=QP_reg_schedule,batch_size=batch_size,supervised_class_nos=supervised_class_no_list,save_increment=100000,atom_reg=0,dtype=dtype,filename=filename,squash_factor=squash_factor,lr=lr)

            paths.append(output_path)
    
    output_dir=f'{filename}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'paths.npy'), paths, allow_pickle=True)
    np.save(os.path.join(output_dir, 'basis_list.npy'), basis_list, allow_pickle=True)
    np.save(os.path.join(output_dir, 'supervised_classes.npy'), supervised_class_no_list, allow_pickle=True)
    return model,semi_supervised_list,basis_list,paths
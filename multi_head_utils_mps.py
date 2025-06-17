import os
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cvxpy as cp
import tensorflow as tf
import scipy as sp
from OT_utils import *
from joblib import Parallel, delayed
from models.multi_head_dir.multi_head_model import *
from models.multi_head_dir.multi_head_training import *
from visualizations import *
from generate_data import *
import torch.optim as optim
from torch.amp import autocast
from scipy.optimize import minimize
from claude_solution import *

#testing stepsize schedulers
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau



class CustomDataset(Dataset):
    # dtype is a string descriptor for the dataset, e.g., 'MNIST'
    def __init__(self, mapping_data, base_point_tensor,dtype):
        self.mapping_data = mapping_data
        self.dtype = dtype
        self.base_point_tensor=base_point_tensor

    def __len__(self):
        return len(self.mapping_data)

    def base_supp_size(self):
        return len(self.mapping_data[0])

    def __getitem__(self, idx):
        return self.mapping_data[idx]  # Only return the data (no labels)

    def permute(self, seed=None):
        """Shuffle the dataset order in-place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.mapping_data)


class PairedTensors(Dataset):
    # dtype is a string descriptor for the dataset, e.g., 'MNIST'
    def __init__(self, base_data,mapping_data, dtype):
        self.base_data = base_data
        self.mapping_data=mapping_data
        self.dtype = dtype

    def __len__(self):
        return len(self.base_data)

    def base_supp_size(self):
        return len(self.base_data[0])

    def __getitem__(self, idx):
        return self.base_data[idx],self.mapping_data[idx]  # Only return the data (no labels)
    
class MappingCoeffTensors(Dataset):
    # dtype is a string descriptor for the dataset, e.g., 'MNIST'
    def __init__(self, mapping_data, base_point_tensor,coefficients_data,dtype):
        self.mapping_data = mapping_data
        self.dtype = dtype
        self.coefficients_data=coefficients_data
        self.base_point_tensor=base_point_tensor

    def __len__(self):
        return len(self.mapping_data)

    def base_supp_size(self):
        return len(self.mapping_data[0])

    def __getitem__(self, idx):
        return self.mapping_data[idx], self.coefficients_data[idx], idx


    #def permute(self, seed=None):
    #    """Shuffle the dataset order in-place."""
    #    if seed is not None:
    #        random.seed(seed)
    #    random.shuffle(self.data)


def adaptive_LASSO_loss(lambda_batch, weights, batch_data, model_output, reg):
    # lambda_batch: (batch_size, no_heads)
    # weights: (batch_size, no_heads)
    # batch_data: (batch_size, supp_size, dim)
    # model_data: (supp_size, no_heads, dim)

    # Compute linear combination of model_data weighted by lambda_batch
    cvx_combo = torch.einsum('bh, shd -> bsd', lambda_batch, model_output)
    residual = cvx_combo - batch_data

    # L2 loss + weighted L1 penalty on lambda_batch
    return 0.5 * residual.pow(2).sum() + reg * (lambda_batch * weights).sum()

def project_to_simplex_batch(v):
    """ Projects each row of v onto the probability simplex. """
    # v shape: (batch_size, no_heads)
    # We do this row-wise for the batch
    sorted_v, _ = torch.sort(v, descending=True, dim=1)
    cssv = torch.cumsum(sorted_v, dim=1) - 1
    ind = torch.arange(1, v.shape[1] + 1, device=v.device).view(1, -1)
    cond = sorted_v - cssv / ind > 0
    rho = cond.sum(dim=1) - 1  # last index where condition is true, per row

    theta = cssv[torch.arange(v.shape[0]), rho] / (rho + 1).float()
    # Reshape theta for broadcasting
    theta = theta.view(-1, 1)
    return torch.clamp(v - theta, min=0)





def optimization_step_mps(displacement_tensor, output_tensor,coefficient_tensor, optimizer, lambda_reg,variance_threshold, device):
    # displacement_tensor: (batch_size, supp_size, no_heads, dim)
    # coefficient_tensor: (batch_size, no_heads)
    outer_product_tensor = displacement_tensor @ displacement_tensor.transpose(-1, -2)  # (batch_size, supp_size, no_heads, no_heads)
    optimizer.zero_grad()
    result = torch.einsum('ik,ijkl,il->ij', coefficient_tensor, outer_product_tensor, coefficient_tensor).to(device)  # (batch_size, supp_size)
    loss_over_batches = result.mean(dim=1).to(device)  # (batch_size,)
    
    reg_loss = torch.tensor(0.0, device=device)  # Initialize on correct device
    vari_max=0
    if lambda_reg > 0:
        #mean_tensor: computes the mean vector for each head
        mean_tensor = output_tensor.mean(dim=1)  # (batch_size, no_heads, dim)

        #diff_tensor: computes the difference vector between each head output point and the mean
        diff_tensor = output_tensor - mean_tensor.unsqueeze(1)  # (batch_size, supp_size, no_heads, dim)

        #sq_norm: computes the squared euclidean distance between each head output point and the mean (norm of diff_tensor)
        sq_norm = (diff_tensor ** 2).sum(dim=-1)  # (batch_size, supp_size, no_heads)

        #variance_per_head: averages over the supp_size to compute the variance per head
        variance_per_head = sq_norm.mean(dim=1)  # (batch_size, no_heads)
        vari_max=variance_per_head.max()
        # Create mask for heads that exceed variance threshold (i.e. vector of 1's and 0's indicating where the threshold is exceeded)
        offending_mask = variance_per_head > variance_threshold  # (batch_size, no_heads)
        
        #The following snippet detaches the variance computation from the graph for heads that do NOT exceed the variance threshold
        #This way only heads that exceed the variance threshold see the loss from this regularization
        masked_variance = torch.where(
            offending_mask,
            variance_per_head,
            variance_per_head.detach()) 
        reg_loss = lambda_reg * masked_variance.mean()
    # Total loss: main loss + regularization
    loss = loss_over_batches.mean() + reg_loss
    loss.backward()
    optimizer.step()
    return loss,vari_max

def optimization_step_nosimplex_mps(mapping_batch, output_tensor,coefficient_tensor, optimizer,scheduler, lambda_reg,variance_threshold, device):
    # output_tensor: (supp_size,no_heads,dim)
    # mapping batch: (batch_size,support_size,dim)    
    # coefficient_tensor: (batch_size, no_heads)
    optimizer.zero_grad()

    linear_combination=torch.einsum('kjl,ij->ikl',output_tensor,coefficient_tensor).to(device)
    #linear_combination: (batch_size,supp_size,dim)
    r_tensor=linear_combination-mapping_batch
    #computes square euclidean distance between points
    r_tensor=(r_tensor**2).sum(dim=-1) #(batch_size,supp_size)
    #compute L2 error over each element in the batch
    loss_over_batches=r_tensor.mean(dim=-1) #(batch_size)

        # Compute mean of each head over support points
    mean_tensor = output_tensor.mean(dim=0)  # (no_heads, dim)

    # Compute mean of mapping_batch across batch and support size
    data_center = mapping_batch.mean(dim=0).mean(dim=0)  # (dim,)

    # Compute L2 distance between each head mean and data center
    center_dist = ((mean_tensor - data_center) ** 2).sum(dim=-1)  # (no_heads,)
    # Centering loss encourages each head to center near the data
    center_loss = center_dist.mean()
    
    reg_loss = torch.tensor(0.0, device=device)  # Initialize on correct device
    vari_max=0
    if lambda_reg > 0:
        #mean_tensor: computes the mean vector for each head
        #diff_tensor: computes the difference vector between each head output *point* and the mean
        diff_tensor = output_tensor - mean_tensor.unsqueeze(0)  # (supp_size,no_heads,dim)
        #sq_norm: computes the squared euclidean distance between each head output *point* and the mean (norm of diff_tensor)
        sq_norm = (diff_tensor ** 2).sum(dim=-1)  # (supp_size, no_heads)
        #variance_per_head: averages over the supp_size to compute the variance per head
        variance_per_head = sq_norm.mean(dim=0)  # (no_heads)
        vari_max=variance_per_head.max()
        # Create mask for heads that exceed variance threshold (i.e. vector of 1's and 0's indicating where the threshold is exceeded)
        offending_mask = variance_per_head > variance_threshold  # (no_heads)
        
        #The following snippet detaches the variance computation from the graph for heads that do NOT exceed the variance threshold
        #This way only heads that exceed the variance threshold see the loss from this regularization
        masked_variance = torch.where(
            offending_mask,
            variance_per_head,
            variance_per_head.detach()) 
        reg_loss = lambda_reg * masked_variance.mean()
    # Total loss: main loss + regularization
    loss = loss_over_batches.mean() + reg_loss + center_loss
    loss.backward()
    optimizer.step()
    if scheduler!=None:
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
    return loss,vari_max





def multi_head_train_mps(model, dataloader, outer_epoch_schedule, inner_epochs_pair,
                                          QP_reg_schedule, batch_size,
                                          save_increment, dtype, filename, lambda_reg,variance_threshold_scaling, lr,base_logic,sparsity_reg,device,warm_start_length):
    start_time=time.time()
    model.train()
    print(model)
    #model: multi head neural net 
    #dataset: list of semi_supervised_embedded_data objects
    #outer_epochs: int
    #inner epochs_pair: (int_1,int_2), 
    # total epochs = outer_epochs*int_1*int_2
    #QP_reg_schedule: reg_schedule object
    #batch_size: int
    #dim: int
    #save_increment: int
    #lr: float    
    no_batches=len(dataloader)
    no_heads=model.no_heads

    #'TRYING REINITIALIZING EVERY STEP'
    time_now=time.time()
    os.makedirs(f'{filename}/Trial_{time_now:.4f}', exist_ok=True)
    counter=0
    inner_loss=0

    #Computing typical variance for variance regularization
    if base_logic=='VaryBase':
        example_mapping_data=dataloader.dataset[0:50][1] # (batch_size, supp_size, dim)
    else:
        example_mapping_data=dataloader.dataset.mapping_data[0:50] # (batch_size, supp_size, dim)

    mean_tensor = example_mapping_data.mean(dim=1)  # (batch_size, dim)
    #diff_tensor: computes the difference vector between each head output point and the mean
    disp_tensor = example_mapping_data - mean_tensor.unsqueeze(1)  # (batch_size, supp_size, dim)
    base_supp_size=example_mapping_data.shape[1]
    #sq_norm: computes the squared euclidean distance between each head output point and the mean (norm of diff_tensor)
    sq_norm = (disp_tensor ** 2).sum(dim=-1)  # (batch_size, supp_size, no_heads)
    print('Sample Variance',sq_norm)
    #variance_per_head: averages over the supp_size to compute the variance per head
    sample_variance = sq_norm.mean(dim=1)  # (batch_size, no_heads)
    sample_variance=sample_variance.mean()
    for outer_epoch_index, outer_epoch_batch in enumerate(outer_epoch_schedule):
        if outer_epoch_index>0 and outer_epoch_index<len(outer_epoch_schedule)-1 and outer_epoch_batch>0:
            save_path = f'{filename}/Trial_{time_now:.4f}/multi_head_model_{counter}_drop_{outer_epoch_index}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at counter {counter}")
        #TRYING TO KEEP THE OPTIMIZER FIXED

        if outer_epoch_index==0:
            optimizer = optim.AdamW(model.parameters(), lr=3*lr,weight_decay=1e-4)

            #scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #        optimizer,
            #        max_lr=3*lr,
            ##        total_steps=inner_epochs_pair[0]*inner_epochs_pair[1]*len(dataloader)*outer_epoch_batch,
             #       pct_start=0.3,
             #       anneal_strategy='cos')
            scheduler=None

            inner_loss,counter=alternating_minimization(model, dataloader,optimizer,scheduler, outer_epoch_batch,outer_epoch_index,
                                            inner_epochs_pair,
                                            QP_reg_schedule,
                                            save_increment, filename, 
                                            lambda_reg,variance_threshold_scaling, sample_variance,
                                            base_logic,sparsity_reg,device,time_now,counter,warm_start_length)
        else:
        #else:
            if outer_epoch_batch>0:
                total_epochs=inner_epochs_pair[0]*inner_epochs_pair[1]*len(dataloader)*outer_epoch_batch
                optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=total_epochs)

               # scheduler = torch.optim. slr_scheduler.OneCycleLR(
               #         optimizer,
               #         max_lr=3*lr,
               #         total_steps=total_epochs,
               #         pct_start=0.3,
               #         anneal_strategy='cos')
            
                inner_loss,counter=alternating_minimization(model, dataloader,optimizer,scheduler, outer_epoch_batch,outer_epoch_index,
                                                        inner_epochs_pair,
                                                        QP_reg_schedule,
                                                        save_increment, filename, 
                                                        lambda_reg,variance_threshold_scaling, sample_variance,
                                                        base_logic,sparsity_reg,device,time_now,counter,0)
    end_time=time.time()
    save_trial('training_list.csv',
               dtype=dtype,
               index=time_now,
               end_loss=inner_loss,
               train_time=end_time-start_time,
               architecture=model.__class__.__name__,
               no_heads=model.no_heads,
               base_supp_size=base_supp_size,
               batch_size=batch_size,
               outer_epochs=outer_epoch_schedule,
               inner_epochs_pair=inner_epochs_pair,
               counter=counter,
               QP_base_scale_min=(QP_reg_schedule.base_value,QP_reg_schedule.scale_factor,QP_reg_schedule.min_i),
                learning_rate =lr, no_data_points=len(dataloader.dataset))

    torch.save(model.state_dict(), '{}/Trial_{:.4f}/multi_head_model_{}.pt'.format(filename,time_now,counter))
    return f'{filename}/Trial_{time_now:.4f}',inner_loss


def alternating_minimization(model, dataloader,optimizer,scheduler, outer_epoch_batch,outer_epoch_idx, inner_epochs_pair,
                                          QP_reg_schedule,
                                          save_increment, filename, 
                                          lambda_reg,variance_threshold_scaling, sample_variance,
                                          base_logic,sparsity_reg,device,time_now,counter,warm_start_length):
    inner_loss=0
    simplex_constraint=False
    no_batches=len(dataloader)
    for outer_epoch in range(outer_epoch_batch):
        if outer_epoch_idx==0:
            if outer_epoch<=warm_start_length-1:
                print('Warm start round')
        
        for batch_idx,batch_data in enumerate(dataloader):
                
                
                start=time.time()
                if base_logic=='VaryBase':
                    base_point_batch=batch_data[0].to(device)
                    mapping_batch=batch_data[1].to(device)
                else:
                    base_point_batch=dataloader.dataset.base_point_tensor.squeeze(0).to(device)
                    mapping_batch=batch_data[0].to(device)
                    coefficients_batch=batch_data[1].to(device)
                    indices=batch_data[2]
                
                #batch_data is a (base_data,mapping_data) tuple 
                #mapping_batch has shape (batch_size,support_size,dim)
                for i in range(inner_epochs_pair[0]):
                    inner_loss=0
                    QP_reg=QP_reg_schedule(outer_epoch)

                    #coefficients_tensor=solve_for_sparse_coefficients(model,base_point_batch,batch_data,device,1e-9,0.1)
                    #if outer_epoch>1000000:
                        #FIX TO INCLUDE INITIALIZATION?
                    #    coefficients_batch=solve_for_coeffs_cvxpy_v2(model,base_point_batch,mapping_batch,QP_reg,device,base_logic)
                    if outer_epoch_idx==0:
                        if outer_epoch<=warm_start_length:
                            coefficients_batch = torch.ones_like(coefficients_batch).to(device)
                            coefficients_batch = coefficients_batch / coefficients_batch.sum(dim=1, keepdim=True)
                        else:
                            model.eval()
                            coefficients_batch=solve_for_sparse_coefficients(
                                                                    model,
                                                                    base_point_batch,
                                                                    mapping_batch,
                                                                    coefficients_batch,
                                                                    device,
                                                                    simplex_constraint,
                                                                    sparse_reg=sparsity_reg,
                                                                    precompute_gram=True,
                                                                    tolerance=1e-9)
                            model.train()
                    else:
                        model.eval()
                        coefficients_batch=solve_for_sparse_coefficients_k_sparse(model,
                                                                base_point_batch,
                                                                mapping_batch,
                                                                coefficients_batch,
                                                                device,
                                                                simplex_constraint,
                                                                sparsity_reg,
                                                                precompute_gram=True,
                                                                k_sparse=outer_epoch_idx,
                                                                tolerance=1e-7)
                        model.train()

                    if counter%(save_increment/10)==0:
                        print('Nonzero over total: {}/{}'.format(torch.count_nonzero(coefficients_batch),(coefficients_batch.shape[0]*coefficients_batch.shape[1])))
                    
                    #coefficients_tensor=solve_for_coeffs_cvxpy_v2(model,base_point_batch,mapping_batch,QP_reg,device,base_logic)
                    coefficients_batch=coefficients_batch.to(device)
                    if counter%save_increment==0:
                        print(coefficients_batch)
                    for j in range(inner_epochs_pair[1]):
                        output = model(base_point_batch)
                        #output=model(base_point_batch).transpose(1, 2)

                        # Time displacement tensor computation
                        if base_logic=='VaryBase':
                            displacement_tensor = (output- mapping_batch.unsqueeze(1).to(device))
                            displacement_tensor=displacement_tensor.transpose(1,2)
                        else:
                            displacement_tensor = (output.unsqueeze(0) - mapping_batch.unsqueeze(2).to(device))
                        
                        #displacement_tensor=displacement_tensor.transpose(1,2)
                        # Time optimization step
                        if simplex_constraint==False:
                            loss,vari_max=optimization_step_nosimplex_mps(
                                            mapping_batch,
                                            output,
                                            coefficients_batch,
                                            optimizer,
                                            scheduler,
                                            lambda_reg,
                                            variance_threshold_scaling*sample_variance,
                                            device)

                        else:
                            loss, vari_max = optimization_step_mps(
                                displacement_tensor,
                                output,
                                coefficients_batch,
                                optimizer,
                                lambda_reg,
                                variance_threshold_scaling*sample_variance,
                                device)
                        # Accumulate loss
                        if counter%save_increment==0:
                            print('Vari_max',vari_max)
                        inner_loss += loss.item()

                        if counter % save_increment == 0 and counter>0:
                            save_path = f'{filename}/Trial_{time_now:.4f}/multi_head_model_{counter}_drop_{outer_epoch_idx}.pt'
                            torch.save(model.state_dict(), save_path)
                            print(f"Model saved at counter {counter}")
                        counter += 1
                inner_epoch_count=inner_epochs_pair[0]*inner_epochs_pair[1]
                dataloader.dataset.coefficients_data[indices] = coefficients_batch.detach().cpu()
                print(f'Outer Idx {outer_epoch_idx},Epoch [{outer_epoch+1}/{outer_epoch_batch}], '
                        f'Batch [{batch_idx+1}/{no_batches}], '
                        f'Counter {counter}, Avg Inner Loss: {inner_loss/inner_epoch_count:.8f}')
                print(f"Batch Time: {time.time() - start:.2f}s")
    return inner_loss,counter




'''
def multi_head_train_cuda_OG(model, dataloader, outer_epoch_schedule, inner_epochs_pair,
                                          QP_reg_schedule, batch_size,
                                          save_increment, dtype, filename, lambda_reg,variance_threshold_scaling, lr,base_logic,sparsity_reg,device):
    start_time=time.time()
    model.train()
    print(model)
    #model: multi head neural net 
    #dataset: list of semi_supervised_embedded_data objects
    #outer_epochs: int
    #inner epochs_pair: (int_1,int_2), 
    # total epochs = outer_epochs*int_1*int_2
    #QP_reg_schedule: reg_schedule object
    #batch_size: int
    #dim: int
    #save_increment: int
    #lr: float    
    no_batches=len(dataloader)
    no_heads=model.no_heads
    optimizer = optim.Adam(model.parameters(), lr=lr)
    time_now=time.time()
    os.makedirs(f'{filename}/Trial_{time_now:.4f}', exist_ok=True)
    counter=0
    inner_loss=0

    #Computing typical variance for variance regularization
    if base_logic=='VaryBase':
        example_mapping_data=dataloader.dataset[0:50][1] # (batch_size, supp_size, dim)
    else:
        example_mapping_data=dataloader.dataset.mapping_data[0:50] # (batch_size, supp_size, dim)

    mean_tensor = example_mapping_data.mean(dim=1)  # (batch_size, dim)
    #diff_tensor: computes the difference vector between each head output point and the mean
    disp_tensor = example_mapping_data - mean_tensor.unsqueeze(1)  # (batch_size, supp_size, dim)
    base_supp_size=example_mapping_data.shape[1]

    #sq_norm: computes the squared euclidean distance between each head output point and the mean (norm of diff_tensor)
    sq_norm = (disp_tensor ** 2).sum(dim=-1)  # (batch_size, supp_size, no_heads)
    print('Sample Variance',sq_norm)
    #variance_per_head: averages over the supp_size to compute the variance per head
    sample_variance = sq_norm.mean(dim=1)  # (batch_size, no_heads)
    sample_variance=sample_variance.mean()
    for outer_epoch_index, outer_epoch_batch in enumerate(outer_epoch_schedule):
        if outer_epoch_index>0 and outer_epoch_index<len(outer_epoch_schedule)-1:
            save_path = f'{filename}/Trial_{time_now:.4f}/multi_head_model_{counter}_drop_{outer_epoch_index}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at counter {counter}")

        inner_loss,counter=alternating_minimization(model, dataloader,optimizer, outer_epoch_batch,outer_epoch_index,
                                                    inner_epochs_pair,
                                                    QP_reg_schedule,
                                                    save_increment, filename, 
                                                    lambda_reg,variance_threshold_scaling, sample_variance,
                                                    base_logic,sparsity_reg,device,time_now,counter)
        
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
               outer_epochs=outer_epoch_schedule,
               inner_epochs_pair=inner_epochs_pair,
               counter=counter,
               QP_base_scale_min=(QP_reg_schedule.base_value,QP_reg_schedule.scale_factor,QP_reg_schedule.min_i),
                learning_rate =lr, no_data_points=len(dataloader.dataset))

    torch.save(model.state_dict(), '{}/Trial_{:.4f}/multi_head_model_{}.pt'.format(filename,time_now,counter))
    return f'{filename}/Trial_{time_now:.4f}',inner_loss



def multi_head_train_cuda(model, dataloader, outer_epoch_schedule, inner_epochs_pair,
                                          QP_reg_schedule, batch_size,
                                          save_increment, dtype, filename, lambda_reg,variance_threshold_scaling, lr,base_logic,sparsity_reg,device):
    start_time=time.time()
    model.train()
    print(model)
    #model: multi head neural net 
    #dataset: list of semi_supervised_embedded_data objects
    #outer_epochs: int
    #inner epochs_pair: (int_1,int_2), 
    # total epochs = outer_epochs*int_1*int_2
    #QP_reg_schedule: reg_schedule object
    #batch_size: int
    #dim: int
    #save_increment: int
    #lr: float    
    no_batches=len(dataloader)
    no_heads=model.no_heads
    optimizer = optim.Adam(model.parameters(), lr=lr)
    time_now=time.time()
    os.makedirs(f'{filename}/Trial_{time_now:.4f}', exist_ok=True)
    counter=0
    inner_loss=0

    #Computing typical variance for variance regularization
    if base_logic=='VaryBase':
        example_mapping_data=dataloader.dataset[0:50][1] # (batch_size, supp_size, dim)
    else:
        example_mapping_data=dataloader.dataset.mapping_data[0:50] # (batch_size, supp_size, dim)

    mean_tensor = example_mapping_data.mean(dim=1)  # (batch_size, dim)
    #diff_tensor: computes the difference vector between each head output point and the mean
    disp_tensor = example_mapping_data - mean_tensor.unsqueeze(1)  # (batch_size, supp_size, dim)
    base_supp_size=example_mapping_data.shape[1]

    #sq_norm: computes the squared euclidean distance between each head output point and the mean (norm of diff_tensor)
    sq_norm = (disp_tensor ** 2).sum(dim=-1)  # (batch_size, supp_size, no_heads)
    print('Sample Variance',sq_norm)
    #variance_per_head: averages over the supp_size to compute the variance per head
    sample_variance = sq_norm.mean(dim=1)  # (batch_size, no_heads)
    sample_variance=sample_variance.mean()
    for outer_epoch_index, outer_epoch_batch in enumerate(outer_epoch_schedule):
        #if outer_epoch_index > 0: 
        #    test_coefficients_tensor = []
        #    print('Sparsifying Coefficients, round {}'.format(outer_epoch_index))
        #    model = model.detach().cpu()
        #    model.eval()
        #    base_points=dataloader.dataset.base_point_tensor.squeeze(0)

#            for i in range(len(dataloader.dataset)):
#                if i % 500 == 0:
#                    print('{}/{}'.format(i, len(dataloader.dataset)))

#                mapping_data=dataloader.dataset[i][0]
 #               coefficients_batch=dataloader.dataset[i][1]
 #               torch.tensor(multi_solve_for_coefficients(model,base_points,dataloader.dataset[i])[0])


#                coefficients_tensor.append(torch.tensor(solve_for_coefficients(model,base_point_tensor,data[i])[0]))
#            coefficients_tensor=torch.stack(coefficients_tensor)

        inner_loss,counter=alternating_minimization(model, dataloader,optimizer, outer_epoch_batch, inner_epochs_pair,
                                          QP_reg_schedule,
                                          save_increment, filename, 
                                          lambda_reg,variance_threshold_scaling, sample_variance,
                                          base_logic,sparsity_reg,device,time_now,counter)
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
               outer_epochs=outer_epoch_schedule,
               inner_epochs_pair=inner_epochs_pair,
               counter=counter,
               QP_base_scale_min=(QP_reg_schedule.base_value,QP_reg_schedule.scale_factor,QP_reg_schedule.min_i),
                learning_rate =lr, no_data_points=len(dataloader.dataset))

    torch.save(model.state_dict(), '{}/Trial_{:.4f}/multi_head_model_{}.pt'.format(filename,time_now,counter))
    return f'{filename}/Trial_{time_now:.4f}',inner_loss
'''

'''
def solve_for_coeffs_cvxpy_v2(model,base_point_tensor,batch_data,QP_reg,device,base_logic):
    #if vary_base==True:
    #           base_point_tensor: (batch_size,supp_size,dim)
    #else:
    #           base_point_tensor: (1,supp_size,dim)
    #batch_data: (batch_size,supp_size,dim)
    with torch.no_grad():
        output=model(base_point_tensor).detach().cpu()
    no_heads=model.no_heads
    batch_size = batch_data.shape[0]
    displacement_tensor=[]
    for i in range(no_heads):
            if base_logic=='VaryBase':
                tensor_map=output[:,i,:,:]-batch_data
            else:
                #To deal with batchnorm bullshit, doing this:
                tensor_map=output[:,i,:].unsqueeze(0)-batch_data
                #tensor_map=output[:,i,:,:]-batch_data
            displacement_tensor.append(tensor_map)
    displacement_tensor=torch.stack(displacement_tensor,dim=1) #displacement_tensor: (batch_size,no_heads,supp_size,dim)
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
        v_solutions.append(v.value)
    # Convert back to torch tensor on the correct device
    v_tensor = torch.tensor(np.array(v_solutions), dtype=torch.float32).to(device)  # (batch_size, no_heads)
    return v_tensor
'''
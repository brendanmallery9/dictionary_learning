
import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from OT_utils import *
from joblib import Parallel, delayed
from models.multi_head_dir.multi_head_model import *
from models.multi_head_dir.multi_head_training import *
from visualizations import *
import gc
from generate_data import *
from self_supervised_training import *
import math
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import qpth

class CustomDataset(Dataset):
    # dtype is a string descriptor for the dataset, e.g., 'MNIST'
    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def base_supp_size(self):
        return len(self.data[0].base.points)

    def __getitem__(self, idx):
        return self.data[idx]  # Only return the data (no labels)

    def permute(self, seed=None):
        """Shuffle the dataset order in-place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

###Distributed Data Parallel (DPP) Processing Functions

# DDP allows you to distribute training across multiple GPUs by automatically splitting up the data set and synching up model updates
# spawn_train_DDP creates a process for each GPU
# DDP_wrapper is run on each process, which takes a portion of the data and creates a copy of the original model to train on the data
# Each copy of the model can be initialized at model_state_dict, and has parameters model_init_args.
# GPUs are indexed by rank, the total number of GPUS=world_size, and each one is handled by an instance of DDP_wrapper
# semi_supervised_multi_head_train_cuda is then run inside each DDP_wrapper
# The magic of DPP is that each time loss.backwards() is called on each of the GPUs, the gradients across all models are averaged, and then each process applies optimizer.step() with the averaged gradient
# So there is no independence between the different copies of the model on each of the GPUs

# version 1 assumes that the input data into the model is a fixed base point tensor
# side note: At a high level I like that the mdoel takes in the base point tensor even though this is fixed across training
# that way, you can do things like query the model on an individual data point
def cuda_solve_for_coefficients_qpth_v1(model,base_point_tensor,batch_data,QP_reg,device):
    #base_point_tensor: (supp_size,dim)
    #batch_mapping: (batch_size,supp_size,dim)
    #batch_basis: (batch_size,supp_size,no_heads,dim)
    with torch.no_grad():
        output=model(base_point_tensor) #output: (no_heads,supp_size,dim)
    batch_size=output.shape[0]
    no_heads=output.shape[1]
    supp_size=base_point_tensor.shape[2]
    displacement_tensor=[]
    for i in range(no_heads):
            tensor_map=output[i,:,:].unsqueeze(0)-batch_data
            displacement_tensor.append(tensor_map)
    displacement_tensor=torch.stack(displacement_tensor,dim=1) #displacement_tensor: (batch_size,no_heads,supp_size,dim)
    eye = torch.eye(no_heads, device=device).expand(batch_size, no_heads, no_heads)
    QP = build_QP_v2(displacement_tensor) + QP_reg * eye
    QP=QP.to(device)
    lin_term = torch.zeros(batch_size, no_heads, device=device) #QP linear term = 0
    #nonnegativity constraints
    nonneg = -torch.eye(no_heads, device=device).unsqueeze(0).expand(batch_size, no_heads, no_heads)
    zero_vec = torch.zeros(batch_size, no_heads, device=device)
    #normalization constraints
    norm = torch.ones(batch_size, 1, no_heads, device=device)
    ones_vec = torch.ones(batch_size, 1, device=device)
    solver = qpth.qp.QPFunction()

    #calls solver(QP,p,nonneg,zero_vec,norm,ones_vec)
    #solves min_v v^T QP v with v\geq 0 and \sum v=1

    v = solver(QP, lin_term, nonneg, zero_vec, norm, ones_vec)
    v = v.clamp(min=1e-7)
    return v


def optimization_step_cuda(displacement_tensor, coefficient_tensor, optimizer, lambda_reg,variance_threshold, device):
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
        mean_tensor = displacement_tensor.mean(dim=1)  # (batch_size, no_heads, dim)

        #diff_tensor: computes the difference vector between each head output point and the mean
        diff_tensor = displacement_tensor - mean_tensor.unsqueeze(1)  # (batch_size, supp_size, no_heads, dim)

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




def multi_head_train_cuda(model, dataloader,base_point_tensor, outer_epochs, inner_epochs_pair,
                                          QP_reg_schedule, batch_size,
                                          save_increment, dtype, filename, lambda_reg,variance_threshold, lr,device):
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
    #lr: float    
    no_batches=len(dataloader)
    no_heads=model.no_heads
    optimizer = optim.Adam(model.parameters(), lr=lr)
    time_now=time.time()
    if torch.distributed.get_rank() == 0:
        os.makedirs(f'{filename}/Trial_{time_now:.4f}', exist_ok=True)
    counter=0
    inner_loss=0
    base_point_tensor.to(device)
    base_supp_size=base_point_tensor.shape[0]
    for outer_epoch in range(outer_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
        #this sets the random seed for the sampler (which results in coordination across all processes)
            dataloader.sampler.set_epoch(outer_epoch)
        for batch_idx,batch_data in enumerate(dataloader):
            start=time.time()
            #mapping_tensor has shape (batch_size,support_size,dim)
            for i in range(inner_epochs_pair[0]):
                inner_loss=0
                QP_reg=QP_reg_schedule(outer_epoch)
                coefficients_tensor=cuda_solve_for_coefficients_qpth_v1(model,base_point_tensor,batch_data,QP_reg,device)
                for j in range(inner_epochs_pair[1]):
                    output = model(base_point_tensor) #output: (supp_size,no_heads,dim)
                    displacement_tensor = (output.unsqueeze(0) - batch_data.unsqueeze(2)).to(device) #displacement_tensor: (batch_size,supp_size,model_head_no,dim)
                    loss,vari_max=optimization_step_cuda(displacement_tensor,coefficients_tensor,optimizer,lambda_reg,variance_threshold,device)
                    inner_loss+=loss.item()
                    #the get_rank() step ensures you're only saving the model once per save 
                    if counter % save_increment == 0 and torch.distributed.get_rank() == 0:
                        save_path = f'{filename}/Trial_{time_now:.4f}/multi_head_model_{counter}.pt'
                        torch.save(model.module.state_dict(), save_path)
                        print(f"Model saved at counter {counter}")
                    counter += 1
            inner_epoch_count=inner_epochs_pair[0]*inner_epochs_pair[1]
            if torch.distributed.get_rank() == 0:
                print(f'Epoch [{outer_epoch+1}/{outer_epochs}], '
                      f'Batch [{batch_idx+1}/{no_batches}], '
                      f'Counter {counter}, Avg Inner Loss: {inner_loss/inner_epoch_count:.8f}')
                print(f"Batch Time: {time.time() - start:.2f}s")
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
                learning_rate =lr, no_data_points=len(dataloader.dataset)
)
    if torch.distributed.get_rank() == 0:
        torch.save(model.module.state_dict(), '{}/Trial_{:.4f}/final_multi_head_model_{}.pt'.format(filename,time_now,counter))


def multiprocess_setup(rank, world_size):
    #hardcoding the ports used to coordinate the procsses
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return device

def initialize_model(device, model_class,model_init_args,model_state_dict,rank):
    model = model_class(*model_init_args).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model = DDP(model, device_ids=[rank])
    return model

def DDP_wrapper(rank,
                world_size,
                model_class,
                model_state_dict,
                model_init_args,
                base_point_tensor,
                dataset,
                batch_size,
                outer_epochs,
                inner_epochs_pair,
                QP_reg_schedule,
                save_increment,
                dtype,
                filename,
                lambda_reg,variance_threshold,
                lr):
    device = multiprocess_setup(rank, world_size)
    model = initialize_model(device, model_class,model_init_args, model_state_dict, rank)

    #Distributed Sampler provides a method of splitting and permuting the data in DataLoader across GPUs
    #By calling dataloader.sampler.set_epoch(i) for some number i, we fix the random seed so that all the processes are coordinated in their splitting and permuting
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x)

    multi_head_train_cuda(model, dataloader,base_point_tensor, outer_epochs, inner_epochs_pair,
                                          QP_reg_schedule, batch_size,
                                          save_increment, dtype, filename, lambda_reg,variance_threshold, lr,device)
    dist.destroy_process_group()


def spawn_train_DDP(model_class, model_state_dict,model_init_args,
                    mapping_tensor,
                    base_point_tensor,
                    world_size,
                    batch_size,
                    outer_epochs,
                    inner_epochs_pair,
                    QP_reg_schedule,
                    save_increment,
                    dtype,
                    filename,
                    lambda_reg,
                    variance_threshold,
                    lr):
    dataset = CustomDataset(mapping_tensor, dtype=dtype)
    #the first argument of mp.spawn is always rank, so it doesn't have to be added here explicitly
    mp.spawn(DDP_wrapper,
             args=(world_size,
                    model_class,
                    model_state_dict,
                    model_init_args,
                    base_point_tensor,
                    dataset,
                    batch_size,
                    outer_epochs,
                    inner_epochs_pair,
                    QP_reg_schedule,
                    save_increment,
                    dtype,
                    filename,
                    lambda_reg,variance_threshold,
                    lr),
             nprocs=world_size,
             join=True)





def run_ddp_training(
    model_class, model_state_dict,model_init_args,
                    mapping_tensor,
                    base_point_tensor,
                    world_size,
                    batch_size,
                    outer_epochs,
                    inner_epochs_pair,
                    QP_reg_schedule,
                    save_increment,
                    dtype,
                    filename,
                    lambda_reg,
                    variance_threshold,
                    lr):
    """
    Launches DDP training with given configuration.
    
    Parameters:
        model_class: the class of the model (e.g. MultiHeadModel)
        model_init_args: tuple of args for model_class
        model_state_dict: pretrained state dict or None
        mapping_tensor: 
        world_size: number of GPUs to use (typically torch.cuda.device_count())
        batch_size: batch size per GPU
        outer_epochs: number of outer epochs
        inner_epochs_pair: tuple (int_1, int_2) for nested training loops
        QP_reg_schedule: a callable taking outer_epoch and returning a float
        save_increment: how often to save the model (in inner steps)
        dtype: torch.float32, torch.float64, etc.
        filename: folder path where models and logs are saved
        lr: learning rate
    """

    assert torch.cuda.is_available(), "CUDA is required for this DDP training."
    assert world_size <= torch.cuda.device_count(), f"world_size={world_size} exceeds available GPUs ({torch.cuda.device_count()})"

    spawn_train_DDP(
        model_class=model_class,
        model_state_dict=model_state_dict,
        model_init_args=model_init_args,
        mapping_tensor=mapping_tensor,
        base_point_tensor=base_point_tensor,
        world_size=world_size,
        batch_size=batch_size,
        outer_epochs=outer_epochs,
        inner_epochs_pair=inner_epochs_pair,
        QP_reg_schedule=QP_reg_schedule,
        save_increment=save_increment,
        dtype=dtype,
        filename=filename,
        lr=lr,
        lambda_reg=lambda_reg,
        variance_threshold=variance_threshold
    )

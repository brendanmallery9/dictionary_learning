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


def save_trial(file_path, **hyperparams):
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=hyperparams.keys())
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(hyperparams)
    
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

def multi_head_train(model,dataset,outer_epochs,inner_epochs_pair,QP_reg_schedule,batch_size,save_increment,atom_reg,lr=0.0001):
    start_time=time.time()
    #model: multi head neural net 
    #dataset: list of embedded_data objects
    #outer_epochs: int
    #inner epochs_pair: (int_1,int_2), 
    # total epochs = outer_epochs*int_1*int_2
    #QP_reg_schedule: reg_schedule object
    #batch_size: int
    #dim: int
    #save_increment: int
    #lr: float
    dtype=dataset.dtype
    base_supp_size=dataset.base_supp_size()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    no_batches=int(np.ceil(len(dataset)/batch_size))
    model.reset_parameters() #initialize the model
    time_now=time.time()
    log_dir = "multi_head_tests/{}/Trial_{:.4f}".format(dtype,time_now)
    os.makedirs(log_dir, exist_ok=True)  
    counter=0
    for outer_epoch in range(outer_epochs):
        dataset.permute()
        for batch in range(no_batches):
            batch_data=dataset[batch_size*batch:min(batch_size*(batch+1),len(dataset))]

            #base_point_tensor has shape (batch_size,support_size,dim)
            #mapping_tensor has shape (batch_size,support_size,dim)

            base_point_tensor=torch.tensor([data.base.points for data in batch_data],dtype=torch.float32,device=device)
            mapping_tensor=torch.tensor([data.mapping for data in batch_data],dtype=torch.float32,device=device)
            supp_size=base_point_tensor.size(1)

            for i in range(inner_epochs_pair[0]):
                inner_loss=0

                #input_reshaped flattens base_point_tensor to shape (batch_size*support_size,dim)
                input_reshaped = base_point_tensor.reshape(-1, base_point_tensor.shape[-1])

                QP_reg=QP_reg_schedule(outer_epoch)

                coefficients_list=[]
                for j in range(len(base_point_tensor)):
                    coefficients,_=multi_solve_for_coefficients(model,base_point_tensor[j],mapping_tensor[j],QP_reg)
                    coefficients_list.append(coefficients)
                #coefficients_tensor: tensor of shape (batch_size,no_heads)

                coefficients_tensor=torch.tensor(coefficients_list,dtype=torch.float32,device=device)
                for j in range(inner_epochs_pair[1]):
                    #output_reshaped: tensor of shape (batch_size*support_size,no_heads,dim)
                    output_reshaped = model(input_reshaped).to(device)
                    #reshapes output_tensor to (batch_size, support_size,no_heads,dim)
                    output_tensor = output_reshaped.view(base_point_tensor.shape[0], base_point_tensor.shape[1], output_reshaped.shape[1],output_reshaped.shape[2])
                    #displacement_tensor: (batch_size,supp_size,no_heads,d)
                    displacement_tensor=(output_tensor-mapping_tensor.unsqueeze(2)).to(device)
                    loss=optimization_step(displacement_tensor,coefficients_tensor,optimizer,atom_reg,device)
                    inner_loss+=loss.item()
                    if counter%save_increment==0:
                        torch.save(model.state_dict(), 'multi_head_tests/{}/Trial_{:.4f}/multi_head_model_{}.pt'.format(dtype,time_now,counter))
                        print(f"Model saved at counter {counter}")
                    counter=counter+1
            inner_epoch_count=inner_epochs_pair[0]*inner_epochs_pair[1]
            print(f'Epoch [{outer_epoch+1}/{outer_epochs}], Batch [{batch+1}/{no_batches}], counter {counter}, Avg Inner Loss: {inner_loss/inner_epoch_count:.8f}')
    end_time=time.time()
    save_trial('multi_head_hyperparams.csv',
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
                learning_rate =lr,atom_reg=atom_reg)
            
    torch.save(model.state_dict(), 'multi_head_tests/{}/Trial_{:.4f}/final_multi_head_model_{}.pt'.format(dtype,time_now,counter))

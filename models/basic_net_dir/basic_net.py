import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch

class Basic_Net(nn.Module):
    def __init__(self,d,dropout_prob=0.5):
        super(Basic_Net, self).__init__()
        self.layer1 = nn.Linear(d, 128)  # Example for d=2
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout layer
        self.layer2 = nn.Linear(128, d)  # Adjust for output dimension

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x=self.dropout(x)
        x = self.layer2(x)
        return x


def basic_netloss(active_output,active_coefficient,mapping,inactive_coefficient_array, inactive_output_array,device):
    # active: (dimension,)  → active map applied to a point in R^d
    # inactive_array: (m, dimension) → m maps applied to the same point in R^d (fixed)
    # coefficients: (m, 1)   → coefficients for each inactive map
    # input_map: (dimension,) → original (transport) map applied to point in R^d
    criterion = nn.MSELoss(reduction='none')  # No reduction to keep per-element loss

    #active_displacement is mapping(x)-T_i
    active_displacement=mapping-active_output

    #inactive_displacement is mapping(x)-T_j, j\neq i
    inactive_displacement_array = mapping.unsqueeze(0) - inactive_output_array  # Broadcasting the operation
    repulsion=[]
    for i in range(inactive_displacement_array.shape[0]):
        #i indexes inactive models
        inactive_displacement_i=inactive_displacement_array[i] #shape: batch_size x dim

        #computes <mapping(x)-T_i,mapping(x)-T_j>, j\neq i
        loss=active_displacement*inactive_displacement_i
        loss=loss.mean(dim=1).to(device) #shape: batch_size
        repulsion.append(loss)
    repulsion = torch.stack(repulsion).to(device)  # (m, active_dim)
    repulsion = repulsion.mean(dim=1)  # (m,)
    #Computes ||mapping(x)-T_i||^2
    zero_vec=torch.zeros(active_displacement.shape).to(device)
    attraction=(criterion(active_displacement,zero_vec)**2).to(device)
    attraction=attraction.mean(dim=1).to(device)
    # Scale losses by coefficients and sum
    # total_attraction=lambda_i^2*||mapping(x)-T_i||^2
    total_attraction=torch.sum(active_coefficient**2*attraction)
    # total_repulsion=lambda_i*lambda_j*<mapping(x)-T_i,mapping(x)-T_j>, j\neq i
    total_repulsion=active_coefficient*torch.sum(inactive_coefficient_array * repulsion)
    total_loss = total_attraction+total_repulsion
    return total_loss


def basic_net_train(model_array,dataset,outer_epochs, inner_epochs,QP_reg,dim, lr=0.001):
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_array = [model.to(device) for model in model_array]
    
    #define optimizer for each map
    optimizer_list=[]
    for i in np.arange(len(model_array)):
        optimizer_list.append(optim.Adam(model_array[i].parameters(), lr=lr))

    for outer_epoch in range(outer_epochs):
        for idx,data in enumerate(dataset):
            #solve for coefficients
            
            #probably don't need to solve for coefficients every time, or for the first few epochs. 
            base_points=torch.tensor(data.base.points,dtype=torch.float32,requires_grad=False).to(device)
            mapping=torch.tensor(data.mapping,dtype=torch.float32,requires_grad=False).to(device)
            coefficients,min_value=solve_for_coefficients(model_array,base_points,QP_reg)
            coefficients=torch.tensor(coefficients,dtype=torch.float32,requires_grad=False).to(device)

            #update maps loop

            #data_loader loads data from a given measure (i.e. base_points and transport map)
            batch_size=1024
            vectors=vector_field(base_points,mapping)
            data_loader=DataLoader(vectors,batch_size=batch_size,shuffle=True)

            for base_point_batch,mapping_batch in data_loader:
                #base_point_batch is of size batch x dim
                #mapping_batch is of size batch x dim
                base_point_batch=torch.tensor(base_point_batch,requires_grad=False).to(device)
                mapping_batch=torch.tensor(mapping_batch,requires_grad=False).to(device)
                #output_tensor stores the image of the base points under each map T_i
                #it gets updated sequentially over the inner epochs
                output_tensor=torch.zeros(len(model_array),base_point_batch.shape[0],dim).to(device)
                for i in np.arange(len(model_array)):
                    model_i=model_array[i]
                    output=model_i(base_point_batch).to(device) #outputs are images of map T_i, shape is batch x dim
                    output_tensor[i]=output
                
                for inner_epoch in range(inner_epochs):
                    inner_loss=0                
                    for i in np.arange(len(model_array)):
                        active_model=model_array[i]
                        active_model.train()
                        optimizer = optimizer_list[i]

                        #gets outputs from inactive maps
                        inactive_output_array = torch.cat((output_tensor[:i].detach(), output_tensor[i+1:].detach()), dim=0).to(device)
                        inactive_coefficients = torch.cat((coefficients[:i], coefficients[i+1:]), dim=0).to(device)
                        active_coefficient = coefficients[i].to(device)
                        #active_output is T_i(x) for all x in the batch
                        active_output=active_model(base_point_batch)
                        loss=netloss(active_output,active_coefficient,mapping,inactive_coefficients,inactive_output_array,device)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            output_tensor[i]=active_model(base_point_batch).detach()
                        inner_loss+=loss.item()

                    #print(f"Trial: {idx}, Outer Epoch [{outer_epoch+1}/{outer_epochs}], Inner Epoch [{inner_epoch+1}/{inner_epochs}], Loss: {inner_loss:.4f}")
                    print(f"Trial: {idx}, Outer Epoch [{outer_epoch+1}/{outer_epochs}], Inner Epoch [{inner_epoch+1}/{inner_epochs}], Loss: {inner_loss:.8f}")

#input: list of torch.nn, torch.tensor
#output: np.array
def solve_for_coefficients(model_array,base_points,QP_reg):
    map_array=[]
    for model_i in model_array:
        tensor_map=sample_displacements(model_i,base_points)
        map_array.append(tensor_map.cpu().detach().numpy())
    #build and solve QP
    base_supp_size=len(base_points)
    QP_mat=build_QP(map_array,base_supp_size)+QP_reg*np.eye(len(map_array))
    coefficients,min_value=solve_QP(QP_mat)
    return coefficients,min_value

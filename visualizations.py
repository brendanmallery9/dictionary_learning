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
from models.multi_head_dir.multi_head_model import *
from PIL import Image

def simplex_to_2d(points):
    """
    Convert 3D simplex points to 2D coordinates for plotting on a triangle.
    
    :param points: Array of points on the 2-simplex.
    :return: Corresponding 2D coordinates for plotting.
    """
    xs = points[:, 1] + 0.5 * points[:, 2]  # x-coordinates in 2D
    ys = np.sqrt(3) / 2 * points[:, 2]      # y-coordinates in 2D
    return xs, ys

def plot_coefficients(coeff_list,opacity):
    x = coeff_list[:,0]
    y=coeff_list[:,1]
    z=coeff_list[:,2]

    import pandas as pd
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='z', color_continuous_scale='viridis', opacity=opacity)

    # Show plot
    fig.show()
    twod_points=simplex_to_2d(coeff_list)
    plt.scatter(twod_points[0],twod_points[1])
    plt.show()

    




def plot_atoms(path, model, n, dim):
    points = np.random.rand(n, dim)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode

    output = model(torch.tensor(points, dtype=torch.float32))
    num_heads = model.no_heads

    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 6))

    for i in range(num_heads):
        image_points = output[:, i, :].detach().numpy()

        ax = axes[i] if num_heads > 1 else axes  # Handle case where num_heads == 1
        ax.scatter(image_points[:, 0], image_points[:, 1], color='red', label='Target', alpha=0.25)
        ax.set_title(f'Map {i}')
        ax.legend()

    plt.tight_layout()
    plt.show()



def plot_training(path,name,model,n,dim,total_iterations,increments):
    os.makedirs('{}_images'.format(name),exist_ok=True)
    for i in np.arange(0, total_iterations, increments):  # Loop over checkpoints
        model.load_state_dict(torch.load('{}/multi_head_model_{}.pt'.format(path,i)))
        model.eval()  # Set to evaluation mode
        points = np.random.rand(n, dim)
        output = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()

        # Fixed axis limits
        x_min, x_max = -1, 2
        y_min, y_max = -1, 2
        plt.figure(figsize=(8, 6))  # Fixed figure size
        # Plot source points
        plt.scatter(points[:, 0], points[:, 1], label="Source", alpha=0.5, color="black")

        for j in range(3):
            image_points = output[:, j, :]
            
            # Quiver plot showing displacement
            plt.quiver(points[:, 0], points[:, 1], 
                    image_points[:, 0] - points[:, 0], 
                    image_points[:, 1] - points[:, 1], 
                    angles="xy", scale_units="xy", scale=1, width=0.0003, alpha=0.3)

            # Scatter plot for each target set
            plt.scatter(image_points[:, 0], image_points[:, 1], alpha=0.3, label=f"Learned Map {j}")

        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.legend(ncol=2, fontsize="small", loc="upper right")  # Adjust legend placement
        plt.title(f"Model Output at Step {i}")
        plt.savefig('{}_images/output_{}'.format(name,i))
        plt.show()  # Show each plot separately
    # Collect all PNG files in the folder that start with 'output_'
    png_files = [f for f in os.listdir('{}_images'.format(name)) if f.startswith('output_') and f.endswith('.png')]

    # Sort the files numerically based on the number in the filename (i.e., 'output_i.png')
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Load images into a list
    images = [Image.open(os.path.join('{}_images'.format(name), file)) for file in png_files]

    # Save the images as a GIF
    safe_name=name.replace('.','_')
    images[0].save('{}_gif.gif'.format(safe_name), save_all=True, append_images=images[1:], loop=0, duration=200)


#Plots a semisupervised training run as a grid of images
def semi_supervised_plot_atoms(directory, model, n, dim,supervised):
    #directory is string to directory which contains: paths.npy, basis_list.npy,supervised_classes.npy
    #model: torch.nn
    #n: integer
    #dim: integer
    #supervised: Boolean
    model=model.to('cpu')

    paths_file='{}/paths.npy'.format(directory)
    basis_file='{}/basis_list.npy'.format(directory)
    class_nos_file='{}/supervised_classes.npy'.format(directory)
    paths=np.load(paths_file, allow_pickle=True)
    basis=np.load(basis_file,allow_pickle=True)
    class_nos=np.load(class_nos_file,allow_pickle=True)
    for path in paths:
        print(path)

    supervised_heads_dict={}
    for i,j in enumerate(class_nos):
        supervised_heads_dict[j]=basis[i][j]

    no_heads = model.no_heads
    k=len(paths)
    #class_nos=[4, 0, 3, 2, 1]
    fig, axs = plt.subplots(k+1, no_heads, figsize=(3 * no_heads, 3 * k))  # Adjust scale as needed
    # Make sure axs is always 2D for consistency
    if k == 1:
        axs = np.expand_dims(axs, axis=0)
    if no_heads == 1:
        axs = np.expand_dims(axs, axis=1)
    for i, path in enumerate(paths):
        #i indexes rows
        
        iteration_class_nos = class_nos[:i]  # Class numbers that are supervised by iteration i
        points = np.random.rand(n, dim)

        # Load the model
        model.load_state_dict(torch.load(path))
        model.eval()

        with torch.no_grad():
            output = model(torch.tensor(points, dtype=torch.float32))
        #plots the entries of the i'th row, j indexes columns
        for j in range(no_heads):
            ax = axs[i][j]
            if supervised == True and j in iteration_class_nos:  # if j is supervised in iteration i
                #basis[i] is the observed classes 
                image_points = supervised_heads_dict[j].mapping
                label = 'Supervised'  # Label for supervised bases
                color = 'blue'  # Change color for supervised points
            else:  # Model output
                image_points = output[:, j, :].detach().numpy()
                label = 'Model Output'  # Label for model output
                color = 'red'  # Use red for model points

        # Plot with different markers or colors
            ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)
            
            ax.set_title(f'Self Supervision Round {i}, Head {j}')
            ax.axis('equal')  # Make aspect ratio square
            ax.set_xticks([])
            ax.set_yticks([])
            
            #ax.legend(loc='upper right')
    for j in range(no_heads):
        ax=axs[k][j]
        image_points = supervised_heads_dict[j].mapping
        label = 'Supervised' 
        color = 'blue' 
        ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)
        
        ax.set_title(f'Model {i}, Head {j}')
        ax.axis('equal')  
        ax.set_xticks([])
        ax.set_yticks([])
        
        #ax.legend(loc='upper right')
    # Adjust layout and show plot
    fig.tight_layout()

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.25, markersize=10, label='Supervised'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.25, markersize=10, label='Model Output')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, .05))

    # Add title below the plot
    fig.subplots_adjust(bottom=0.15)  # Add space at bottom
    fig.text(0.5, .1, directory, ha='center', fontsize=14)
    plt.savefig(os.path.join(directory, 'semi_supervised_plot_atoms.png'))
    plt.show()

    

# plots all generators on the same plot, with example data points
def semi_supervised_plot_all(directory, model, n, dim, supervised):
    #dir is string to directory which contains: paths.npy, basis_list.npy,supervised_classes.npy
    #model is torch.nn
    #n is integer
    #dim is integer
    #supervised is Boolean
    model = model.to('cpu')

    paths_file = '{}/paths.npy'.format(directory)
    basis_file = '{}/basis_list.npy'.format(directory)
    class_nos_file = '{}/supervised_classes.npy'.format(directory)
    paths = np.load(paths_file, allow_pickle=True)
    basis = np.load(basis_file, allow_pickle=True)
    class_nos = np.load(class_nos_file, allow_pickle=True)

    supervised_heads_dict = {}
    for i, j in enumerate(class_nos):
        supervised_heads_dict[j] = basis[i][j]

    no_heads = model.no_heads
    k = len(paths)

    # Create a single figure for all points
    fig, ax = plt.subplots(figsize=(8, 8))  # Single plot for all points

    for i, path in enumerate(paths):
        if i>=0:
            iteration_class_nos = class_nos[:i]  # Class numbers that are supervised by iteration i
            points = np.random.rand(n, dim)

            # Load the model
            model.load_state_dict(torch.load(path))
            model.eval()

            with torch.no_grad():
                output = model(torch.tensor(points, dtype=torch.float32))
            
            # Plot the points for each iteration and head
            for j in range(no_heads):
                if supervised == True and j in iteration_class_nos:  # if j is supervised in iteration i
                    image_points = supervised_heads_dict[j].mapping
                    label = 'Supervised'  # Label for supervised bases
                    color = 'blue'  # Change color for supervised points
                else:  # Model output
                    image_points = output[:, j, :].detach().numpy()
                    label = 'Model Output'  # Label for model output
                    color = 'red'  # Use red for model points

                # Plot the points
                ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)

    # Plot the supervised points separately (all of them together)
    for j in range(no_heads):
        image_points = supervised_heads_dict[j].mapping
        ax.scatter(image_points[:, 0], image_points[:, 1], color='blue', alpha=0.25, label='Supervised')

    # Add titles and labels
    ax.set_title('All Iterations and Heads on One Plot')
    ax.axis('equal')  # Make aspect ratio square
    ax.tick_params(axis='both', which='both', direction='in', length=6)  # Enable ticks
    ax.set_xticks(np.linspace(-1, 1, 5))  # Set specific x-axis ticks
    ax.set_yticks(np.linspace(-1, 1, 5))  # Set specific y-axis ticks

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.25, markersize=10, label='Supervised'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.25, markersize=10, label='Model Output')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, .5))

    # Show the plot
    plt.title(dir)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'semi_supervised_plot_all.png'))
    plt.show()

#example:
#dir='training_runs/2025-05-09_zero_four_eight_cvx_randomref_(8000,200)_self_supervised_test_(8000,200)_some_self_supervise'
#model=FatFourLayer_Net_Multihead(d=2,no_heads=5,dropout_prob=0)
#semi_supervised_plot_atoms(dir, model, 1000, 2,True)

def semi_supervised_plot_atoms(directory, model, n, dim,supervised):
    #directory: string
    #model: torch.nn
    #n: integer
    #dim: integer
    #supervised: Boolean
    model=model.to('cpu')

    paths_file='{}/paths.npy'.format(directory)
    basis_file='{}/basis_list.npy'.format(directory)
    class_nos_file='{}/supervised_classes.npy'.format(directory)
    paths=np.load(paths_file, allow_pickle=True)
    basis=np.load(basis_file,allow_pickle=True)
    class_nos=np.load(class_nos_file,allow_pickle=True)
    for path in paths:
        print(path)

    supervised_heads_dict={}
    for i,j in enumerate(class_nos):
        supervised_heads_dict[j]=basis[i][j]

    no_heads = model.no_heads
    k=len(paths)
    #class_nos=[4, 0, 3, 2, 1]
    fig, axs = plt.subplots(k+1, no_heads, figsize=(3 * no_heads, 3 * k))  # Adjust scale as needed
    # Make sure axs is always 2D for consistency
    if k == 1:
        axs = np.expand_dims(axs, axis=0)
    if no_heads == 1:
        axs = np.expand_dims(axs, axis=1)
    for i, path in enumerate(paths):
        #i indexes rows
        
        iteration_class_nos = class_nos[:i]  # Class numbers that are supervised by iteration i
        points = np.random.rand(n, dim)

        # Load the model
        model.load_state_dict(torch.load(path))
        model.eval()

        with torch.no_grad():
            output = model(torch.tensor(points, dtype=torch.float32))
        #plots the entries of the i'th row, j indexes columns
        for j in range(no_heads):
            ax = axs[i][j]
            if supervised == True and j in iteration_class_nos:  # if j is supervised in iteration i
                #basis[i] is the observed classes 
                image_points = supervised_heads_dict[j].mapping
                label = 'Supervised'  # Label for supervised bases
                color = 'blue'  # Change color for supervised points
            else:  # Model output
                image_points = output[:, j, :].detach().numpy()
                label = 'Model Output'  # Label for model output
                color = 'red'  # Use red for model points

        # Plot with different markers or colors
            ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)
            
            ax.set_title(f'Self Supervision Round {i}, Head {j}')
            ax.axis('equal')  # Make aspect ratio square
            ax.set_xticks([])
            ax.set_yticks([])
            
            #ax.legend(loc='upper right')
    for j in range(no_heads):
        ax=axs[k][j]
        image_points = supervised_heads_dict[j].mapping
        label = 'Supervised' 
        color = 'blue' 
        ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)
        
        ax.set_title(f'Model {i}, Head {j}')
        ax.axis('equal')  
        ax.set_xticks([])
        ax.set_yticks([])
        
        #ax.legend(loc='upper right')
    # Adjust layout and show plot
    fig.tight_layout()

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.25, markersize=10, label='Supervised'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.25, markersize=10, label='Model Output')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, .05))

    # Add title below the plot
    fig.subplots_adjust(bottom=0.15)  # Add space at bottom
    fig.text(0.5, .1, directory, ha='center', fontsize=14)

    plt.show()

    


def semi_supervised_plot_all(directory, model, n, dim, supervised):
    model = model.to('cpu')

    paths_file = '{}/paths.npy'.format(directory)
    basis_file = '{}/basis_list.npy'.format(directory)
    class_nos_file = '{}/supervised_classes.npy'.format(directory)
    paths = np.load(paths_file, allow_pickle=True)
    basis = np.load(basis_file, allow_pickle=True)
    class_nos = np.load(class_nos_file, allow_pickle=True)

    supervised_heads_dict = {}
    for i, j in enumerate(class_nos):
        supervised_heads_dict[j] = basis[i][j]

    no_heads = model.no_heads
    k = len(paths)

    # Create a single figure for all points
    fig, ax = plt.subplots(figsize=(8, 8))  # Single plot for all points

    for i, path in enumerate(paths):
        if i>=0:
            iteration_class_nos = class_nos[:i]  # Class numbers that are supervised by iteration i
            points = np.random.rand(n, dim)

            # Load the model
            model.load_state_dict(torch.load(path))
            model.eval()

            with torch.no_grad():
                output = model(torch.tensor(points, dtype=torch.float32))
            
            # Plot the points for each iteration and head
            for j in range(no_heads):
                if supervised == True and j in iteration_class_nos:  # if j is supervised in iteration i
                    image_points = supervised_heads_dict[j].mapping
                    label = 'Supervised'  # Label for supervised bases
                    color = 'blue'  # Change color for supervised points
                else:  # Model output
                    image_points = output[:, j, :].detach().numpy()
                    label = 'Model Output'  # Label for model output
                    color = 'red'  # Use red for model points

                # Plot the points
                ax.scatter(image_points[:, 0], image_points[:, 1], color=color, alpha=0.25, label=label)

    # Plot the supervised points separately (all of them together)
    for j in range(no_heads):
        image_points = supervised_heads_dict[j].mapping
        ax.scatter(image_points[:, 0], image_points[:, 1], color='blue', alpha=0.25, label='Supervised')

    # Add titles and labels
    ax.set_title('All Iterations and Heads on One Plot')
    ax.axis('equal')  # Make aspect ratio square
    ax.tick_params(axis='both', which='both', direction='in', length=6)  # Enable ticks
    ax.set_xticks(np.linspace(-1, 1, 5))  # Set specific x-axis ticks
    ax.set_yticks(np.linspace(-1, 1, 5))  # Set specific y-axis ticks

    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.25, markersize=10, label='Supervised'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.25, markersize=10, label='Model Output')
    ]
    #fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, .5))

    # Show the plot
    plt.title(dir)
    plt.tight_layout()
    plt.show()


#EXAMPLE:
#dir='training_runs/2025-05-18_zero_four_eight_cvx_randomref_(8000,400)_(0,4,8)_5_heads_(8000,400)_40_reps'
#model=FatFourLayer_Net_Multihead(d=2,no_heads=5,dropout_prob=0)
#semi_supervised_plot_atoms(dir, model, 1000, 2,True)
#semi_supervised_plot_all(dir, model, 100, 2,True)

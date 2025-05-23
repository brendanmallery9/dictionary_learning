
import os
import sys
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
from generate_data import *
from self_supervised_training import *
import math
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import qpth
from self_supervised_cuda_utils import *
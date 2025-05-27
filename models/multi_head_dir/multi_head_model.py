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

class Basic_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(Basic_Net_Multihead, self).__init__()
        self.no_heads=no_heads
        self.layer1 = nn.Linear(d, 128)  
        self.dropout = nn.Dropout(p=dropout_prob)  
        self.heads = nn.ModuleList([nn.Linear(128, d) for _ in range(no_heads)])

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        # outputs: (batch_size, no_heads, d)
        outputs = torch.stack([head(x) for head in self.heads], dim=1)  
        return outputs
        
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        for layer in [self.layer1] + list(self.heads):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class Fat_Basic_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(Fat_Basic_Net_Multihead, self).__init__()
        self.no_heads = no_heads
        self.layer1 = nn.Linear(d, 256)  
        self.dropout = nn.Dropout(p=dropout_prob)  
        self.heads = nn.ModuleList([nn.Linear(256, d) for _ in range(no_heads)])

    def forward(self, x):
        # Apply first linear layer and ReLU activation
        x = torch.relu(self.layer1(x))
        
        # Apply dropout after the first layer
        x = self.dropout(x)

        # Now apply each head to the output of the first layer and stack the results
        outputs = torch.stack([head(x) for head in self.heads], dim=1)
        
        return outputs

    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize weights and biases for the first layer and each head's layer
        nn.init.xavier_uniform_(self.layer1.weight)  # For the first layer
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        # Now initialize weights and biases for each head in the heads ModuleList
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)


class ThreeLayer_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(ThreeLayer_Net_Multihead, self).__init__()
        self.no_heads = no_heads
        self.layer1 = nn.Linear(d, 128)  
        self.dropout1 = nn.Dropout(p=dropout_prob)  
        self.layer2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.heads = nn.ModuleList([nn.Linear(256, d) for _ in range(no_heads)])

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # Apply dropout after layer1
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)  # Apply dropout after layer2
        # outputs: (batch_size, no_heads, d)
        outputs = torch.stack([head(x) for head in self.heads], dim=1)  
        return outputs

    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize layer1
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        # Initialize layer2
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        # Initialize each head in the heads ModuleList
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

class FatFourLayer_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(FatFourLayer_Net_Multihead, self).__init__()

        self.no_heads = no_heads
        
        self.layer1 = nn.Linear(d, 256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(512, 256)  # Add this to connect back to 128 dimensions
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        self.heads = nn.ModuleList([nn.Linear(256, d) for _ in range(no_heads)])
        self.register_buffer("head_mask", torch.ones(no_heads))  # <-- Add this line

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        
        outputs = []
        for i, head in enumerate(self.heads):
            # If head is frozen (mask[i] = 0), the output will be zeros
            head_output = head(x) * self.head_mask[i]
            outputs.append(head_output)

        outputs = torch.stack(outputs, dim=1)  # Use the masked outputs
        return outputs

    def freeze_heads(self, head_indices):
        """
        Freeze specific heads by setting their mask values to 0.
        
        Args:
            head_indices: List of indices of heads to freeze
        """
        with torch.no_grad():
            for idx in head_indices:
                if 0 <= idx < self.no_heads:
                    self.head_mask[idx] = 0
    
    def unfreeze_heads(self, head_indices=None):
        """
        Unfreeze specific heads by setting their mask values to 1.
        If head_indices is None, unfreeze all heads.
        
        Args:
            head_indices: List of indices of heads to unfreeze, or None to unfreeze all
        """
        with torch.no_grad():
            if head_indices is None:
                # Unfreeze all heads
                self.head_mask.fill_(1)
            else:
                for idx in head_indices:
                    if 0 <= idx < self.no_heads:
                        self.head_mask[idx] = 1
    
    def get_frozen_status(self):
        """
        Return a list of booleans indicating which heads are frozen.
        
        Returns:
            List of booleans where True means the head is frozen
        """
        return [(mask == 0).item() for mask in self.head_mask]
    

    
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize layer1
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        # Initialize layer2
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        # Initialize layer3
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
        
        # Initialize each head in the heads ModuleList
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
                
        # Reset the head mask to all ones (all heads active)
        with torch.no_grad():
            self.head_mask.fill_(1)

'''
class FatFourLayer_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(FatFourLayer_Net_Multihead, self).__init__()
        self.no_heads = no_heads
        
        self.layer1 = nn.Linear(d, 256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(512, 512)  
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        self.heads = nn.ModuleList([nn.Linear(512, d) for _ in range(no_heads)])
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.layer3(x))  # Use layer3 to reduce dimensions back to 128
        x = self.dropout3(x)
        
        # outputs: (batch_size, no_heads, d)
        outputs = torch.stack([head(x) for head in self.heads], dim=1)
        return outputs
    
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize layer1
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        # Initialize layer2
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        # Initialize layer3
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
        
        # Initialize each head in the heads ModuleList
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
'''
class FourLayer_Net_Multihead_With_Skip(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(FourLayer_Net_Multihead_With_Skip, self).__init__()
        self.no_heads = no_heads
        
        # Main transformation layers
        self.layer1 = nn.Linear(d, 128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Projection layer for the residual connection
        # This maps the input dimension to match the output of layer3
        self.skip_proj = nn.Linear(d, 128)
        
        # Output heads
        self.heads = nn.ModuleList([nn.Linear(128, d) for _ in range(no_heads)])
    
    def forward(self, x):
        # Create the skip connection with projection
        # This skips the entire core transformation path
        skip = self.skip_proj(x)
        
        # Main transformation path
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        
        # Add the skip connection to the output of the transformation path
        # This creates a residual connection spanning the entire network
        x = x + skip
        
        # Apply the multi-heads
        outputs = torch.stack([head(x) for head in self.heads], dim=1)
        return outputs
    
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
            
        nn.init.xavier_uniform_(self.skip_proj.weight)
        if self.skip_proj.bias is not None:
            nn.init.zeros_(self.skip_proj.bias)
        
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)


class FourLayer_Net_Multihead_With_Skip_and_Batch(nn.Module):
    #should dropout to compensate for batchnorm
    def __init__(self, d, no_heads, dropout_prob):
        super(FourLayer_Net_Multihead_With_Skip_and_Batch, self).__init__()
        self.no_heads = no_heads
        
        # Main transformation layers
        self.layer1 = nn.Linear(d, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(128, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(256, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Projection layer for the residual connection
        self.skip_proj = nn.Linear(d, 128)
        
        # Output heads
        self.heads = nn.ModuleList([nn.Linear(128, d) for _ in range(no_heads)])
    
    def forward(self, x):
        skip = self.skip_proj(x)
        
        x = torch.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.batchnorm3(self.layer3(x)))
        x = self.dropout3(x)
        
        x = x + skip
        
        outputs = torch.stack([head(x) for head in self.heads], dim=1)
        return outputs
    
    def reset_parameters(self):
        torch.manual_seed(42)
        
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
            
        nn.init.xavier_uniform_(self.skip_proj.weight)
        if self.skip_proj.bias is not None:
            nn.init.zeros_(self.skip_proj.bias)
        
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
'''
class FatFiveLayer_Net_Multihead_Deep_Split(nn.Module):
    def __init__(self, d, no_heads, dropout_prob):
        super(FatFiveLayer_Net_Multihead_Deep_Split, self).__init__()
        self.no_heads = no_heads
        
        self.layer1 = nn.Linear(d, 256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.layer3 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Create two separate layers for each head
        self.head_layers = nn.ModuleList([nn.Linear(512, 256) for _ in range(no_heads)])
        self.head_dropouts = nn.ModuleList([nn.Dropout(p=dropout_prob) for _ in range(no_heads)])
        
        # Add projection layers to match dimensions for the residual connection
        self.projection_layers = nn.ModuleList([nn.Linear(512, 256) for _ in range(no_heads)])
        
        self.output_layers = nn.ModuleList([nn.Linear(256, d) for _ in range(no_heads)])
    
    def forward(self, x):
        # Shared layers
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        
        # Process through each head's dedicated layers
        outputs = []
        for i in range(self.no_heads):
            # Store input for skip connection
            residual = self.projection_layers[i](x)  # Project to match dimensions
            
            # Process through the head layer
            head_out = torch.relu(self.head_layers[i](x))
            head_out = self.head_dropouts[i](head_out)
            
            # Add skip connection
            head_out = head_out + residual
            
            # Final output layer
            head_out = self.output_layers[i](head_out)
            outputs.append(head_out)
        
        # Stack the outputs: (batch_size, no_heads, d)
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize shared layers
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
        
        # Initialize each head's dedicated layers
        for i in range(self.no_heads):
            # Initialize head layer
            nn.init.xavier_uniform_(self.head_layers[i].weight)
            if self.head_layers[i].bias is not None:
                nn.init.zeros_(self.head_layers[i].bias)
            
            # Initialize projection layer
            nn.init.xavier_uniform_(self.projection_layers[i].weight)
            if self.projection_layers[i].bias is not None:
                nn.init.zeros_(self.projection_layers[i].bias)
            
            # Initialize output layer
            nn.init.xavier_uniform_(self.output_layers[i].weight)
            if self.output_layers[i].bias is not None:
                nn.init.zeros_(self.output_layers[i].bias)
'''


class SixLayer_Net_Multihead(nn.Module):
    def __init__(self, d, no_heads, dropout_prob,base_supp_size):
        super(SixLayer_Net_Multihead, self).__init__()

        self.no_heads = no_heads
        self.base_supp_size=base_supp_size
        
        self.layer1 = nn.Linear(d, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.layer2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.layer3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(p=dropout_prob)

        self.layer4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.layer5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(p=dropout_prob)
        
        self.heads = nn.ModuleList([nn.Linear(256, d) for _ in range(no_heads)])
        self.register_buffer("head_mask", torch.ones(no_heads))

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout5(x)
        outputs = []
        for i, head in enumerate(self.heads):
            head_output = head(x) * self.head_mask[i]
            outputs.append(head_output)

        outputs = torch.stack(outputs, dim=1)
        return outputs


    def freeze_heads(self, head_indices):
        """
        Freeze specific heads by setting their mask values to 0.
        
        Args:
            head_indices: List of indices of heads to freeze
        """
        with torch.no_grad():
            for idx in head_indices:
                if 0 <= idx < self.no_heads:
                    self.head_mask[idx] = 0
    
    def unfreeze_heads(self, head_indices=None):
        """
        Unfreeze specific heads by setting their mask values to 1.
        If head_indices is None, unfreeze all heads.
        
        Args:
            head_indices: List of indices of heads to unfreeze, or None to unfreeze all
        """
        with torch.no_grad():
            if head_indices is None:
                # Unfreeze all heads
                self.head_mask.fill_(1)
            else:
                for idx in head_indices:
                    if 0 <= idx < self.no_heads:
                        self.head_mask[idx] = 1
    
    def get_frozen_status(self):
        """
        Return a list of booleans indicating which heads are frozen.
        
        Returns:
            List of booleans where True means the head is frozen
        """
        return [(mask == 0).item() for mask in self.head_mask]
    

    
    def reset_parameters(self):
        """Reset parameters with a fixed random seed."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Initialize layer1
        nn.init.xavier_uniform_(self.layer1.weight)
        if self.layer1.bias is not None:
            nn.init.zeros_(self.layer1.bias)
        
        # Initialize layer2
        nn.init.xavier_uniform_(self.layer2.weight)
        if self.layer2.bias is not None:
            nn.init.zeros_(self.layer2.bias)
        
        # Initialize layer3
        nn.init.xavier_uniform_(self.layer3.weight)
        if self.layer3.bias is not None:
            nn.init.zeros_(self.layer3.bias)
        
        # Initialize each head in the heads ModuleList
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
                
        # Reset the head mask to all ones (all heads active)
        with torch.no_grad():
            self.head_mask.fill_(1)
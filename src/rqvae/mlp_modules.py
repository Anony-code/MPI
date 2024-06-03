"""borrowed and modified from https://github.com/CompVis/taming-transformers"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
from layers import (AttnBlock, Downsample, Normalize, ResnetBlock, Upsample, nonlinearity)
from torch.nn import Linear


class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_dim):
        super(SimpleDNN, self).__init__()
        # Initialize the ModuleList for all layers
        self.layers = nn.ModuleList()
        # Add the first layer (input layer to first hidden layer)
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        # Dynamically add all hidden layers specified in the hidden_layers list
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        # Add the last layer (last hidden layer to output layer)
        self.last = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        # Apply ReLU activation function after each layer except the last
        for layer in self.layers:
            x = F.relu(layer(x))
        # No activation function after the last layer
        x = self.last(x)
        return x

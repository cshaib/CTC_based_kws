import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

class LabelModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LabelModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # data shape: [batch, seq, feature]
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=output_dim)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.linear(self.relu(out[:,-1]))
        out = self.softmax(out)
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


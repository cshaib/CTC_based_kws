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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LabelModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LabelModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # data shape: [batch, seq, feature] --> (e.g.,) [16, 35843, 30]
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.linear = nn.Linear(hidden_dim, 42)
        self.relu = nn.ReLU()
        self.softmax = nn.functional.log_softmax
        
    def forward(self, x, h):
        x = x.permute(0, 2, 1)
        # x = x.unsqueeze(-1)
        out, h = self.gru(x, h)
        out = self.softmax(out, dim=2)
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

9

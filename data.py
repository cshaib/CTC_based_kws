import torch
import torch.utils.data
import os
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader

def get_data(filepath):
    dataset = pd.read_pickle(filepath)

    unique_id = list(range(len(dataset['name'])))
    dataset['unique_id'] =  unique_id
 
    unique_id_col = 'unique_id'
    audio_feat_col = 'audio'
    label_col = 'label'

    dataset[audio_feat_col] = dataset[audio_feat_col]
    dataset[unique_id_col] = dataset[unique_id_col].to_numpy()
    dataset[label_col] = dataset[label_col].to_numpy()

    ids = list(dataset[unique_id_col])
    total_len = len(ids)
    np.random.shuffle(ids)


    # set parameters for DataLoader -- num_workers = cores
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 0
              }

    # create data generator
    _len = [x.shape[1] for x in dataset['audio']]
    _lab_len = [len(x) for x in dataset['label']]

    padded_data = []
    padded_label = []

    for val in dataset['audio']:
        if val.shape[1] < max(_len):
            val = np.pad(val, ((0,0), (0, max(_len)-len(val[1]))))
        padded_data.append(val)

    for lab in dataset['label']:
        if len(lab) < max(_lab_len):
            lab = np.array(np.pad(lab, ((0, max(_lab_len)-len(lab)))))
        padded_label.append(lab)


    dataset['padded_audio'] = padded_data
    dataset['padded_label'] = padded_label

    labels = {}

    for i in dataset[unique_id_col]:
        labels[i] = dataset['padded_label'][i]

    data_set = AudioDataset(data=dataset, labels=labels, list_IDs=ids, in_len=_len, targ_len=_lab_len)
    data_generator = DataLoader(data_set, **params)
    return max(_len), data_generator

class AudioDataset(torch.utils.data.Dataset):

  def __init__(self, data, list_IDs: list, labels: dict, in_len, targ_len):
    """Create custom torch Dataset. """

    self.data = data
    self.labels = labels
    self.list_IDs = list_IDs
    self.label_lengths = targ_len
    self.input_lengths = in_len

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    # select sample
    ID = self.list_IDs[index]
    original_len = self.input_lengths[index]
    original_label_len = self.label_lengths[index]

    # Load data 
    X = self.data[self.data['unique_id'] == ID]['padded_audio'].values[0]
    y = self.labels[ID]

    return original_len, original_label_len, self.list_IDs, torch.from_numpy(X), torch.tensor(y) 

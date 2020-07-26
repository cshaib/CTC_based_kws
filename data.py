import torch
import torch.utils.data
import torchaudio
import os
import soundfile as sf
import numpy as np
import configparser
import multiprocessing
import json
import pandas as pd


def get_data(path_to_data):
    pass


def __get_audio_features__(audio):
    pass

# class PhonemeTokenizer:
# # tokenize phonemes into IDs  
# 	def __init__(self, phoneme_to_phoneme_index):
# 		self.phoneme_to_phoneme_index = phoneme_to_phoneme_index
# 		self.phoneme_index_to_phoneme = {v: k for k, v in self.phoneme_to_phoneme_index.items()}

# 	def EncodeAsIds(self, phoneme_string):
# 		return [self.phoneme_to_phoneme_index[p] for p in phoneme_string.split()]

# 	def DecodeIds(self, phoneme_ids):
# 		return " ".join([self.phoneme_index_to_phoneme[id] for id in phoneme_ids])


class GenericDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, list_IDs: list, labels: dict, max_len: int=128):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        X = self.data[ID]
        y = self.labels[ID]

        return self.list_IDs, torch.tensor(X), torch.tensor(y) 


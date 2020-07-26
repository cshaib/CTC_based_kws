import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from model import LabelModel
from training import train_label_model, train_wakeword_model, evaluate_audio
from data import get_data

def main():

    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG) # if you want to pipe the logging to a file
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='path to the json config file')
    args = parser.parse_args()

    # load config file 
    with open(args.config_dir) as f:
        config = json.load(f)

    # check for or pretrain label model
    if not config["label_model_path"]:
        # get pretraining data
        pretrain_data_generator = get_data(config["pretrain_label_data"])
        label_model = train_label_model(pretrain_data_generator, config["num_labels"], learn_rate=0.001)
    else: 
        label_model = torch.load(config["label_model_path"])

    # get keywords of interest, train wakeword model
    keyword_data_generator = get_data(config["train_wakeword_data"])
    wakeword_model = train_wakeword_model(keyword_data_generator, config["labels"], label_model, query_by_string=True)

    # evaluate on new audio 
    test_data_generator = get_data(config["test_data"])
    scores = {}

    for idx, audio, label in enumerate(keyword_data_generator):
        scores[idx] = evaluate_audio(audio, wakeword_model, label_model)

    print(scores)
    
    with open("keyword_scores.txt", w+) as f:
        f.write(scores)

    return


if __name__ == "__main__":
    main()

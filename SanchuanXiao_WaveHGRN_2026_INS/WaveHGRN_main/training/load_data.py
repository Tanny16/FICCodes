import pandas as pd
import csv
import numpy as np
import os
from torch import nn
import torch
import random
import pickle

def read_data2(task, period, rnn_length):
    data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "CAS_data"
    )
    dataset_specs = {
        "CSI300": ("CSI300.csv", 225, 1309),
        "CSI500": ("CSI500.csv", 371, 1309),
        "CAS_A": ("data202006dao202506.csv", 245, 1213),
        "CAS_B": ("data201806dao202306.csv", 198, 1214),
        "CAS_C": ("data201606dao202106.csv", 125, 1217),
    }
    if period not in dataset_specs:
        raise ValueError(f"Unsupported period: {period}. Use CSI300, CSI500, CAS_A, CAS_B, or CAS_C.")

    data_name, num_stock, num_timestep = dataset_specs[period]
    data_path = os.path.join(data_root, data_name)

    market_data = pd.read_csv(data_path, header=None)


    x_col = market_data.iloc[:, 0:9]
    y_col = market_data.iloc[:, -1]

    x_ = x_col.to_numpy().astype(np.float32).reshape(num_stock, num_timestep, -1).transpose(1, 0, 2)
    #y = y_col.to_numpy().astype(np.float32).reshape(num_stock, num_timestep).transpose(1, 0)
    y = y_col.to_numpy().astype(np.int64).reshape(num_stock, num_timestep).transpose(1, 0)
    length = num_timestep

    train_index = int(0.8 * length)
    vaild_index = int(0.9 * length)

    feature_mean = x_[:train_index].mean(axis=(0, 1), keepdims=True)
    feature_std = x_[:train_index].std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x_ = (x_ - feature_mean) / feature_std

    x_train, y_train = x_[:train_index], y[:train_index]
    x_eval, y_eval = x_[train_index - rnn_length + 1:vaild_index], y[train_index - rnn_length + 1:vaild_index]
    x_test, y_test = x_[vaild_index - rnn_length + 1:], y[vaild_index - rnn_length + 1:]



    return x_train,x_eval,x_test,y_train,y_eval,y_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


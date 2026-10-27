import os
import random

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_data2(task, period, rnn_length):
    if task != "cls":
        raise ValueError("FFreGNN uses the classification task")

    dataset_config = {
        "CAS_A": ("data202006dao202506.csv", 245, 1213),
        "CAS_B": ("data201806dao202306.csv", 198, 1214),
        "CAS_C": ("data201606dao202106.csv", 125, 1217),
    }
    if period not in dataset_config:
        raise ValueError(f"Unsupported Chinese dataset: {period}")

    data_file, num_stock, num_timestep = dataset_config[period]
    data_path = os.path.join(PROJECT_ROOT, "CAS_data", data_file)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. "
            "Place the three Chinese CSV files in the CAS_data directory."
        )

    market_data = pd.read_csv(data_path, header=None)
    if len(market_data) != num_stock * num_timestep:
        raise ValueError(
            f"{period} row count mismatch: got {len(market_data)}, "
            f"expected {num_stock * num_timestep}"
        )

    x_ = market_data.iloc[:, 0:9].to_numpy(dtype=np.float32).reshape(
        num_stock, num_timestep, 9
    ).transpose(1, 0, 2)
    y = market_data.iloc[:, -1].to_numpy(dtype=np.int64).reshape(
        num_stock, num_timestep
    ).transpose(1, 0)

    rnn_length = int(rnn_length)
    train_index = int(0.8 * num_timestep)
    valid_index = int(0.9 * num_timestep)
    x_train, y_train = x_[:train_index], y[:train_index]
    x_eval, y_eval = (
        x_[train_index - rnn_length + 1:valid_index],
        y[train_index - rnn_length + 1:valid_index],
    )
    x_test, y_test = (
        x_[valid_index - rnn_length + 1:],
        y[valid_index - rnn_length + 1:],
    )

    return x_train, x_eval, x_test, y_train, y_eval, y_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import os
import random

import numpy as np
import pandas as pd
import torch


def read_data_cas(period, rnn_length):
    cas_config = {
        "CAS_133": ("data2016dao2020.csv", 133, 1218),
        "CAS_187": ("data2018dao2022.csv", 187, 1215),
        "CAS_232": ("data2020dao2024.csv", 232, 1211),
    }
    if period not in cas_config:
        raise ValueError(f"Unsupported CAS dataset: {period}")

    data_file, expected_stocks, expected_dates = cas_config[period]
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", data_file)
    columns = [
        "stock", "date", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9", "label",
    ]
    market_data = pd.read_csv(data_path, header=None, names=columns)

    stock_ids = market_data["stock"].drop_duplicates().tolist()
    dates = sorted(market_data["date"].unique().tolist())
    if len(stock_ids) != expected_stocks or len(dates) != expected_dates:
        raise ValueError(
            f"{period} shape mismatch: got {len(stock_ids)} stocks and {len(dates)} dates, "
            f"expected {expected_stocks} stocks and {expected_dates} dates"
        )

    index = pd.MultiIndex.from_product([stock_ids, dates], names=["stock", "date"])
    panel = market_data.set_index(["stock", "date"]).reindex(index)
    feature_columns = columns[2:11]
    valid_mask = (
        panel["label"].notna().to_numpy()
        .reshape(expected_stocks, expected_dates)
        .transpose(1, 0)
    )

    feature_panel = panel[feature_columns].groupby(level=0).ffill().groupby(level=0).bfill()
    panel.loc[:, feature_columns] = feature_panel
    panel["label"] = panel["label"].fillna(0)

    x_ = panel[feature_columns].to_numpy(dtype=np.float32).reshape(
        expected_stocks, expected_dates, len(feature_columns)
    ).transpose(1, 0, 2)
    y = panel["label"].to_numpy(dtype=np.int64).reshape(
        expected_stocks, expected_dates
    ).transpose(1, 0)

    if not np.isfinite(x_).all():
        raise ValueError(f"{period} contains non-finite feature values after alignment")

    length = expected_dates
    train_index = int(0.8 * length)
    valid_index = int(0.9 * length)
    overlap_start = train_index - rnn_length + 1
    valid_start = valid_index - rnn_length + 1

    x_train, y_train, m_train = x_[:train_index], y[:train_index], valid_mask[:train_index]
    x_eval, y_eval, m_eval = (
        x_[overlap_start:valid_index],
        y[overlap_start:valid_index],
        valid_mask[overlap_start:valid_index],
    )
    x_test, y_test, m_test = (
        x_[valid_start:],
        y[valid_start:],
        valid_mask[valid_start:],
    )

    return (
        x_train, x_eval, x_test,
        y_train, y_eval, y_test,
        m_train, m_eval, m_test,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

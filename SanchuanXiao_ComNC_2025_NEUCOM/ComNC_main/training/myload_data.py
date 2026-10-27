import pandas as pd
import csv
import numpy as np
import os
from torch import nn
import torch
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('COMNC_DATA_DIR', str(PROJECT_ROOT / 'CAS1')))
LEGACY_DATA_DIR = PROJECT_ROOT / 'data'

def load_data(): #加载一个CSV文件中的数据并对其进行处理，最后返回EOD（End of Day）数据和相应的标签（ground truth）
    code_num = int(os.environ.get('COMNC_CODE_NUM', '173'))
    fts = 9
    f = DATA_DIR / 'data.csv'
    df = pd.read_csv(f, header=None) #如果不使用 header=None（默认行为），Pandas 会把 CSV 文件的第一行自动识别为列名。
    data = df.iloc[:, 0:-1].values #从df中提取前10列数据，使用iloc来基于位置索引数据（即所有行:，第0到9列0:10），并将其转换为NumPy数组values，赋值给data。
    eod_data = data.reshape(-1, code_num, fts) #将data重新调整形状为三维数组，形状为(code_num, trade_day, fts)(758, 1702, 10)，并赋值给eod_data。这里
    data_label = df.iloc[:, -1].values
    ground_truth = data_label.reshape(code_num, -1)

    return eod_data, ground_truth

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_industry_adj():
    adj = np.mat(pd.read_csv(LEGACY_DATA_DIR / 'industry_adj(758).csv', header=None))
    return adj

def get_hyper_adj():
    hyper_edge=[]
    industry_adj = np.mat(pd.read_csv(DATA_DIR / 'industry_hyper.csv', header=None))
    region_adj = np.mat(pd.read_csv(DATA_DIR / 'region_hyper.csv', header=None))
    hyper_edge.append(industry_adj)
    hyper_edge.append(region_adj)
    return hyper_edge

def get_heter_adj():
    heter_edge=[]
    f_e_adj = np.mat(pd.read_csv(DATA_DIR / 'firm_executives.csv'))
    #e_e_adj = np.mat(pd.read_csv(open(r'../mydata/executives_executives.csv')))
    f_s_adj = np.mat(pd.read_csv(DATA_DIR / 'firm_gudong.csv'))
    heter_edge.append(f_e_adj)
    #heter_edge.append(e_e_adj)
    heter_edge.append(f_s_adj)
    return heter_edge


def get_matrix():
    matrix = []
    e_f_matrix = np.mat(pd.read_csv(DATA_DIR / 'matrix.csv', header=None))
    s_f_matrix = np.mat(pd.read_csv(DATA_DIR / 'matrix2.csv', header=None))
    matrix.append(e_f_matrix)
    matrix.append(s_f_matrix)

    return matrix

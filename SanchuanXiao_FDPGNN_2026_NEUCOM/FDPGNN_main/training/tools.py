import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
#from training.load_data import load_EOD_data, get_batch, get_fund_adj_H, get_industry_adj
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def cal_performance(pred, gold, smoothing=False):

    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    percision = metrics.precision_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    recall = metrics.recall_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    f1_score = metrics.f1_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='weighted')

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct, percision, recall, f1_score

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = 2

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss


import random
def train_epoch(model,training_data,train_y,optimizer, device, criterion,args, train_mask=None): #smoothing: 标签平滑的参数，用于改进模型的泛化能力。
    ''' Epoch operation in training phase'''
    model.train()
    seq_len = len(training_data)  # 计算训练数据集 x_train 的长度。
    train_seq = list(range(seq_len))[
                args.length:]  # rnn_length=20 #创建一个列表 train_seq，其中包含从 rnn_length 开始到训练数据集的末尾的所有索引。这个列表将用于遍历训练数据集
    random.shuffle(train_seq)  # 将 train_seq 列表中的索引随机打乱，以确保每次训练都是随机的。通过打乱样本顺序，我们可以更好地积累损失，因为模型在每个批次中看到的样本是随机选择的，而不是固定的。
    train_y = train_y.long()
    n_count = 0
    total_loss = 0
    total_loss_count = 0
    #train_y = train_y.to(device)
    #train_y = torch.from_numpy(train_y).float().to(device)  # 转换为张量并移动到设备

    batch_train = args.batch_size  # 设置训练批次的大小，这表示在每个批次中会处理多少个数据点。
    #print(train_seq)
    # 计算滞后值矩阵和lead-lag时间差矩阵
    #training_data1 = apply_batch_norm(training_data,training_data.shape[2])
    #lag_matrix, lead_lag_diff = compute_lead_lag(training_data1, 20)

    # 将结果转换为 PyTorch 张量并移动到 GPU
    #lag_matrix = torch.tensor(lag_matrix).to('cuda')
    #lead_lag_diff = torch.tensor(lead_lag_diff).to('cuda')
    k=0
    for i in train_seq: #对 train_seq 中的每个索引进行迭代，即对训练数据集进行迭代。train_seq相当于一个迭代器     此时train_seq的长度为416-20=396
        X_train= training_data[i - args.length + 1: i + 1].to(device)
        mean = X_train.mean(dim=(0, 1), keepdim=True)  # 计算均值
        std = X_train.std(dim=(0, 1), keepdim=True)  # 计算标准差

        # 标准化：每个公司各自减去自己的均值和标准差
        X_train = (X_train - mean) / (std + 1e-6)

        if torch.isnan(X_train).any():
            print("X_train 中存在 NaN 值！")
        pred_1 = model(X_train)
        # pred, p = model(X_train.float()) #输入Eod  torch.Size([10, 163, 8])
        # Step 3: Compute state probability regularization (L_SP)
        # s_i: [N, D] -> variance and mean over all elements

        loss_values = criterion(pred_1, train_y[i])
        if loss_values.dim() == 0:
            loss_1 = loss_values
        elif train_mask is None:
            loss_1 = loss_values.mean()
        else:
            active = train_mask[i].bool()
            if not active.any():
                continue
            loss_1 = loss_values[active].mean()
        count = (pred_1[:, 0] > pred_1[:, 1]).sum().item()
        #print("train",count)  # 1 的总数


        loss = loss_1
        k = k + 1
        #print(i)
        #print('第', k, '轮训练', 'sumoutput=', loss)
        #print('第', i, '轮训练', 'sumoutput=', loss)

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print_gradients(model, i)
                    #print(i)
                    print(f"梯度中存在 NaN 或 Inf 值: {name}")
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        #print_gradients(model,i)
        if total_loss_count % batch_train == batch_train-1 :
            #print((pred[:, 0] > pred[:, 1]).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train-1  :  # 如果最后一个批次的大小不等于 batch_train，则需要额外执行一次优化。

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        #optimizer.step_and_update_lr()
    return total_loss / total_loss_count





# def train_epoch(model, training_data, train_y, optimizer, device, criterion, args):
#     model.train()
#     seq_len = len(training_data)
#     train_seq = list(range(seq_len))[args.length:]
#     random.shuffle(train_seq)
#
#     total_loss = 0
#     total_loss_count = 0
#     batch_train = args.batch_size
#     k = 0
#
#     for i in train_seq:
#         X_train = training_data[i - args.length + 1: i + 1].to(device)
#         mean = X_train.mean(dim=(0, 1), keepdim=True)
#         std = X_train.std(dim=(0, 1), keepdim=True)
#         X_train = (X_train - mean) / (std + 1e-4)
#
#         if torch.isnan(X_train).any() or torch.isinf(X_train).any():
#             print(f"NaN or Inf in X_train at index {i}, batch {k}")
#             continue
#         print(f"Batch {k}, X_train min: {X_train.min().item()}, max: {X_train.max().item()}")
#
#         if torch.isnan(train_y[i]).any() or torch.isinf(train_y[i]).any():
#             print(f"NaN or Inf in train_y at index {i}, batch {k}")
#             continue
#         print(f"Batch {k}, train_y min: {train_y[i].min().item()}, max: {train_y[i].max().item()}")
#
#         pred_1, s_n = model(X_train)
#         if torch.isnan(pred_1).any() or torch.isinf(pred_1).any():
#             print(f"NaN or Inf in pred_1 at index {i}, batch {k}")
#             continue
#         if torch.isnan(s_n).any() or torch.isinf(s_n).any():
#             print(f"NaN or Inf in s_n at index {i}, batch {k}")
#             continue
#         print(f"Batch {k}, pred_1 min: {pred_1.min().item()}, max: {pred_1.max().item()}")
#
#         loss_1 = criterion(pred_1, train_y[i])
#         if torch.isnan(loss_1).any() or torch.isinf(loss_1).any():
#             print(f"NaN or Inf in loss_1 at index {i}, batch {k}")
#             continue
#
#         count = (pred_1[:, 0] > pred_1[:, 1]).sum().item()
#         print(f"train batch {k}, count: {count}, Loss: {loss_1.item()}")
#
#
#
#         torch.autograd.set_detect_anomaly(True)
#         loss_1.backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
#                 print(f"NaN or Inf in gradient {name} at batch {k}")
#
#
#
#         total_loss += loss_1.item()
#         total_loss_count += 1
#         k += 1
#
#         if total_loss_count % batch_train == batch_train - 1:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
#
#     if total_loss_count % batch_train != 0:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)
#
#     return total_loss / total_loss_count if total_loss_count > 0 else 0



def print_gradients(model,i):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - num: {i},Mean: {param.grad.mean()}, Std: {param.grad.std()}, Max: {param.grad.max()}")


def evaluate_epoch(model, x_eval, y_eval, optimizer, device, args, eval_mask=None):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[args.length:]
    preds = []
    trues = []
    y_eval = y_eval.long().to(device)  # 确保 y_eval 是 float32 类型



    # 计算滞后值矩阵和lead-lag时间差矩阵
    #training_data1 = apply_batch_norm(x_eval, x_eval.shape[2])
    #lag_matrix, lead_lag_diff = compute_lead_lag(training_data1, 20)

    # 将结果转换为 PyTorch 张量并移动到 GPU
    #lag_matrix = torch.tensor(lag_matrix).to('cuda')
    #lead_lag_diff = torch.tensor(lead_lag_diff).to('cuda')

    for i in seq:
        X = x_eval[i - args.length + 1: i + 1].to(device)  # 确保 X 是 float32 类型
        mean = X.mean(dim=(0, 1), keepdim=True)  # 计算均值
        std = X.std(dim=(0, 1), keepdim=True)  # 计算标准差
        #标准化：每个公司各自减去自己的均值和标准差
        X = (X - mean) / (std + 1e-6)
        output = model(X)
        #output = model(X.float())
        #print('第', i, '轮训练', 'sumoutput=', sum(output))
        #print(sum(output))
        output = output.detach().cpu()
        true = y_eval[i].cpu()
        if eval_mask is not None:
            active = eval_mask[i].bool().cpu()
            output = output[active]
            true = true[active]
        if output.shape[0] == 0:
            continue
        preds.append(np.exp(output.numpy()))
        trues.append(true.numpy())
        count = (output[:, 0] > output[:, 1]).sum().item()
        #print("test",count)  # 1 的总数

    acc, auc = metrics(trues, preds)
    #print(trues[0], preds[0])
    return acc, auc


# def metrics(trues, preds):
#     trues = np.concatenate(trues,-1)
#     preds = np.concatenate(preds,0)
#     acc = sum(preds.argmax(-1) == trues) / len(trues)
#     auc = roc_auc_score(trues,preds[:,1]) #表示模型的二元分类性能，通常用于处理正类和负类之间的不平衡。这里假设preds的第二列包含了正类别的预测概率。
#     return acc, auc

def metrics(trues, preds):
    trues = np.concatenate(trues, -1)
    preds = np.concatenate(preds, 0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues, preds[:, 1])  # Assumes preds[:, 1] contains the probability for the positive class
    mcc = matthews_corrcoef(trues, preds.argmax(-1))  # Calculates MCC based on true labels and predicted classes
    return acc, auc

from torch import nn
import math
def reset_parameters(named_parameters):
    # 函数的目的是重置神经网络模型的参数。它遍历模型的命名参数（named_parameters），并根据参数的形状（大小）来初始化参数。具体来说：
    for i in named_parameters():
        #如果参数的形状是一维的（长度为1），则使用均匀分布初始化，范围从 -std 到 std，其中 std 是通过计算得出的标准差。
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        #如果参数的形状是二维或更高维的，则使用 Xavier（也称为Glorot）初始化，这是一种常用的权重初始化方法，旨在有效地初始化网络权重以促进训练收敛。
        else:
            nn.init.xavier_normal_(i[1])


def apply_batch_norm(src_seq1, num_features=9):
    # Initialize BatchNorm1d with given number of features
    batch_norm = nn.BatchNorm1d(num_features=num_features).to(src_seq1.device)

    # Reshape src_seq1 for normalization
    X = src_seq1.reshape(-1, num_features)

    # Apply batch normalization
    X_normalized = batch_norm(X)

    # Reshape back to original shape
    src_seq = X_normalized.view(src_seq1.shape[0], src_seq1.shape[1], -1)

    return src_seq

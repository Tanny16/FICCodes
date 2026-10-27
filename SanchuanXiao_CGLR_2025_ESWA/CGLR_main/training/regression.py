import os
import glob
import time
import random
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, R2_score_calculate, IC_ICIR_score_calculate ,compute_lead_lag
from sklearn import metrics
from CGLR.models import CGLR
import logging

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--data_file', type=str, default=str(PROJECT_ROOT / 'data' / 'CAS_B.csv'))
parser.add_argument('--code_num', type=int, default=305)
parser.add_argument('--run_name', type=str, default='CAS_B_manual')
parser.add_argument('--data_type', type=str, default='raw')
parser.add_argument('--run_dir', type=str, default=str(PROJECT_ROOT / 'manual_runs' / 'CAS_B'))
parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
parser.add_argument('--patience', type=int, default=20, help='Patience.')
parser.add_argument('--min_epochs', type=int, default=30, help='Minimum epochs before early stopping.')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient Accumulation.')
parser.add_argument('-length', type=int, default=15,
                        help='length of historical sequence for feature')

parser.add_argument('-feature', type=int, default=9, help='input_size')
parser.add_argument('-n_class', type=int, default=1, help='output_size')
parser.add_argument('-epoch', type=int, default=300)
parser.add_argument('-batch_size', type=int, default=32)

parser.add_argument('--rnn_unit', type=int, default=50, help='Number of rnn hidden units.')
parser.add_argument('-d_model', type=int, default=16)


parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('-dropout', type=float, default=0.20
                        )
parser.add_argument('-proj_share_weight', default='True')

parser.add_argument('-log', default=str(PROJECT_ROOT / 'manual_runs' / 'CAS_B'))
parser.add_argument('-save_model', default=str(PROJECT_ROOT / 'manual_runs' / 'CAS_B'))
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', default='True')
parser.add_argument('-n_warmup_steps', type=int, default=4000)

parser.add_argument('--weight-constraint', type=float, default='5e-4',
                        help='L2 weight constraint')

parser.add_argument('--clip', type=float, default='0.5',
                        help='rnn clip')

parser.add_argument('--lr', type=float, default='5e-4',  #5e-4
                        help='Learning rate ')

parser.add_argument('-steps', default=1,
                        help='steps to make prediction')

parser.add_argument('--save', type=bool, default=True,
                        help='save model')

parser.add_argument('--soft-training', type=int, default='0',
                        help='0 False. 1 True')

parser.add_argument('--use_hidden_rel', type=int, default='1',
                        help='use hidden relationship or not')
parser.add_argument('--max_step', type=int, default='10',
                        help='Lead-lag max step')

parser.add_argument('--num_path', type=int, default='10',
                        help='Number of message passing path')

parser.add_argument('--top_k', type=int, default='6',
                        help='Top-k of message passing path')

args = parser.parse_args()
run_dir = Path(args.run_dir) if args.run_dir else Path.home() / 'CGLR_runs' / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(run_dir / 'training.log'), level=logging.DEBUG)
args.cuda = not args.no_cuda and torch.cuda.is_available()
# Set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')
# load data
# features shape: [Days, Firms, Dimension of features]
features, labels = load_data(args.data_file, args.code_num or None, args.feature)
print('[DATA]', str(Path(args.data_file).resolve()))
print('[SHAPE]', tuple(features.shape), tuple(labels.shape))
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
features = features.to(device)
labels = labels.to(device)
train_end_time = int(len(features)*0.7)
val_end_time = int(len(features)*0.9)
feature_mean = features[:train_end_time].mean(dim=(0, 1), keepdim=True)
feature_std = features[:train_end_time].std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
features = (features - feature_mean) / feature_std
labels = torch.transpose(labels, 1, 0)
labels = labels*100


# Model and optimizer
# Split data
rnn_length = args.length
X_train, X_eval, X_test = features[:train_end_time], features[train_end_time - rnn_length + 1:val_end_time], features[val_end_time - rnn_length + 1:]
y_train, y_eval, y_test = labels[:train_end_time], labels[train_end_time - rnn_length + 1:val_end_time], labels[val_end_time - rnn_length + 1:]

args.d_word_vec = args.d_model #16
model = CGLR(
        num_stock =  X_train.shape[1],
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        dropout=args.dropout,
        use_hidden_rel=args.use_hidden_rel,
        window_size = args.length,
        max_step =  args.max_step,
        num_path = args.num_path,
        top_k = args.top_k
    )

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_constraint)
loss_fun = nn.MSELoss()



model.to(torch.float)
model = model.to(device)


lag_matrix, lead_lag_diff = compute_lead_lag(X_train, args.max_step)
lag_matrix = torch.tensor(lag_matrix, device=device)
lead_lag_diff = torch.tensor(lead_lag_diff, device=device)


lag_matrix2, lead_lag_diff2 = compute_lead_lag(X_eval, args.max_step)
lag_matrix2 = torch.tensor(lag_matrix2, device=device)
lead_lag_diff2 = torch.tensor(lead_lag_diff2, device=device)


lag_matrix3, lead_lag_diff3 = compute_lead_lag(X_test, args.max_step)
lag_matrix3 = torch.tensor(lag_matrix3, device=device)
lead_lag_diff3 = torch.tensor(lead_lag_diff3, device=device)





def train(epoch, lag_matrix, lead_lag_diff, lag_matrix2, lead_lag_diff2):
    t = time.time()
    model.train()
    train_seq = list(range(len(X_train) + 1))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    count_train = 0

    for i in train_seq:
        output = model(X_train[i - rnn_length: i], lead_lag_diff, lag_matrix)
        # regression loss
        reg_loss = loss_fun(output, y_train[i - 1].reshape(-1, 1))
        # total loss
        loss = reg_loss
        total_loss += loss.item()
        count_train += 1
        loss = loss / args.accumulation_steps
        loss.backward()
        if (count_train % args.accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
    if (count_train % args.accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    phase_pred_val = []
    phase_label_val = []
    eval_seq = list(range(len(X_eval) + 1))[rnn_length:]
    for i in eval_seq:
        with torch.no_grad():
            output = model(X_eval[i - rnn_length: i], lead_lag_diff2, lag_matrix2)
        phase_pred_val.extend(output.detach().cpu().numpy().reshape(-1))
        phase_label_val.extend(y_eval[i - 1].detach().cpu().numpy())

    mse_val = metrics.mean_squared_error(np.array(phase_label_val), np.array(phase_pred_val))
    r2_val = R2_score_calculate(np.array(phase_label_val), np.array(phase_pred_val))
    rank_ic_val, rank_ic_ir_val = IC_ICIR_score_calculate(phase_label_val, phase_pred_val, len(eval_seq))

    return total_loss / count_train, mse_val, r2_val, rank_ic_val, rank_ic_ir_val


def compute_test(lag_matrix3, lead_lag_diff3):
    model.eval()
    phase_pred_test = []
    phase_label_test = []
    test_seq = list(range(len(X_test) + 1))[rnn_length:]
    for i in test_seq:
        with torch.no_grad():
            output = model(X_test[i - rnn_length: i], lead_lag_diff3, lag_matrix3)
        phase_pred_test.extend(output.detach().cpu().numpy().reshape(-1))
        phase_label_test.extend(y_test[i - 1].detach().cpu().numpy())

    mse_test = metrics.mean_squared_error(np.array(phase_label_test), np.array(phase_pred_test))
    r2_test = R2_score_calculate(np.array(phase_label_test), np.array(phase_pred_test))
    rank_ic_test, rank_ic_ir_test = IC_ICIR_score_calculate(phase_label_test, phase_pred_test, len(test_seq))

    return mse_test, r2_test, rank_ic_test, rank_ic_ir_test


# Training model
t_total = time.time()
bad_counter = 0
best_val_r2 = -float('inf')
best_val_epoch = 0
best_test_r2 = -float('inf')
best_test_epoch = 0
best_test_record = None
checkpoint_dir = run_dir / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_val_model_path = checkpoint_dir / 'best_validation_model.pkl'
best_test_model_path = checkpoint_dir / 'best_test_model.pkl'
last_model_path = checkpoint_dir / 'last_model.pkl'
metrics_path = run_dir / 'metrics.csv'
metric_fields = [
    'epoch', 'train_loss', 'loss_val', 'r2_val', 'rank_ic_val',
    'rank_icir_val', 'loss_test', 'r2_test', 'rank_ic_test',
    'rank_icir_test', 'epoch_time'
]

with metrics_path.open('w', newline='', encoding='utf-8-sig') as metrics_file:
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
    metrics_writer.writeheader()

    for epoch in range(args.epoch):
        train_loss, mse_val, r2_val, rank_ic_val, rank_ic_ir_val = train(
            epoch, lag_matrix, lead_lag_diff, lag_matrix2, lead_lag_diff2
        )
        mse_test, r2_test, rank_ic_test, rank_ic_ir_test = compute_test(
            lag_matrix3, lead_lag_diff3
        )
        epoch_time = time.time() - t_total
        epoch_number = epoch + 1
        print('Epoch: {:04d}'.format(epoch_number),
              'train_loss: {:.4f}'.format(train_loss),
              'loss_val: {:.4f}'.format(mse_val),
              'R2_val: {:.4f}'.format(r2_val),
              'Rank_IC_val: {:.4f}'.format(rank_ic_val),
              'Rank_ICIR_val: {:.4f}'.format(rank_ic_ir_val),
              'loss_test: {:.4f}'.format(mse_test),
              'R2_test: {:.4f}'.format(r2_test),
              'Rank_IC_test: {:.4f}'.format(rank_ic_test),
              'Rank_ICIR_test: {:.4f}'.format(rank_ic_ir_test),
              'epoch_time: {:.4f}s'.format(epoch_time), flush=True)

        metrics_writer.writerow({
            'epoch': epoch_number,
            'train_loss': train_loss,
            'loss_val': mse_val,
            'r2_val': r2_val,
            'rank_ic_val': rank_ic_val,
            'rank_icir_val': rank_ic_ir_val,
            'loss_test': mse_test,
            'r2_test': r2_test,
            'rank_ic_test': rank_ic_test,
            'rank_icir_test': rank_ic_ir_test,
            'epoch_time': epoch_time,
        })
        metrics_file.flush()

        if np.isfinite(r2_test) and r2_test > best_test_r2:
            best_test_r2 = r2_test
            best_test_epoch = epoch_number
            best_test_record = {
                'epoch': epoch_number,
                'loss_test': mse_test,
                'r2_test': r2_test,
                'rank_ic_test': rank_ic_test,
                'rank_icir_test': rank_ic_ir_test,
                'r2_val': r2_val,
                'rank_ic_val': rank_ic_val,
                'rank_icir_val': rank_ic_ir_val,
            }
            best_test_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(best_test_model_path))

        if np.isfinite(r2_val) and r2_val > best_val_r2:
            best_val_r2 = r2_val
            best_val_epoch = epoch_number
            bad_counter = 0
            best_val_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(best_val_model_path))
        elif epoch_number > args.min_epochs:
            bad_counter += 1

        if epoch_number > args.min_epochs and bad_counter >= args.patience:
            break

last_model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), str(last_model_path))
print("Optimization Finished!", flush=True)
print("Total time elapsed: {:.4f}s".format(time.time() - t_total), flush=True)

final_mse_test, final_r2_test, final_rank_ic_test, final_rank_ic_ir_test = compute_test(
    lag_matrix3, lead_lag_diff3
)

best_test_evaluated = None
if best_test_model_path.is_file():
    model.load_state_dict(torch.load(str(best_test_model_path), map_location=device))
    best_test_evaluated = compute_test(lag_matrix3, lead_lag_diff3)

print('Best Test Results:',
      'epoch: {:04d}'.format(best_test_epoch),
      'loss_test: {:.4f}'.format(best_test_evaluated[0] if best_test_evaluated else best_test_record['loss_test']),
      'R2_test: {:.4f}'.format(best_test_evaluated[1] if best_test_evaluated else best_test_record['r2_test']),
      'Rank_IC_test: {:.4f}'.format(best_test_evaluated[2] if best_test_evaluated else best_test_record['rank_ic_test']),
      'Rank_ICIR_test: {:.4f}'.format(best_test_evaluated[3] if best_test_evaluated else best_test_record['rank_icir_test']), flush=True)
print('Final Test Results:',
      'loss_test: {:.4f}'.format(final_mse_test),
      'R2_test: {:.4f}'.format(final_r2_test),
      'Rank_IC_test: {:.4f}'.format(final_rank_ic_test),
      'Rank_ICIR_test: {:.4f}'.format(final_rank_ic_ir_test), flush=True)

summary = {
    'run_name': args.run_name,
    'data_type': args.data_type,
    'data_file': str(Path(args.data_file).resolve()),
    'code_num': int(features.shape[1]),
    'feature_count': int(features.shape[2]),
    'time_count': int(features.shape[0]),
    'epochs_completed': int(epoch_number),
    'best_test_epoch': int(best_test_epoch),
    'best_test_r2': float(best_test_evaluated[1] if best_test_evaluated else best_test_record['r2_test']),
    'best_test_loss': float(best_test_evaluated[0] if best_test_evaluated else best_test_record['loss_test']),
    'best_test_rank_ic': float(best_test_evaluated[2] if best_test_evaluated else best_test_record['rank_ic_test']),
    'best_test_rank_icir': float(best_test_evaluated[3] if best_test_evaluated else best_test_record['rank_icir_test']),
    'val_r2_at_best_test': float(best_test_record['r2_val']),
    'rank_ic_val_at_best_test': float(best_test_record['rank_ic_val']),
    'rank_icir_val_at_best_test': float(best_test_record['rank_icir_val']),
    'best_validation_epoch': int(best_val_epoch),
    'best_validation_r2': float(best_val_r2),
    'final_test_r2': float(final_r2_test),
    'final_test_rank_ic': float(final_rank_ic_test),
    'final_test_rank_icir': float(final_rank_ic_ir_test),
    'best_test_model': str(best_test_model_path),
    'best_validation_model': str(best_val_model_path),
    'last_model': str(last_model_path),
    'metrics_file': str(metrics_path),
}
with (run_dir / 'summary.json').open('w', encoding='utf-8') as summary_file:
    json.dump(summary, summary_file, ensure_ascii=False, indent=2)

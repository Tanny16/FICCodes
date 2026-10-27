import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import random
from training.myload_data import load_data,get_hyper_adj,get_heter_adj,get_matrix
import torch
from HGTAN.mymodels import myHGTAN
from training.myload_data import *
#from training import *
from HGTAN.Optim import ScheduledOptim
from training.mytools import train_epoch,evaluate_epoch
import torch.optim as optim
import os

import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
def main():
    ''' Main function '''
    set_seed(2020)
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', type=int, default=10,
                        help='length of historical sequence for feature')
    parser.add_argument('-train_index', type=int, default=390)
    parser.add_argument('-valid_index', type=int, default=439)
    parser.add_argument('-feature', default=9, help='input_size') #代表我输入数据的特征维度，假如要经过一个线性层则需要从8变换到别的维度
    parser.add_argument('-n_class', default=2, help='output_size')
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('--rnn_unit', type=int, default=60, help='Number of hidden units.')
    parser.add_argument('-d_model', type=int, default=16)

    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=35, help='Number of hidden units.')
    parser.add_argument('-dropout', type=float, default=0.2
                        )
    parser.add_argument('-proj_share_weight', default='True')

    parser.add_argument('-log', default='../10_days/lstm+trans+HGAT3_5_valid1')
    parser.add_argument('-save_model', default='../10_days/lstm+HGAT3_5_valid1')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    # 权重的L2约束。
    parser.add_argument('--weight-constraint', type=float, default='0',
                        help='L2 weight constraint')
    # 梯度裁剪（gradient clipping）。
    parser.add_argument('--clip', type=float, default='0.50',
                        help='rnn clip')
    # 学习率。
    parser.add_argument('--lr', type=float, default='5e-4',  #5e-4
                        help='Learning rate ')            

    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')
    # --save: 是否保存模型。
    parser.add_argument('--save', type=bool, default=True,
                        help='save model')
    # --soft-training: 软训练开关。
    parser.add_argument('--soft-training', type=int, default='0',
                        help='0 False. 1 True')

    parser.add_argument('--use_hidden_rel', type=int, default='1',
                        help='use hidden relationship or not')
    parser.add_argument('--use_interaction', type=int, default='1',
                        help='use interaction or not')

    args = parser.parse_args()

    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model #16

    eod_data, ground_truth = load_data() #(488, 173, 9)
    hyper_adj = get_hyper_adj()
    heter_adj = get_heter_adj()
    matrix = get_matrix()
    #eod_data = np.transpose(data, (1, 0, 2)) #变换维度的目的是为了做时间的切片
    ground_truth = np.transpose(ground_truth, (1, 0))

    train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = prepare_mydataloaders(eod_data, ground_truth, args.length,args)

    # ========= Preparing Model =========#
    print(args)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = myHGTAN(
        num_stock =  train_eod.shape[1],
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        n_head=args.n_head,
        dropout=args.dropout,
        use_hidden_rel=args.use_hidden_rel,
        use_interaction= args.use_interaction

    ).to(device)
    '''
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)
        '''
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    #train(model, train_eod,train_gt,valid_eod,valid_gt,hyper_adj,heter_adj,e_f_matrix, optimizer, device, args) #train_gt.shape [200, 163] train_eod [200, 163, 8]
    best_model_file = 0

    # seed=2024
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = float('-inf')
    selection_metric = os.environ.get('COMNC_SELECTION_METRIC', 'val_auc')
    eval_best_str = ""
    MAX_EPOCH=100
    criterion = torch.nn.NLLLoss()
    while epoch < MAX_EPOCH:
        start_time = time.time()
        train_loss = train_epoch(model, train_eod,train_gt,hyper_adj,heter_adj,matrix, optimizer, device,criterion, args)
        eval_acc, eval_auc = evaluate_epoch(model, valid_eod, valid_gt, hyper_adj,heter_adj,matrix, optimizer, device, args)
        test_acc, test_auc = evaluate_epoch(model, test_eod, test_gt, hyper_adj,heter_adj,matrix, optimizer, device, args)
        # 计算 epoch 的运行时间
        epoch_time = time.time() - start_time
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}, time{:.2f}s".format(
            epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc, epoch_time
        )
        print(eval_str)

        #if eval_auc > eval_epoch_best and epoch>20:
        current_metric = eval_auc if selection_metric == 'val_auc' else test_auc
        if current_metric > eval_epoch_best:
            #eval_epoch_best = eval_auc
            eval_epoch_best = test_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch, train_loss, eval_auc,eval_acc, test_auc, test_acc)
            wait_epoch = 0

            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./eval_auc{:.3f}_acc{:.3f}_test_auc{:.3f}_acc{:.3f}.pkl".format(eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 20:
            print("saved_model_result:",eval_best_str)
            break
        epoch += 1






import torch.utils.data as Data


def prepare_mydataloaders(eod_data, gt_data,rnn_length ,args):  #eod_data: EOD数据（形状为(code_num, trade_day, fts)）。gt_data: 对应的标签数据（形状为(code_num, trade_day)）。args: 参数对象，包含训练所需的超参数，如length, train_index, valid_index和batch_size等。
    # ========= Preparing DataLoader =========#
    EOD, GT = eod_data, gt_data

    #EOD有1691个交易日的，
    train_end = int(os.environ.get('COMNC_TRAIN_END', str(args.train_index)))
    valid_end = int(os.environ.get('COMNC_VALID_END', str(args.valid_index)))
    train_eod, train_gt = EOD[:train_end], GT[:train_end]
    valid_eod, valid_gt = EOD[train_end- rnn_length+ 1:valid_end], GT[train_end- rnn_length+ 1:valid_end]
    test_eod, test_gt = EOD[valid_end- rnn_length+ 1:], GT[valid_end- rnn_length+ 1:]

    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod) #train_eod, valid_eod, test_eod: 转换为浮点型张量FloatTensor，因为EOD数据通常为连续数值。
    train_gt, valid_gt, test_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt), torch.LongTensor(test_gt) #train_gt, valid_gt, test_gt: 转换为长整型张量LongTensor，通常标签是整数分类值。




    return train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt

if __name__ == '__main__':
    main()

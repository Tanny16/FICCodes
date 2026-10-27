import argparse
import numpy as np
import random
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.load_data import *
import torch
from WaveHGRN.models import WaveHGRN
from training.load_data import *
from training.tools import train_epoch,evaluate_epoch
import torch.optim as optim
import os
import pickle


import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
def main():
    torch.autograd.set_detect_anomaly(True)
    ''' Main function '''
    set_seed(223) #199
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', type=int, default=96,
                        help='length of historical sequence for feature')
    parser.add_argument('-feature', default=9, help='input_size')
    parser.add_argument('-n_class', type=int, default=2, help='output_size')
    parser.add_argument('-epoch', type=int, default=150)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('--tem_dim', type=int, default=64, help='Number of Temporal Feature Dimension.')



    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=96, help='Number of hidden units.')
    parser.add_argument('-hyper_edge', type=int, default=20, help='Number of hyper-edges.')
    parser.add_argument('-dropout', type=float, default=0.35
                        )
    parser.add_argument('--concentration_weight', type=float, default=0.5)


    parser.add_argument('-log', default='../days/MSDGNN_valid1')
    parser.add_argument('-save_model', default='../days/MSDGNN_valid1')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('--weight-constraint', type=float, default='0',
                        help='L2 weight constraint')

    parser.add_argument('--clip', type=float, default='0.50',
                        help='rnn clip')

    parser.add_argument('--lr', type=float, default='5e-4',  #5e-4
                        help='Learning rate ')            

    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')

    parser.add_argument('--save', type=bool, default=True,
                        help='save model')

    parser.add_argument('--soft-training', type=int, default='0',
                        help='0 False. 1 True')

    parser.add_argument('--scale_num', type=int, default=2,
                        help='number of time series scales')
    parser.add_argument('--path_num', type=int, default='6',
                        help='number of message passing paths')

    parser.add_argument('--mem_dim', type=int, default='96',
                        help='number of message passing paths')
    parser.add_argument('--period', type=str, default='CSI300')


    args = parser.parse_args()

    args.cuda = not args.no_cuda

    device = torch.device('cuda' if args.cuda else 'cpu')
    #train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = load_dataset2(device, args.length)

    train_eod, valid_eod, test_eod, train_gt, valid_gt, test_gt = read_data2('cls', args.period, args.length)
    train_gt = torch.from_numpy(train_gt).to(device)
    valid_gt = torch.from_numpy(valid_gt).to(device)
    test_gt = torch.from_numpy(test_gt).to(device)
    train_eod = torch.from_numpy(train_eod).to(device)
    valid_eod = torch.from_numpy(valid_eod).to(device)
    test_eod = torch.from_numpy(test_eod).to(device)
    train_gt.to(torch.float32)
    valid_gt.to(torch.float32)
    test_gt.to(torch.float32)



    # ========= Preparing Model =========#
    print(args)
    #device = torch.device('cuda' if args.cuda else 'cpu')

    model = WaveHGRN(
        num_stock =  train_eod.shape[1],
        tem_dim = args.tem_dim,   #时序维度F
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        dropout=args.dropout,
        scale_num=args.scale_num,
        hyper_edge = args.hyper_edge,
        path_num= args.path_num,
        window_size = args.length,
        mem_dim = args.mem_dim

   ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    best_model_file = None
    best_test_auc = float('-inf')
    best_str = None

    # seed=2024
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    epoch = 0
    wait_epoch = 0
    MAX_EPOCH = args.epoch
    criterion = torch.nn.NLLLoss()
    while epoch < MAX_EPOCH:
        start_time = time.time()

        train_loss = train_epoch(model, train_eod, train_gt, optimizer, device, criterion, args)
        eval_acc, eval_auc, eval_mcc= evaluate_epoch(model, valid_eod, valid_gt, optimizer, device, args)
        test_acc, test_auc, test_mcc = evaluate_epoch(model, test_eod, test_gt, optimizer, device, args)

        epoch_time = time.time() - start_time
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}, time{:.2f}s".format(
            epoch, train_loss, eval_auc, eval_acc, eval_mcc, test_auc, test_acc, test_mcc, epoch_time
        )
        print(eval_str)


        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}".format(
                epoch, train_loss, eval_auc, eval_acc, eval_mcc, test_auc, test_acc, test_mcc)
            wait_epoch = 0

            if args.save:
                if best_model_file and os.path.exists(best_model_file):
                    os.remove(best_model_file)
                checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{args.period}_test_auc"))
                best_model_file = os.path.join(checkpoint_dir, "test_auc{:.4f}_eval_auc{:.4f}_eval_acc{:.4f}_epoch{}.pkl".format(
                    test_auc, eval_auc, eval_acc, epoch))
                os.makedirs(os.path.dirname(best_model_file), exist_ok=True)
                with open(best_model_file, "wb") as model_file:
                    torch.save(model.state_dict(), model_file)
                print(f"New best test AUC model saved: {best_model_file}")
        else:
            wait_epoch += 1

        # 早停
        if wait_epoch > 200:
            print("Early stopping triggered!")
            print("Best result:", best_str)
            break

        epoch += 1


    if best_model_file:
        print(f"Training finished. Best model: {best_model_file}")
        print("Best performance by test AUC:", best_str)
    else:
        print("No model was saved.")







if __name__ == '__main__':
    main()

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSDGNN.models import MSDGNN
from training.load_data import read_data_cas, set_seed
from training.tools import evaluate_epoch, train_epoch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["CAS_133", "CAS_187", "CAS_232"],
        default="CAS_133",
        help="Chinese A-share dataset",
    )
    parser.add_argument("-length", type=int, default=45)
    parser.add_argument("-feature", type=int, default=9)
    parser.add_argument("-n_class", type=int, default=2)
    parser.add_argument("-epoch", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("--rnn_unit", type=int, default=256)
    parser.add_argument("-d_model", type=int, default=64)
    parser.add_argument("-n_layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("-dropout", type=float, default=0.35)
    parser.add_argument("-proj_share_weight", default="True")
    parser.add_argument("-log", default="../days/MSDGNN_valid1")
    parser.add_argument("-save_model", default="../days/MSDGNN_valid1")
    parser.add_argument("-save_mode", choices=["all", "best"], default="best")
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-label_smoothing", default="True")
    parser.add_argument("-n_warmup_steps", type=int, default=4000)
    parser.add_argument("--weight-constraint", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("-steps", type=int, default=1)
    parser.add_argument("--save", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--soft-training", type=int, default=0)
    parser.add_argument("--scale_num", type=int, default=4)
    parser.add_argument("--path_num", type=int, default=6)
    parser.add_argument("--mem_dim", type=int, default=96)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(209)
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model
    device = torch.device("cuda" if args.cuda else "cpu")

    (
        train_eod,
        valid_eod,
        test_eod,
        train_gt,
        valid_gt,
        test_gt,
        train_mask,
        valid_mask,
        test_mask,
    ) = read_data_cas(args.dataset, args.length)

    train_gt = torch.from_numpy(train_gt).to(device)
    valid_gt = torch.from_numpy(valid_gt).to(device)
    test_gt = torch.from_numpy(test_gt).to(device)
    train_eod = torch.from_numpy(train_eod).to(device)
    valid_eod = torch.from_numpy(valid_eod).to(device)
    test_eod = torch.from_numpy(test_eod).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    valid_mask = torch.from_numpy(valid_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)

    model = MSDGNN(
        num_stock=train_eod.shape[1],
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        dropout=args.dropout,
        scale_num=args.scale_num,
        path_num=args.path_num,
        window_size=args.length,
        mem_dim=args.mem_dim,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_constraint
    )
    criterion = torch.nn.NLLLoss(reduction="none")
    best_model_file = None
    best_test_auc = -float("inf")
    eval_best_str = None
    wait_epoch = 0
    patience = 50

    for epoch in range(args.epoch):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_eod, train_gt, optimizer, device, criterion, args, train_mask
        )
        eval_acc, eval_auc = evaluate_epoch(
            model, valid_eod, valid_gt, optimizer, device, args, valid_mask
        )
        test_acc, test_auc = evaluate_epoch(
            model, test_eod, test_gt, optimizer, device, args, test_mask
        )
        epoch_time = time.time() - start_time
        print(
            "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, "
            "test_auc{:.4f}, test_acc{:.4f}, time{:.2f}s".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc, epoch_time
            )
        )

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            eval_best_str = (
                "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, "
                "test_auc{:.4f}, test_acc{:.4f}"
            ).format(epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0

            if args.save:
                if best_model_file and os.path.exists(best_model_file):
                    os.remove(best_model_file)
                best_model_file = (
                    "./eval_auc{:.3f}_acc{:.3f}_test_auc{:.3f}_acc{:.3f}.pkl"
                ).format(eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch >= patience:
            break

    print("saved_model_result:", eval_best_str)
    print("Best model file:", best_model_file)


if __name__ == "__main__":
    main()

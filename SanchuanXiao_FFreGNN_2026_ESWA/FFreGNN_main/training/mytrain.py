import argparse
import os
import time

import torch
import torch.optim as optim

from FFreGNN.models import FFreGNN
from training.load_data import read_data2, set_seed
from training.tools import evaluate_epoch, train_epoch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-length", "--length", type=int, default=60,
        help="length of historical sequence for feature",
    )
    parser.add_argument("-feature", type=int, default=9)
    parser.add_argument("-n_class", type=int, default=2)
    parser.add_argument("-epoch", "--epoch", type=int, default=100)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=32)
    parser.add_argument("-d_model", type=int, default=64)
    parser.add_argument("-n_layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("-dropout", type=float, default=0.35)
    parser.add_argument("-proj_share_weight", default="True")
    parser.add_argument("-log", default="../days/FFreGNN_valid1")
    parser.add_argument("-save_model", default="../days/FFreGNN_valid1")
    parser.add_argument("-save_mode", choices=["all", "best"], default="best")
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu",
        help="execution device",
    )
    parser.add_argument("-label_smoothing", default="True")
    parser.add_argument("-n_warmup_steps", type=int, default=4000)
    parser.add_argument("--weight-constraint", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("-steps", type=int, default=1)
    parser.add_argument(
        "--save", type=lambda value: value.lower() == "true", default=True,
    )
    parser.add_argument("--soft-training", type=int, default=0)
    parser.add_argument("--scale_num", type=int, default=3)
    parser.add_argument("--path_num", type=int, default=6)
    parser.add_argument(
        "--dataset", choices=["CAS_A", "CAS_B", "CAS_C"], default="CAS_B",
    )
    parser.add_argument(
        "--selection_metric", choices=["eval_auc", "test_auc"], default="test_auc",
    )
    parser.add_argument("--mem_dim", type=int, default=96)
    return parser


def select_device(args):
    if args.no_cuda:
        return torch.device("cpu")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available on this machine.")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available on this machine.")
    return device


def main():
    args = build_parser().parse_args()
    set_seed(223)
    device = select_device(args)
    args.cuda = device.type == "cuda"
    args.d_word_vec = args.d_model

    (
        train_eod, valid_eod, test_eod,
        train_gt, valid_gt, test_gt,
    ) = read_data2("cls", args.dataset, args.length)
    train_gt = torch.from_numpy(train_gt).to(device)
    valid_gt = torch.from_numpy(valid_gt).to(device)
    test_gt = torch.from_numpy(test_gt).to(device)
    train_eod = torch.from_numpy(train_eod).to(device)
    valid_eod = torch.from_numpy(valid_eod).to(device)
    test_eod = torch.from_numpy(test_eod).to(device)

    print(args)
    model = FFreGNN(
        num_stock=train_eod.shape[1],
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        dropout=args.dropout,
        scale_num=args.scale_num,
        path_num=args.path_num,
        window_size=args.length,
        mem_dim=args.mem_dim,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_constraint
    )
    criterion = torch.nn.NLLLoss()
    best_model_file = None
    best_result_str = None
    best_selection_value = -float("inf")
    wait_epoch = 0

    for epoch in range(args.epoch):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_eod, train_gt, optimizer, device, criterion, args
        )
        eval_acc, eval_auc = evaluate_epoch(
            model, valid_eod, valid_gt, optimizer, device, args
        )
        test_acc, test_auc = evaluate_epoch(
            model, test_eod, test_gt, optimizer, device, args
        )
        elapsed = time.time() - start_time
        print(
            "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, "
            "test_auc{:.4f}, test_acc{:.4f}, time{:.2f}s".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc, elapsed
            )
        )

        selection_value = eval_auc if args.selection_metric == "eval_auc" else test_auc
        if selection_value > best_selection_value:
            best_selection_value = selection_value
            best_result_str = (
                "epoch{}, selected_{}{:.4f}, train_loss{:.4f}, eval_auc{:.4f}, "
                "eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}"
            ).format(
                epoch, args.selection_metric, selection_value,
                train_loss, eval_auc, eval_acc, test_auc, test_acc,
            )
            wait_epoch = 0

            if args.save:
                best_model_file = os.path.join(
                    PROJECT_ROOT,
                    "checkpoints/best_{}_eval_auc{:.4f}_acc{:.4f}_test_auc{:.4f}_acc{:.4f}_epoch{}.pkl",
                ).format(args.selection_metric, eval_auc, eval_acc, test_auc, test_acc, epoch)
                os.makedirs(os.path.dirname(best_model_file), exist_ok=True)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 200:
            break

    print("Best performance:", best_result_str)
    print("Best model:", best_model_file)


if __name__ == "__main__":
    main()

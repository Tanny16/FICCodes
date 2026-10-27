# GFMixer: Decoupled Temporal Gradient and Fourier-Aware Attention for Time Series Forecasting

| 项目 | 信息 |
| --- | --- |
| 作者 | Lin Zhang, Qing Li, Jingmei Zhao |
| 发表 | Proceedings of the ACM Web Conference 2026 (WWW 2026), pp. 7156–7166 |
| 论文 | [DOI: 10.1145/3774904.3792345](https://doi.org/10.1145/3774904.3792345) |
| 代码负责人 | Lin Zhang ([superlin30](https://github.com/superlin30)) |
| 源仓库快照 | [superlin30/GFMixer](https://github.com/superlin30/GFMixer/tree/d07a65454fdfd8ff8c75b16d85d259384bbca320) |

本目录同步 GFMixer 源代码、训练脚本、数据加载和评估工具，保留原 Apache-2.0 `LICENSE`。源快照固定为 `d07a65454fdfd8ff8c75b16d85d259384bbca320`；不包含嵌套 Git 仓库、缓存或数据。原说明与依赖分别保留为 `README.upstream.md` 和 `requirements.upstream.txt`；本文件给出经过核查的归档说明。

## 环境

此次检查使用 Python 3.11.15、PyTorch 2.6.0+cu124、einops 0.8.0、NumPy 1.23.5、pandas 1.5.3、SciPy 1.10.1、scikit-learn 1.2.2、matplotlib 3.7.0 和 tqdm 4.64.1。`requirements.txt` 固定此次检查所用版本；PyTorch 的具体 CUDA 构建应与设备匹配。该环境不是对论文历史训练环境的追溯确认。

```bash
python -m venv .venv
# 激活环境后：
python -m pip install -r requirements.txt
python run.py --help
```

原 README 中的 Python/PyTorch 描述与原 `requirements.txt` 不一致，后者还含彼此不兼容的 Python 3.8 / IPython 8.5 组合；因此原文件只作为来源记录，不建议照原说明混合安装。此次使用现有环境检查，未新建干净环境重新安装全部依赖。

## 数据准备

数据名称、官方来源、切分和放置位置详见 [DATA.md](DATA.md)。项目原仓库提供的预处理数据入口：

- [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)，提取码 `i9iy`。

网盘入口来自源仓库，未记录版本号或有效期，此次未下载核验。将 CSV 直接放在 `dataset/` 下，例如 `dataset/ETTh1.csv`，而不是额外套一层 `ETT/`；实际路径以训练脚本为准。模型从随机初始化训练，基础实验不依赖预训练权重。

## 训练、测试与预测

以下命令均从本论文目录执行；`.sh` 脚本使用 Bash，Windows 可在 WSL 或 Git Bash 中运行。

```bash
bash scripts/GFMixer/etth1.sh
bash scripts/GFMixer/etth2.sh
bash scripts/GFMixer/ettm1.sh
bash scripts/GFMixer/ettm2.sh
bash scripts/GFMixer/weather.sh
bash scripts/GFMixer/electricity.sh
bash scripts/GFMixer/traffic.sh
```

源码实际脚本名为小写 `etth1.sh`。以 ETTh1、输入长度 720、预测长度 96 为例，单组训练命令为：

```bash
python run.py --task_name long_term_forecast --random_seed 2021 --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_720_96 --model GFMixer --data ETTh1 --features M --seq_len 720 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.25 --fc_dropout 0.15 --kernel_list 3 7 11 --period 24 --patch_len 1 --stride 1 --des Exp --pct_start 0.2 --train_epochs 100 --patience 10 --itr 1 --batch_size 128 --learning_rate 0.0001
```

训练入口每轮训练后自动测试。使用已有检查点单独评估时，将完全相同的训练命令中的 `--is_training 1` 改为 `--is_training 0`；模型标识和结构参数必须相同，检查点路径由这些参数生成。训练命令附加 `--do_predict` 可继续调用预测流程。

输出为 `checkpoints/<setting>/checkpoint.pth`、`results/<setting>/pred.npy`、`test_results/<setting>/` 和 `result.txt`，脚本日志在 `logs/`。这些均由目录内 `.gitignore` 排除。原 README 中的结果文本文件名与当前代码不一致，此处以 `exp/exp_main.py` 为准。`exchange.sh` 和 `M4.sh` 也保留在归档内，属于额外脚本，未在此次验证其完整实验。

## 同步时的必要修正与验证边界

源快照在 `run.py` 顶部导入了不存在的 `exp_long_term_forecasting_pir.py`；实验模块还在加载 GFMixer 前强制导入缺失的 `G4P`，`PIR.py` 同样引用缺失模型。本次移除未使用的 PIR 实验导入，并改为按 `--model` 加载所选模型，使已提供的 GFMixer 可以正常启动；不补造缺失模型，不改动 GFMixer、TGB、FAT 的计算逻辑。

此次通过入口检查、所有 Python 源码语法检查、CUDA 上的小型合成输入前向/反向计算，以及不同 Bash 脚本的语法检查。复查命令：

```bash
python examples/smoke_test.py
```

未下载真实数据、未完成论文全量训练或重新核对论文指标。模型内部设备由 CUDA 可用性决定；在有 CUDA 的主机上将模型强行留在 CPU 会产生原始代码的设备不一致错误，此次在 CUDA 上验证通过。不存在的 G4P/PIR 相关资源仍不能使用；M4 未端到端验证。详见 [VALIDATION.md](VALIDATION.md)。

# FFreGNN 论文代码与数据说明

## 1. 论文信息

- 论文全名：Learning to understand financial risk contagion from a frequency-domain graph learning framework
- 作者：Sanchuan Xiao, Jingmei Zhao, Rui Cheng, Yang Wang, Shaofei Shen, Qing Li
- 发表年份：2026
- 期刊：Expert Systems With Applications
- DOI：https://doi.org/10.1016/j.eswa.2026.131619
- 代码负责人英文名：Sanchuan Xiao
- GitHub 用户名：Xsc-Swufe
- 代码仓库：https://github.com/Xsc-Swufe/FFreGNN
- 提交目录名：`SanchuanXiao_FFreGNN_2026_ESWA`

## 2. 环境与依赖

- Python：3.11
- PyTorch：2.0.1
- torchvision：0.15.2
- NumPy：1.24.4
- pandas
- scikit-learn
- PyTorch Geometric
- GPU：NVIDIA 4070 Ti Super

依赖安装：

```powershell
python -m pip install -r requirements.txt
```

## 3. 数据来源与下载

### 中国数据

中国 A 股数据来自 CSMAR，CSI 300 成分股来自 Tushare。论文使用三个数据集：CAS-A 为 245 家公司、1213 个交易日（2020-06-30 至 2025-06-30）；CAS-B 为 198 家公司、1214 个交易日（2018-06-30 至 2023-06-30）；CAS-C 为 125 家公司、1217 个交易日（2016-06-30 至 2021-06-30）。每个数据集包含 9 个输入特征和 1 个趋势标签，训练、验证、测试按时间顺序划分为 80%、10%、10%。

- CSMAR：https://data.csmar.com/
- Tushare：https://tushare.pro/
- 百度网盘：https://pan.baidu.com/s/13gYOAfn6BcK-aP8jxC_wWA?pwd=8miw
- 提取码：8miw

下载后将三个 CSV 文件放置为：

- `FFreGNN_main/CAS_data/data202006dao202506.csv`
- `FFreGNN_main/CAS_data/data201806dao202306.csv`
- `FFreGNN_main/CAS_data/data201606dao202106.csv`

美国数据：来源为 [LeadLagNet GitHub](https://github.com/FICLAB/LeadLagNet)，数据路径为 `FFreGNN_main/SP500/`。

## 4. 目录结构

- `FFreGNN_main/FFreGNN/`：FFreGNN 模型、网络层和优化器
- `FFreGNN_main/training/load_data.py`：中国 CAS 数据读取与时间切分
- `FFreGNN_main/training/tools.py`：训练与评估函数
- `FFreGNN_main/training/mytrain.py`：训练入口
- `FFreGNN_main/main.py`：主运行入口
- `FFreGNN_main/CAS_data/`：中国 CAS 数据放置目录

## 5. 运行说明

在 `FFreGNN_main` 目录执行中国数据：

```powershell
python main.py --dataset CAS_A --length 60 --hidden 96 --path_num 6 --batch_size 256 --device cuda
python main.py --dataset CAS_B --length 60 --hidden 96 --path_num 9 --batch_size 256 --device cuda
python main.py --dataset CAS_C --length 75 --hidden 64 --path_num 9 --batch_size 256 --device cuda
```

论文参数设置为：CAS-A 使用 `T=60, G=6, F'=96`；CAS-B 使用 `T=60, G=9, F'=96`；CAS-C 使用 `T=75, G=9, F'=64`；优化器为 Adam，初始学习率为 `5e-4`。

训练过程中生成的模型权重和日志保存在运行目录，不属于提交文件。

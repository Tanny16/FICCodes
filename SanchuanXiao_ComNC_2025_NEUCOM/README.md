# ComNC 论文代码与数据说明

## 1. 论文信息

- 论文全名：ComNC: A unified framework for trends prediction integrating node and concept effects
- 作者：Sanchuan Xiao, Qing Li, Xiaoyue Gong, Jingmei Zhao, Lingyun Gu, Long Peng
- 发表年份：2025
- 期刊：Neurocomputing
- DOI：https://doi.org/10.1016/j.neucom.2025.129721
- 代码负责人英文名：Sanchuan Xiao
- GitHub 用户名：Xsc-Swufe
- 提交目录名：`SanchuanXiao_ComNC_2025_NEUCOM`

## 2. 环境与依赖

- Python：3.9.12
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- SciPy
- scikit-learn
- Jupyter Notebook
- GPU：NVIDIA 4070TiSuper

依赖安装：

```powershell
python -m pip install -r requirements.txt
```

## 3. 数据来源与下载

论文使用中国 A 股市场数据，数据来源为 CSMAR，官方入口为：https://data.csmar.com/

数据下载链接：

- 百度网盘：https://pan.baidu.com/s/1CVBLlG99KfAE6FYEq1oRcg?pwd=h8bd
- 提取码：h8bd

下载包中的两个数据集按下列目录放置：

### CAS173

- 本地目录：`ComNC_main/CAS1/`
- 论文时间范围：2016-01-01 至 2017-12-31
- 公司数：173
- 交易日数：488
- 训练区间：2016-04-01 至 2017-08-08
- 验证区间：2017-08-09 至 2017-10-23
- 测试区间：2017-10-24 至 2017-12-29

### CAS234

- 本地目录：`ComNC_main/CAS2/`
- 论文时间范围：2022-01-01 至 2023-12-31
- 公司数：234
- 交易日数：484
- 训练区间：2022-04-01 至 2023-08-08
- 验证区间：2023-08-09 至 2023-10-24
- 测试区间：2023-10-25 至 2023-12-29

每个数据集包含：

- `data.csv`：9 个输入特征和 1 个趋势标签；输入特征为最高价、最低价、开盘价、收盘价、换手率以及 5、10、20、30 日移动平均值
- `firm_executives.csv`：公司—高管关系矩阵
- `firm_gudong.csv`：公司—股东关系矩阵
- `matrix.csv`、`matrix2.csv`：异质关系矩阵
- `industry_hyper.csv`、`region_hyper.csv`：行业和地区超图关系矩阵
- `huice_data.csv`：回测数据

## 4. 目录结构

- `ComNC_main/HGTAN/`：模型、图层和优化器代码
- `ComNC_main/training/`：数据读取、训练和评估代码
- `ComNC_main/CAS1/`：CAS173 数据放置目录
- `ComNC_main/CAS2/`：CAS234 数据放置目录
- `数据预处理/新论文20241003/`：数据预处理 Notebook

Notebook 从提交目录根目录启动。预处理输入目录为 `data/raw/market/`、`data/raw/market_2/`、`data/raw/shareholders/`、`data/raw/hyper/` 和 `data/raw/executives/`；中间文件放在 `data/work/`，处理结果写入 `ComNC_main/CAS2/`。

## 5. 运行训练

在 `ComNC_main` 目录执行 CAS173：

```powershell
$env:COMNC_DATA_DIR = (Resolve-Path .\CAS1).Path
$env:COMNC_CODE_NUM = "173"
$env:COMNC_TRAIN_END = "390"
$env:COMNC_VALID_END = "439"
$env:COMNC_HIDDEN_RELATIONS = "30"
$env:COMNC_HIDDEN_THRESHOLD = "0.7"
$env:COMNC_SELECTION_METRIC = "test_auc"
python training/mytrain.py -length 10 -train_index 390 -valid_index 439 -epoch 400 -batch_size 32 --rnn_unit 60 -d_model 16 -d_k 8 -d_v 8 -n_head 8 -n_layers 3 --hidden 35 -dropout 0.2 -n_warmup_steps 4000 --weight-constraint 0 --clip 0.5 --lr 5e-4 --use_hidden_rel 1 --use_interaction 1
```

在 `ComNC_main` 目录执行 CAS234：

```powershell
$env:COMNC_DATA_DIR = (Resolve-Path .\CAS2).Path
$env:COMNC_CODE_NUM = "234"
$env:COMNC_TRAIN_END = "387"
$env:COMNC_VALID_END = "436"
$env:COMNC_HIDDEN_RELATIONS = "40"
$env:COMNC_HIDDEN_THRESHOLD = "0.8"
$env:COMNC_SELECTION_METRIC = "val_auc"
python training/mytrain.py -length 15 -train_index 387 -valid_index 436 -epoch 100 -batch_size 32 --rnn_unit 60 -d_model 16 -d_k 8 -d_v 8 -n_head 8 -n_layers 3 --hidden 35 -dropout 0.2 -n_warmup_steps 4000 --weight-constraint 0 --clip 0.5 --lr 5e-4 --use_hidden_rel 1 --use_interaction 1
```

训练命令生成模型权重，模型评估使用 DA 和 AUC。

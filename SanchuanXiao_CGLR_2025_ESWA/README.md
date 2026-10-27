# CGLR 论文代码与复现资料

## 1. 论文信息

- 论文全名：Unveiling risk propagation: a lead-lag-aware framework for financial market prediction
- 作者：Sanchuan Xiao, Qing Li, Xiaoyue Gong, Jingmei Zhao, Lingyun Gu, Long Peng
- 发表年份：2025
- 期刊：Expert Systems With Applications
- DOI：https://doi.org/10.1016/j.eswa.2025.128143
- 代码负责人英文名：Sanchuan Xiao
- GitHub 用户名：Xsc-Swufe
- 提交目录名：`SanchuanXiao_CGLR_2025_ESWA`

## 2. 环境与依赖

- Python：3.11
- PyTorch：2.0.1
- torchvision：0.15.2
- NumPy：1.24.4
- pandas
- scikit-learn
- PyTorch Geometric
- GPU：NVIDIA 4070 Ti Super

## 3. 数据来源与获取

论文说明 CAS-S、CAS-T、CAS-B 来自中国上海和深圳 A 股市场的 CSMAR 数据库，官方获取入口为：https://data.csmar.com/

数据下载链接：

- 百度网盘：https://pan.baidu.com/s/1vPKZD-B2MKdttCm9BCgudg?pwd=5ddv
- 提取码：5ddv

论文中的数据描述如下：

| 数据集 | 论文时间范围 | 公司数 | 交易日数 | 训练/验证/测试划分 | 下载后放置 |
|---|---:|---:|---:|---|---|
| CAS-S / CAS-MaxSize | 2021-01-01 至 2022-12-31 | 772 | 485 | 前 70% / 中间 20% / 后 10% | `CGLR_main/data/CAS_S.csv` |
| CAS-T / CAS-MaxTime | 2018-01-01 至 2023-12-31 | 120 | 2187 | 前 70% / 中间 20% / 后 10% | `CGLR_main/data/CAS_T.csv` |
| CAS-B / CAS-Balance | 2020-01-01 至 2023-12-31 | 305 | 970 | 前 70% / 中间 20% / 后 10% | `CGLR_main/data/CAS_B.csv` |

数据文件为无表头 CSV，每行 10 列，依次为开盘价变化率、最高价变化率、最低价变化率、收盘价变化率、换手率、5 日移动平均、10 日移动平均、20 日移动平均、30 日移动平均和收益率标签。

## 4. 目录结构

- `CGLR_main/CGLR/`：CGLR 模型代码
- `CGLR_main/training/`：训练、评估和批量运行入口
- `数据预处理/`：数据处理 Notebook
- `CGLR_main/data/`：下载或获得数据后放置完整 CSV；完整数据不直接提交代码仓库

## 5. 数据预处理

预处理 Notebook 默认要求从 `数据预处理` 目录运行，并使用环境变量 `CGLR_RAW_DATA_DIR` 指向 CSMAR 原始日频数据目录；未设置时默认读取项目根目录下的 `data/raw`。

处理顺序：

1. `第五个数据总数据.ipynb` -> `CGLR_main/data/CAS_T.csv`
2. `第6个数据.ipynb` -> `CGLR_main/data/CAS_S.csv`
3. `第7个数据.ipynb` -> `CGLR_main/data/CAS_B.csv`

## 6. 运行说明

在 `CGLR_main` 目录执行：

```powershell
python training/run_cglr_batch.py
```

该命令读取 `data/CAS_B.csv`、`data/CAS_S.csv` 和 `data/CAS_T.csv`，并在 `batch_runs/` 生成日志和结果。


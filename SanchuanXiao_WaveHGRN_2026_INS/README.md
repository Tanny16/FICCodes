# WaveHGRN 论文代码与数据说明

## 1. 论文信息

- 论文全名：Adaptive spatio-temporal wavelet hypergraph routing for evolutionary financial risk contagion
- 作者：Sanchuan Xiao, Jingmei Zhao, Lin Zhang, Shaofei Shen, Yang Wang, Rui Cheng, Qing Li
- 发表年份：2026
- 期刊：Information Sciences
- DOI：https://doi.org/10.1016/j.ins.2026.123795
- 代码负责人英文名：Sanchuan Xiao
- GitHub 用户名：Xsc-Swufe
- 提交目录名：`SanchuanXiao_WaveHGRN_2026_INS`

## 2. 环境与依赖

- Python：3.11
- PyTorch：2.0.1
- torchvision：0.15.2
- NumPy：1.24.4
- pandas
- scikit-learn
- PyTorch Geometric
- GPU：NVIDIA GeForce RTX 4070 Ti Super

依赖安装：

```powershell
python -m pip install -r requirements.txt
```

## 3. 数据来源与下载

### 中国数据

中国 A 股交易数据来自 CSMAR，指数成分股来自 Tushare。论文使用 CSI 300 和 CSI 500 两个指数，时间范围为 2020-01-01 至 2025-05-31；CSMAR 官方入口为 https://data.csmar.com/，Tushare 入口为 https://tushare.pro/。

数据下载链接：

- 百度网盘：https://pan.baidu.com/s/1vJXYibqg3_K_n76jyPVzIg?pwd=kft6
- 提取码：kft6

美国数据来源：LeadLagNet 项目：https://github.com/FICLAB/LeadLagNet

下载后放置：

- `WaveHGRN_main/CAS_data/CSI300.csv`
- `WaveHGRN_main/CAS_data/CSI500.csv`

论文中的数据统计为：CSI 300 包含 225 家公司，CSI 500 包含 371 家公司，两个数据集均为 1309 个交易日。每行数据包含 9 个输入特征和 1 个趋势标签；输入特征包括开盘价、收盘价、最高价、最低价、换手率以及 5、10、20、30 日移动平均值。训练、验证、测试按时间顺序划分为 80%、10%、10%。

中国 CSV 为无表头文件，训练代码按 `header=None` 读取。

## 4. 目录结构

- `WaveHGRN_main/WaveHGRN/`：WaveHGRN 模型、层和优化器
- `WaveHGRN_main/training/`：数据读取、训练和评估代码
- `WaveHGRN_main/CAS_data/`：中国 CSI 300、CSI 500 数据放置目录
- `WaveHGRN_main/run_CSI300.sh`：CSI 300 运行脚本
- `WaveHGRN_main/run_CSI500.sh`：CSI 500 运行脚本

## 5. 运行说明

在 `WaveHGRN_main` 目录执行中国数据：

```bash
bash run_CSI300.sh
bash run_CSI500.sh
```

训练模型权重写入 `checkpoints/`，模型评估使用 DA、AUC 和 MCC。

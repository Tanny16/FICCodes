# FDPGNN 论文代码与数据说明

## 1. 论文信息

- 论文全名：Frequency-decoupled progressive graph learning for unveiling heterogeneous risk contagion in financial markets
- 作者：Sanchuan Xiao, Rong Xing, Rui Cheng, Jingmei Zhao, Qing Li
- 发表年份：2026
- 期刊：Neurocomputing
- DOI：https://doi.org/10.1016/j.neucom.2026.133695
- 代码负责人英文名：Sanchuan Xiao
- GitHub 用户名：Xsc-Swufe
- 代码仓库：https://github.com/Xsc-Swufe
- 提交目录名：`SanchuanXiao_FDPGNN_2026_NEUCOM`

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

中国 A 股数据来自 CSMAR，CSI 300 成分股来自 Tushare。论文使用三个数据集：CAS232 为 232 家公司、1211 个交易日（2020-01-01 至 2024-12-31）；CAS187 为 187 家公司、1215 个交易日（2018-01-01 至 2022-12-31）；CAS133 为 133 家公司、1218 个交易日（2016-01-01 至 2020-12-31）。每个数据集包含 9 个输入特征和 1 个趋势标签，训练、验证、测试按时间顺序划分为 80%、10%、10%。

- CSMAR：https://data.csmar.com/
- Tushare：https://tushare.pro/
- 百度网盘：https://pan.baidu.com/s/119O1v2DnykeDvJBnygn87A?pwd=uw78
- 提取码：uw78

下载后将三个 CSV 文件放置为：

- `FDPGNN_main/data/data2016dao2020.csv`
- `FDPGNN_main/data/data2018dao2022.csv`
- `FDPGNN_main/data/data2020dao2024.csv`

美国数据：来源为 [LeadLagNet GitHub](https://github.com/FICLAB/LeadLagNet)，数据路径为 `FDPGNN_main/SP500/`。

## 4. 目录结构

- `FDPGNN_main/MSDGNN/`：FDPGNN 模型、网络层和优化器
- `FDPGNN_main/training/load_data.py`：中国 CAS 数据读取与时间切分
- `FDPGNN_main/training/tools.py`：训练与评估函数
- `FDPGNN_main/training/mytrain.py`：训练入口
- `FDPGNN_main/data/`：中国 CAS 数据放置目录

## 5. 运行说明

在 `FDPGNN_main` 目录执行：

```powershell
python -m training.mytrain --dataset CAS_232 -length 75 --hidden 128 --scale_num 4
python -m training.mytrain --dataset CAS_187 -length 45 --hidden 96 --scale_num 3
python -m training.mytrain --dataset CAS_133 -length 45 --hidden 96 --scale_num 4
```

论文参数设置为：CAS232 使用 `T=75, K=4, F=128`；CAS187 使用 `T=45, K=3, F=96`；CAS133 使用 `T=45, K=4, F=96`；优化器为 Adam，初始学习率为 `5e-4`。

训练过程中生成的模型权重和日志保存在运行目录，不属于提交文件。

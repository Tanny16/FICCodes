# 数据获取与处理

数据均不随 Git 提交。为匹配源仓库训练脚本，优先使用原项目提供的预处理 CSV：[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) / [百度网盘](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)，提取码 `i9iy`。来源为 `superlin30/GFMixer` 的固定源码快照；数据包未附精确版本号、文件校验值或链接有效期，此次未下载验证。

| 数据及实验入口 | 官方原始来源 | 本地 CSV 路径（相对本论文目录） |
| --- | --- | --- |
| ETTh1 / ETTh2（小时）与 ETTm1 / ETTm2（15 分钟），7 变量 | [数据发布者 ETTDataset](https://github.com/zhouhaoyi/ETDataset) | `dataset/ETTh1.csv`、`ETTh2.csv`、`ETTm1.csv`、`ETTm2.csv` |
| Weather，21 变量 | [MPI-BGC Jena 气象站](https://www.bgc-jena.mpg.de/wetter/) | `dataset/weather.csv` |
| Electricity，321 个输入序列 | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) | `dataset/electricity.csv` |
| Traffic，862 个输入序列 | [Caltrans PeMS](https://pems.dot.ca.gov/)，原始站点需申请账号 | `dataset/traffic.csv` |
| Exchange（附加脚本），8 变量 | [LSTNet 数据发布仓库](https://github.com/laiguokun/multivariate-time-series-data) | `dataset/exchange_rate.csv` |
| M4（附加脚本） | [M4 官方比赛仓库](https://github.com/Mcompetitions/M4-methods) | `dataset/m4/`，格式由 `data_provider/m4.py` 读取 |

Electricity 的 UCI 原始文件含 370 个客户，原始格式与项目使用的 321 序列 CSV 不同；气象与交通原始来源同样不等于已经处理的基准 CSV。代码快照未给出从原始来源完整生成这些基准版本的脚本或精确筛选日期，不能声称下载任意原始版本后直接复现。使用项目预处理包，并遵守各原始来源的使用条件；需要严格比对历史版本时，应先核对实际 CSV 的时间范围和列。

## 划分与预处理

- ETT：`Dataset_ETT_hour` 使用连续 12/4/4 个月作为训练/验证/测试，每月按 30 天计；分钟数据对应乘 4。验证和测试向前保留输入窗口所需上下文，标签仍处于对应分区。
- Weather、Electricity、Traffic、Exchange：`Dataset_Custom` 按文件时间顺序切分，训练 70%，测试 20%，其余为验证；有输入窗口上下文重叠。
- CSV 应有 `date` 列和数值序列；ETT 与默认其他脚本使用 `OT` 作为目标列名，具体以 `--target` 为准。
- 标准化只在训练区间拟合，验证/测试使用同一 scaler；时间特征由 `utils/timefeatures.py` 生成。无需先对 CSV 全体拟合标准化。
- 各脚本的输入窗口、预测窗口、变量数和采样频率以实际 `.sh` 参数及加载器为准；此次未下载数据确认精确日期范围、样本数或时间戳连续性。
- M4 按官方预定义训练/测试序列使用 `M4Meta` 的各频率预测长度；还需要官方基线文件才能进行完整竞赛评估，详见源加载器。此次未测试 M4 数据包和评估流程。

基础 GFMixer 实验从随机参数开始，无外部预训练权重。训练生成的检查点存放在 `checkpoints/`；若仅评估，应使用与同一配置训练出的检查点，不应替换成不同模型权重。

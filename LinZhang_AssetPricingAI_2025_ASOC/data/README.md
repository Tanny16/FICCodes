# 数据、外部资源与权限

本次 Git 提交不含完整金融数据或模型权重。提交者提供的完整归档为 [百度网盘](https://pan.baidu.com/s/1rkFMfRYUZCSNHaiaH7jIEA?pwd=daek)，提取码 `daek`；原文见 `baidu_archive.txt`。有效期未提供，此次未登录验证。请保留原文件名，解压后按下表放置。

| 数据 | 版本及内容 | 官方获取入口、条件 | 本地路径 |
| --- | --- | --- | --- |
| Gu–Kelly–Xiu firm characteristics | 本地 `datashare/readme.txt` 标注 April 2019；DATE、permno、94 个滞后企业特征、sic2 | [Dacheng Xiu 官方主页](https://dachxiu.chicagobooth.edu/)，在 Empirical Asset Pricing via Machine Learning 条目获取数据。当前官方版本可能更新，不能假设与归档完全一致；须引用原研究并遵守原数据条款 | `raw/datashare.csv` |
| CRSP 股票月收益 | 原始文件名 `kjtt6tbmn0gz9chy.csv`；代码使用 PERMNO、date、RET。导出查询及下载日期未归档 | [WRDS 的 CRSP 官方数据入口](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/)，需机构订阅、注册及适用权限；不能视为可自由再分发数据 | `raw/kjtt6tbmn0gz9chy.csv` |
| Welch–Goyal 宏观预测变量 | 归档名 `PredictorData2023.xlsx - Monthly.csv`；yyyymm、D12、E12、Index、b/m、ntis、tbl、lty、AAA、BAA、svar 等 | [Amit Goyal 官方主页](https://sites.google.com/view/agoyal145)。使用 2023 归档 Monthly 版本；当前官方更新版本可能不同 | `raw/PredictorData2023.xlsx - Monthly.csv` |
| 按年份处理数据 | 原始材料中 1957–2017 年的 Feather 对；主要实验使用至 2016 年。`*_df_data.feather` 为处理后输入、`*_real_RET.feather` 为标签；具体列由预处理决定 | 可在满足原数据权限后使用提交者归档，或从以上来源自行处理 | `processed/yearly/` |

上表路径均相对于本 `data/` 目录。企业特征原说明保留在 [characteristics_readme.txt](characteristics_readme.txt)，其中说明滞后处理和收益数据不在该公开特征文件内。

## 样本与处理

按 permno/PERMNO 和日期合并企业特征与股票收益，去除无收益观测；按月与宏观变量合并，构造超额收益，进行缺失值处理、截面秩归一化，再按年保存。主要预处理命令在论文目录执行：

```bash
ipython -c "%run notebooks/preprocessing/factor_processing.ipynb"
```

主要训练集为 1957–1986 年，测试集为 1987–2016 年；深度学习另含逐年扩展窗口。不同补充实验有各自划分，应以对应代码为准。原始完整数据的总样本数、CRSP 精确筛选条件、完整软件版本均未随材料归档，本次不据此编造。

## 可选集成与历史资源

- `notebooks/ensemble/集成训练(NN3).ipynb` 需要 `data/resources/best_model.pth`，可对应归档 `原始数据代码存档/模型训练原始代码和结果/训练结果/NN3/best_model.pth`；需确认与相同 NN3 定义及特征维度匹配。
- `notebooks/ensemble/dl-ensemble.ipynb` 保留另一历史实验，引用 `data/resources/one_loop_best_model.pth`、`NN1_best_predictions.csv`、`bagging_model_predictions.csv`。归档中存在其他模型的同名检查点，不能用它们替代所需 NN2 检查点；这些准确历史输入未完整找到。
- `code/ensemble/vail_predict.py` 引用 `data/resources/NN1_Run2_Epochs50_LR0.001_Patience10_R2_-0.032997.pth`，准确文件未在此次材料中找到。
- 网盘归档中有 `NN3_best_predictions.csv`、`NN3_best_predictions2.csv`，以及多个 `.pth`/`.joblib`；原相对位置和字节数见 [external_files.csv](external_files.csv)。文件名不同的预测或检查点不应直接互换。

模型检查点及预测表无独立公开下载版本或可验证模型版本号；以上说明保留实际可知的来源与缺口。无法取得金融数据时，主要真实数据训练与评估不能执行；语法和小型合成样例检查仍可执行。

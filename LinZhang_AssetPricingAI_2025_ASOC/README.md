# Unraveling asset pricing with AI: A systematic literature review

| 项目 | 信息 |
| --- | --- |
| 作者 | Yan Chen, Lin Zhang, Zhilong Xie, Wenjie Zhang, Qing Li |
| 发表 | 2025, Applied Soft Computing (ASOC), Volume 175, Article 112978 |
| 正式论文 | [DOI: 10.1016/j.asoc.2025.112978](https://doi.org/10.1016/j.asoc.2025.112978) |
| 作者发布的另一个版本 | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5380505)；与期刊终稿可能存在差异 |
| 归档及代码负责人 | Lin Zhang ([superlin30](https://github.com/superlin30)) |

本目录保存综述配套的资产定价实验代码、QuantPlus 数据处理代码、文献整理表、作图数据和小型结果表。虽然论文为综述，原始材料仍包含模型对比实验，因此一并提供可浏览源码。

## 内容与来源

- `code/platform/`：数据预处理、传统机器学习评估、深度学习模型及训练函数。
- `notebooks/preprocessing/`、`notebooks/training/`：主要因子预处理与模型训练笔记本。
- `notebooks/ensemble/`、`code/ensemble/`：集成预测和投资组合评估；部分需要额外历史预测或检查点。
- `notebooks/supplemental/`：补充实验的不同版本；`stu3` 至 `stu8` 保留原始分组以区分来源，未将不同实验合并。
- `results/`、`figures/`、`literature/`：已有结果、图表基础材料与文献表。这些是历史归档结果，不代表此次重新训练得到的结果。
- `SOURCE_MANIFEST.json`：归档路径、原材料相对路径及原文件 SHA-256。

整理时清空笔记本运行输出、移除系统缓存和重复压缩包，将研究者电脑上的绝对路径改为本目录内的相对路径。未改动模型结构、超参数、训练划分或指标定义。未上传论文投稿信、审稿回复等投稿文件。

## 环境

原笔记本记录的 Python 版本分布在 3.9 至 3.12；原始完整环境锁定文件未归档。`requirements.txt` 根据实际导入重建，不能视为历史实验的精确版本记录。

```bash
python -m venv .venv
# 激活虚拟环境后安装；GPU 版 PyTorch 请按本机 CUDA 条件选择官方安装包。
python -m pip install -r requirements.txt
```

此次小样例检查环境：Python 3.11.15、PyTorch 2.6.0+cu124、NumPy 1.23.5、pandas 1.5.3、SciPy 1.10.1、scikit-learn 1.2.2、joblib 1.5.3。`pyarrow` 是 Feather 文件读写依赖，`torch-geometric` 用于 GNN；此次未安装或验证这两个依赖。不同补充实验可能需要独立环境，尤其是旧版 scikit-learn 参数接口。

## 数据获取与放置

详细来源、版本、字段、权限、划分及下载路径见 [data/README.md](data/README.md)。提交者提供的原始归档链接同时保存在 [data/baidu_archive.txt](data/baidu_archive.txt)：

- [百度网盘归档](https://pan.baidu.com/s/1rkFMfRYUZCSNHaiaH7jIEA?pwd=daek)，提取码 `daek`。
- 链接由提交者提供，未提供有效期；此次未登录网盘验证文件完整性或有效期。
- 网盘归档包含第三方来源数据，使用仍须满足原数据许可，尤其是 CRSP/WRDS；网盘链接不替代访问授权。
- 完整数据、Feather 文件、模型权重、大型预测表未进入 Git。文件目录及大小见 [data/external_files.csv](data/external_files.csv)。

将原始数据整理为：

```text
data/raw/kjtt6tbmn0gz9chy.csv
data/raw/datashare.csv
data/raw/PredictorData2023.xlsx - Monthly.csv
data/processed/yearly/1957_df_data.feather
data/processed/yearly/1957_real_RET.feather
...（其他年份同样成对放置）
```

## 运行顺序

以下命令均从本论文目录执行。主要笔记本按根目录相对路径读取数据；使用命令行 `%run` 可使工作目录保持一致。

1. 获取并放置上述数据。若使用归档中已有的按年份 Feather 数据，可跳过预处理。
2. 运行原始因子处理笔记本，合并收益、企业特征和宏观变量，完成缺失值处理及截面秩变换，再保存按年份数据：

   ```bash
   ipython -c "%run notebooks/preprocessing/factor_processing.ipynb"
   ```

3. 按所需实验分别执行训练和内置评估，不必依次执行所有训练脚本：

   ```bash
   ipython -c "%run notebooks/training/data_loaders.ipynb"
   ipython -c "%run notebooks/training/deep_learning_nonrecurrent.ipynb"
   ipython -c "%run notebooks/training/deep_learning.ipynb"
   ```

4. 比较输出与 `results/model_comparison.xlsx`。模型和结果默认写入本地 `outputs/`；这些输出不会提交到仓库。训练笔记本含多个模型和长循环，执行前应阅读并选择所需实验。

`code/platform/preprocess.py` 是另一份函数化处理版本，可用 `python code/platform/preprocess.py` 执行；其清理步骤与主因子笔记本并非同一版本，不应把两条处理路线的输出混用。传统 ML 的 `train_model`、深度学习的模型定义和训练函数供集成调用，单独执行这些定义文件不会自动复现所有实验。补充笔记本的模型、文件名与划分各异，运行前须核对其输入；不承诺任意顺序全部执行。

## 划分、模型与已知限制

- 主要 `data_loaders.ipynb` 使用 1957–1986 年训练，1987–2016 年测试。
- 主要深度学习笔记本按年份扩展训练窗口：测试年为 1987–2016 年，每次使用 1957 年至测试年前一年的数据。原代码存在在名为 test 的集合上选择停止时点/超参数的实现；本次未重新设计验证集，不能把这些归档结果解释为新增的独立留出验证。
- 基础训练无需预训练模型。集成及历史预测分析所需资源见 [data/README.md](data/README.md)；部分历史路径对应的文件未在此次材料中找到，不能保证这些可选分析直接运行。
- 函数化 `clean_and_fill_na` 从含缺失值的列中构建候选特征集，可能遗漏完全观测的特征。此为原始代码行为，本次仅记录；完整预处理和论文数值一致性尚未重新验证。
- 未运行全部真实数据训练、GNN、全流程预处理或完整投资组合实验。已执行的语法、指标、秩变换及合成数据模型检查见 [VALIDATION.md](VALIDATION.md)。

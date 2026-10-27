# Getting LLM to think and act like a human being: logical path reasoning and replanning

| 项目 | 信息 |
| --- | --- |
| 作者 | Lin Zhang, Qing Li, Yang Wang, Jingmei Zhao |
| 发表 | 2025 11th International Conference on Computing and Artificial Intelligence (ICCAI), pp. 301–308 |
| 论文 | [DOI: 10.1109/ICCAI66501.2025.00054](https://doi.org/10.1109/ICCAI66501.2025.00054) |
| 代码负责人 | Lin Zhang ([superlin30](https://github.com/superlin30)) |
| 原仓库 | [Real-World-Web-Agent 固定快照](https://github.com/superlin30/Real-World-Web-Agent/tree/1cd1e00b41794e7e07873b1d269c0fbdedbc0851) |

本目录将公开仓库中的提示词、方法图与本地实验笔记本合并归档。现有材料未包含完整运行依赖，**不能据此完整重跑网页 Agent**；以下区分可离线运行的组件与保留供审阅的研究原型。原仓库 MIT `LICENSE` 保留；新增整理文件不扩大第三方材料或数据的授权范围。

## 材料结构

- `notebooks/RUN_code.ipynb`：主要 Agent 研究原型，原环境元数据为 Python 3.9.7。
- `notebooks/drafts/Task_Tree.ipynb`：后续任务树/SQL 方向草稿，原环境为 Python 3.11.5；不是稳定的完整实验入口。
- `prompts/`：初始规划、新技能创建、知识图谱推理及重新规划提示词。
- `data/examples/`：原始三元组文本 `KG.txt` 与 `Database_kg(1).txt`。
- `examples/`：从原笔记本提取的知识图谱路径传播与任务树离线示例，不调用 API、不操作真实网页。
- `figs/`、[VIDEOS.md](VIDEOS.md)：论文方法图和已有公开演示视频入口。
- `SOURCE_MANIFEST.json`：每个源材料的来源及 SHA-256。

整理仅清空笔记本运行输出、移除硬编码凭据及账号注释、将 API Key 改为读取环境变量、将 KG 文件改为项目内相对路径。原型中未完成的草稿和算法逻辑保留，并明确其限制。

## 先运行可复查的离线组件

从本论文目录执行，Python 标准库即可，无 API Key、浏览器或第三方包要求：

```bash
python examples/kg_reasoning_demo.py
python examples/task_tree_demo.py
```

第一条从 `data/examples/KG.txt` 读取三元组，以 `My father`、`Academic conference` 为实体构造推理路径树；第二条检查子树查找、叶节点状态及节点路径。它们是原型的局部行为检查，不是论文整体成功率评估。

## 历史 Agent 原型的环境与运行方法

主笔记本使用 `openai.ChatCompletion.create` 等旧接口，草稿使用 `OpenAI()` 客户端；两者不能共用同一 OpenAI 依赖约束。根据原代码导入整理了两个环境清单，未将缺失模块伪装成可通过 pip 安装的标准 MetaGPT。

主原型：Python 3.9（原记录 3.9.7）、`openai>=0.27,<1`、`langchain<0.1`、Selenium、ChromaDB、Gradio 等。版本界限依据旧接口重建，其他包的精确历史版本未知，`requirements-legacy.txt` 不是已验证的锁定环境。

```bash
python -m venv .venv
# 激活独立环境后：
python -m pip install -r requirements-legacy.txt
```

通过系统环境变量配置 `OPENAI_API_KEY`；Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY = Read-Host '请输入你自己的 API Key'
```

`.env.example` 仅列变量名，不含真实凭据；笔记本不自动加载 `.env`。如有需要，再配置 `OPENAI_PROXY`。模型名保留原代码中的 `gpt-4`、`gpt-3.5-turbo` 系列历史标识；模型的当前可用性、权限及费用未在此次任务中验证，替换模型会改变实验条件。

**当前阻塞点：**主笔记本引用 `metagpt.management.WebBrowser`、`UserManager`、`SkillManager`，以及 `metagpt.management.skill_library` 中的 `Open_Browser`、`Flight_Booking`、`Login_And_Book`、`Search`、`Fly_Direct`、`Scrape_Flight_Data`。此次公开仓库和本地资料都没有这些自定义实现，也没有可验证下载地址。负责人确认仅按现有材料归档，因此未编造替代实现或下载链接。

若以后补齐上述自定义源码和相应用户/技能存储资源，使模块位于该 Python 环境可导入路径，并具备浏览器/驱动及本人 API 访问条件，可从本论文目录启动：

```bash
jupyter lab --notebook-dir=.
```

打开 `notebooks/RUN_code.ipynb` 后先将内核工作目录设置为本论文目录：

```python
%cd ..
```

在 Jupyter 默认以内核笔记本所在目录启动时，上述命令从 `notebooks/` 回到论文目录；先用 `%pwd` 核对位置。按单元格顺序审阅/执行，KG 输入为 `data/examples/KG.txt`。本原型是有交互状态的历史实验，包含真实网站操作及旧日期示例，不作为无交互批量执行脚本。此次未执行任何付费 API、登录网站或订票动作。

任务树草稿如需审阅，单独用 Python 3.11 环境安装 `requirements-draft.txt` 并在 Jupyter 打开 `notebooks/drafts/Task_Tree.ipynb`。草稿第 18 个单元格存在未完成 `else` 分支，且有 `get_kg_info`、`Task_Decomposition`、`TreeNode` 等未定义对象；不应执行“Run All”。已提供的离线任务树示例仅提取其中完整的类定义。

## 数据、外部资源与验证

数据来源、字段及缺失范围见 [data/README.md](data/README.md)。两份 KG 文本是此次材料中实际提供的小型示例；没有附带完整任务集、统一划分、评分标签或训练/评估脚本，因此不能声称已经复现论文的任务成功率。完整 Agent 也依赖缺失的技能库和用户管理资源。

本地两个大型原始视频保留在提交者电脑，不放入 Git；公开仓库已有的演示视频链接见 [VIDEOS.md](VIDEOS.md)。视频只作演示，不替代测试集或完整执行记录。

主笔记本语法与两项离线示例检查通过；草稿缺陷和端到端限制如实记录在 [VALIDATION.md](VALIDATION.md)。

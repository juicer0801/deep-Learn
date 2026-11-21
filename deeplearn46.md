---
title: PyTorch 深度学习全流程 API 速查
date: 2026-10-14 21:27:54
tags:
  - PyTorch
  - 深度学习
  - API
---

**目录**

按写代码的顺序：数据清洗与划分 → 加载与预处理 → 特征工程 → 模型构建 → 训练 → 评估 → 可视化 → 导出与部署。

- **数据清洗与划分**
  - 基础清洗操作：去重、缺失值、异常值
  - 算法驱动的智能清洗（高级）
  - 数据质量校验与管道集成（高级）
  - 清洗 API 的层级对照
  - 先看决策逻辑：你的数据该用什么划分
  - 基础随机划分
  - 分层划分：分类任务的基本约束
  - 分组划分：防止实体级泄漏
  - 交叉验证
  - 时间序列划分
  - 分布偏移感知的划分（高级）
  - 高级集成：把划分逻辑收进管道
  - 划分策略速查与关键陷阱
- **加载与预处理**
  - 数据加载容器：`Dataset` 与 `DataLoader`
  - 采样与批次整理
  - 图像与视频：`torchvision.transforms.v2`
  - 变换能力的增强
  - 文本：从 `torchtext` 迁移
  - 音频：`torchaudio`
  - 预处理流水线的位置
  - 管道的组织与复用
  - 加载性能与规模
  - 高级数据 API 的定位对照
- **特征工程**
  - 数值特征：缩放、变换、分箱、交互
  - 类别特征：编码策略的选择取决于基数
  - 日期时间特征
  - 特征选择：先构造，再筛选
  - 文本特征：从 TF-IDF 到上下文嵌入
  - 图像特征：手工算子与 CNN 嵌入
  - 时间序列特征：统计描述子与大规模自动提取
  - 聚合与滞后特征：业务场景中信息量最高的手工特征
  - 自动化特征工程（高级）
  - 深度学习时代的表格特征：让模型学习交叉，而非手工枚举
  - 特征工程 API 的层级对照与关键原则
- **模型构建**
  - 模型容器
  - 基础层
  - 激活函数
  - 权重初始化
  - 模型选择：先定约束，再定基线
- **训练**
  - 训练循环骨架：五步是不变的核心
  - 损失函数
  - 优化器：起点决定收敛路径
  - 学习率调度：训练中最具杠杆的超参数
  - 正则化与稳定训练：防止过拟合与训练崩溃
  - 精度与显存
  - 分布式训练
  - 编译加速
  - 训练流程控制
  - 超参数搜索
  - 训练配置模板与避坑清单
  - 高级 API 与底层 API 的映射
- **评估**
  - 基础任务指标：选对指标比调高指标更重要
  - 高级评估维度：指标之外的可信度
  - 统计比较：两个模型的差距是真实的吗
  - 持续评估：模型上线后，评估没有结束
  - 评估协议与避坑清单
- **可视化**
  - 训练监控：曲线是第一个诊断信号
  - 模型结构：确认数据流与参数
  - 数据与预测诊断：混淆矩阵、ROC、散点
  - 特征归因与注意力：模型在看哪里
  - 实验追踪与交互式面板（高级）
  - 出版级图表的规范（高级）
  - 绘图 API 的选择对照
- **导出与部署**
  - 模型导出
  - 推理期加速
  - 量化
  - 导出与部署的选择对照

<!-- more -->

---

## 数据清洗与划分

数据清洗是决定“数据里有什么”的上游环节，与下游决定“数据长什么样”的变换处理分开。数据划分是深度学习工作流中**最容易被低估、出错后代价最高**的环节——随机划分在时间序列或分组结构上会造成静默的信息泄漏，验证分数虚高，最终模型上线后表现断崖式下跌。这一篇先决定保留什么（清洗），再决定怎么切（划分）。

### 基础清洗操作：去重、缺失值、异常值

#### 1. `torcharrow.DataFrame`（PyTorch 原生表格清洗）

**解释**：PyTorch 生态中内置的 DataFrame 库，基于 Apache Arrow 列式内存格式，提供类 Pandas 的 API，并可直接转换为 PyTorch Tensor。它的 **Data Cleaning** 模块提供了 `fill_null`（填充空值）、`drop_null`（删除含空值的行）、`drop_duplicate`（去重）等操作，适合需要在 PyTorch 管道内部完成轻量表格清洗的场景。需要注意的是 `drop_duplicate` 目前标记为**实验性 API**。

**传参数示例**：

```python
import torcharrow as ta

df = ta.DataFrame({
    "id": [1, 2, 2, 3, 4],
    "score": [0.9, None, 0.8, 0.7, None],
})

# 删除重复值，保留第一个
df = df.drop_duplicate(subset=["id"], keep="first")

# 删除含空值的行
df = df.drop_null(subset=["score"])

# 填充空值（需在填充前保留）
df = df.fill_null(value=0.0, subset=["score"])
```

`torcharrow` 的独特价值在于**从表格清洗到张量转换的链路最短**：清洗后的 DataFrame 可以直接转为 `torch.Tensor`，不需要经过 Pandas 中转。

---

#### 2. `polars` 的惰性清洗

**解释**：Polars 的 **Lazy API** 允许把去重、空值过滤、类型转换、条件填充等清洗步骤组合成一个优化后的执行计划，最后用 `.collect()` 一次性执行。在数据量较大时，这比逐条 pandas 操作高效得多。常用清洗方法包括 `unique` / `drop_duplicates`（去重）、`drop_nulls`（删除空值行）、`fill_null`（填充空值）、`cast`（类型转换）。

**传参数示例**：

```python
import polars as pl

clean_df = (
    pl.scan_csv("raw.csv")
    .unique(subset=["id"], keep="first")                     # 去重
    .drop_nulls(subset=["label"])                             # 删除关键字段缺失
    .with_columns([
        pl.col("age").cast(pl.Int32),                         # 类型转换
        pl.col("age").fill_null(pl.col("age").median()),      # 中位数填充
        pl.col("score").fill_null(0.0),                       # 常量填充
    ])
    .filter(pl.col("age").is_between(0, 120))                # 异常值过滤
    .collect()
)
```

`qualityclean` 是构建在 Polars 之上的自动化清洗包装器，提供 `qc.clean()` 一键完成列名规范化、空白修剪、占位符转空值、缺失值处理、去重和类型推断，并生成**可审计的清洗报告**。

---

### 算法驱动的智能清洗（高级）

#### 1. `cleanlab.Datalab`：一站式数据问题诊断

**解释**：Cleanlab 是当前数据为中心 AI 领域最成熟的开源工具之一。`Datalab` 类可以**用你已有的模型输出（预测概率或特征嵌入）** 来自动检测多种数据问题：标签错误、离群点、近似重复、IID 违反、表现不佳的群体等。它的核心优势在于**不依赖手工规则**，而是利用模型行为来发现“人工难以枚举”的脏数据模式。Cleanlab 已经在对 ImageNet、MNIST 等基准数据集的清洗中发现了数千个标签错误。

**关键方法**：`find_issues(features=..., pred_probs=...)` 执行问题检测；`report()` 生成汇总报告；`rank_classes_by_label_quality` 用于类级别的问题排序。

**传参数示例**：

```python
from cleanlab import Datalab

lab = Datalab(data=your_dataset, label_name="label")
lab.find_issues(
    features=feature_embeddings,     # 模型倒数第二层输出或任意嵌入
    pred_probs=pred_probs,           # (N, num_classes) 预测概率
)
lab.report()                         # 输出每个数据点的问题严重程度汇总
```

对于**标签纠错**，可以使用 `CleanLearning` 包装任意 sklearn 兼容分类器，一行代码完成标签问题检测和鲁棒模型训练：

```python
from cleanlab.classification import CleanLearning

cl = CleanLearning(sklearn_compatible_model)
label_issues_info = cl.find_label_issues(data, labels)   # 检测标签问题
cl.fit(data, labels)                                     # 用清洗后的数据训练鲁棒模型
```

Cleanlab 通过 `skorch` 可以无缝适配 PyTorch 模型，因此不要求你切换到 sklearn 生态。

---

#### 2. `PyOD` / `alibi-detect`：专用异常检测

**解释**：当清洗需求集中在**异常值/离群点**时，专用异常检测库比通用清洗工具更直接。`PyOD` 提供了数十种无监督异常检测算法（Isolation Forest、LOF、AutoEncoder 等），API 高度统一，适合在训练前对特征矩阵做离群点过滤或降权。`alibi-detect` 则同时覆盖离群检测和**数据漂移检测**，适合生产环境的数据质量监控。

**传参数示例（PyOD）**：

```python
from pyod.models.iforest import IForest

detector = IForest(contamination=0.05)       # 预计 5% 为异常
detector.fit(X_features)
outlier_scores = detector.decision_scores_    # 越高越异常
inlier_mask = detector.labels_ == 0          # 0 表示正常样本
X_clean, y_clean = X_features[inlier_mask], y[inlier_mask]
```

---

### 数据质量校验与管道集成（高级）

#### 1. `Great Expectations`：可执行的数据契约

**解释**：Great Expectations（GX）把“什么算合法数据”写成 **Expectation Suite**——一组可执行的断言（如某列不能为空、某列值必须在范围内、某列必须唯一）。校验在 **Checkpoint** 中运行，结果可以生成 Data Docs 报告，也可以在失败时直接阻断管道。它的核心价值是**把数据清洗从“事后修数据”前移为“事前拦坏数据”**，适合在数据进入训练管道之前作为质量门禁。

**传参数示例（现代 Fluent API）**：

```python
import great_expectations as gx

context = gx.get_context()
datasource = context.data_sources.add_pandas(name="train_df")
batch = datasource.read_dataframe(df, asset_name="train")

suite = context.suites.add(gx.ExpectationSuite(name="train.critical"))
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="label")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age", min_value=0, max_value=120
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="id")
)

result = batch.validate(suite)
assert result.success, result          # 校验失败则阻断训练
```

建议将 **“关键期望”**（空值率、范围、唯一性）与 **“诊断性期望”** 分开管理：关键失败直接 `sys.exit(1)` 阻断管道，诊断失败仅记录告警。

---

#### 2. `DVC` 数据质量门禁

**解释**：DVC 本身不执行清洗，而是把**清洗和校验脚本编排进管道**，让数据质量检查成为可复现、可版本化的一个阶段。典型模式是：在 `dvc.yaml` 中定义一个 `validate_data` 阶段，运行你的校验脚本；脚本对脏数据返回非零退出码，DVC 管道随即失败，坏数据不会进入下游训练。校验产物（如 `quality_report.json`、`flagged_rows.csv`）可以作为下游阶段的依赖，实现**质量报告驱动的数据流**。

**传参数示例（`dvc.yaml` 片段）**：

```yaml
stages:
  validate_data:
    cmd: python validate_quality.py data/raw.csv --output quality_report.json
    deps:
      - data/raw.csv
      - validate_quality.py
    outs:
      - quality_report.json
      - flagged_rows.csv
    metrics:
      - quality_report.json:
          cache: false
```

校验脚本的核心逻辑是：读取数据 → 逐列检查空值率、范围、唯一性 → 写入报告 → **根据是否通过返回退出码**。

---

### 清洗 API 的层级对照

| 层级 | 代表 API / 工具 | 它解决的问题 | 输出形态 |
|---|---|---|---|
| **表内基础操作** | `torcharrow`、`polars` Lazy | 去重、空值填充/删除、类型转换 | 清洗后的 DataFrame / Tensor |
| **一键自动化** | `qualityclean`（Polars） | 重复清洗代码的模板化 | 清洗后数据 + 审计报告 |
| **算法驱动诊断** | `cleanlab.Datalab` | 标签错误、离群点、近似重复 | 问题分数 + 索引，供人工复核/过滤 |
| **专用异常检测** | `PyOD`、`alibi-detect` | 特征空间离群点、数据漂移 | 异常分数 / 布尔掩码 |
| **质量契约校验** | `Great Expectations` | 定义并强制执行数据合法性规则 | 通过/失败断言 + Data Docs |
| **管道级门禁** | `DVC` stages + 自定义脚本 | 让清洗和校验可复现、可版本化 | 管道失败/继续 + 质量报告 |

**策略建议**：**规则明确的基础清洗**（去重、类型、范围）用 `polars` / `torcharrow` 离线完成；**标签噪声和隐性脏数据**交给 `cleanlab` 做算法诊断，结果人工复核；**数据合法性契约**用 `Great Expectations` 固化，并通过 `DVC` 嵌入管道作为质量门禁。清洗的产出应当是**一份经过校验、可追溯的样本集合与索引**，而不是在每个训练 epoch 中动态执行清洗逻辑。

---

### 先看决策逻辑：你的数据该用什么划分

在调用任何 API 之前，先用三个问题确定约束，再选择工具：

```
数据是否有时间维度？
  ├── 是 → 时序划分（TimeSeriesSplit），严禁随机打乱
  └── 否 → 观测是否有分组结构？（同一患者/用户/文档的多条记录）
            ├── 是 → 分组划分（GroupKFold / GroupShuffleSplit）
            └── 否 → 目标是否严重不平衡？（少数类 < 10%）
                      ├── 是 → 分层划分（StratifiedKFold / stratify=y）
                      └── 否 → 简单随机划分（random_split / train_test_split）
```

数据集规模也会影响划分策略：**小于 5,000 行**时，交叉验证比单次留出更可靠；**大于 100,000 行**时，测试集比例可以压到 5% 甚至更低，因为大样本下单次留出的估计已经足够稳定。

---

### 基础随机划分

#### 1. `torch.utils.data.random_split`

**解释**：PyTorch 原生随机划分，把数据集按给定长度随机切成**互不重叠**的 `Subset`。它只做索引层面的切分，不涉及分层或分组，适合数据量足够大、类别分布均匀、无时间/分组结构的场景。

**关键参数**：`dataset`，`lengths`（各子集长度，比例或绝对数），`generator`（固定种子以保证可复现）。

**传参数示例**：

```python
import torch
from torch.utils.data import random_split, DataLoader

train_set, val_set, test_set = random_split(
    dataset,
    lengths=[0.7, 0.15, 0.15],
    generator=torch.Generator().manual_seed(42),   # 可复现
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
```

常见误区：认为 `random_split` 会修改原始数据集——它返回的是索引视图，底层数据共享。如果训练集和验证集需要不同的变换，需要先拿到索引，再用 `Subset` 分别套用不同的变换。

---

#### 2. `sklearn.model_selection.train_test_split`

**解释**：scikit-learn 的一行式划分入口，接受任意可索引对象（数组、DataFrame、稀疏矩阵），返回对齐的训练/测试子集。可以直接接收 `X, y` 并返回 `X_train, X_test, y_train, y_test`，不需要手动索引对齐。`stratify` 参数使其成为分类任务中最常用的分层划分工具。

**关键参数**：`test_size`（浮点比例或整数绝对数），`train_size`，`random_state`，`shuffle`（默认 `True`），`stratify`（分层标签）。

**传参数示例**：

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42,
)
```

`shuffle=False` 时 `stratify` 必须为 `None`；同时指定 `test_size` 和 `train_size` 导致总和超过 1 会抛错。

---

### 分层划分：分类任务的基本约束

#### 1. `stratify` 参数（sklearn）

**解释**：分层抽样保证训练集、验证集、测试集的**类别比例与原始数据一致**。不做分层时，随机划分可能导致某些类别在验证集中完全缺失或严重不足，使评估指标失去意义。`train_test_split(..., stratify=y)` 是最直接的入口；`StratifiedKFold` 和 `StratifiedShuffleSplit` 是交叉验证版本。

**关键参数**：`stratify` 接收与 `y` 同长度的类别标签数组。

**传参数示例**：

```python
from sklearn.model_selection import train_test_split

# 逐层划分：先 train/temp，再在 temp 内分层切 val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42,
)
```

对于 PyTorch，**不要用 `WeightedRandomSampler` 做划分**。它是有放回采样，默认 `replacement=True`，会让某些样本在每个 epoch 中重复出现、另一些被跳过，并且在交叉验证中复用同一个采样器会引入数据泄漏。正确做法是**先用分层索引拿到 `train_idx` / `val_idx`，再用 `Subset` 构造子集**。

---

### 分组划分：防止实体级泄漏

#### 1. `GroupKFold` / `GroupShuffleSplit`

**解释**：当数据中存在**分组结构**（同一患者的多次检查、同一用户的多个行为、同一文档的多个段落）时，普通随机划分会把同一组的数据同时放进训练集和验证集，导致**信息泄漏**——模型在验证集上“见过”同一实体的其他样本，分数虚高。`GroupKFold` 保证同一个组不会同时出现在训练折和验证折中；`GroupShuffleSplit` 是单次划分版本。分组信息可以编码为任意领域特定的分组标签（整数），例如 `patient_id`。

**关键参数**：`n_splits`，`test_size`，`groups`（传给 `.split(X, y, groups=groups)` 的分组标签数组）。

**传参数示例**：

```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# 单次划分：同一患者不会跨越 train/test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

# 交叉验证：分组
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=patient_ids):
    ...
```

**关键检查**：分组划分不仅要在 train/test 边界上成立，**每一个切分边界都要检查**。常见失败模式是 train/test 做了分组，但 validation 仍然随机切分，泄漏依然存在。

---

#### 2. `StratifiedGroupKFold`

**解释**：同时满足**分组约束**和**分层约束**，是分类任务中分组数据的最强划分器。在分组和分层无法同时完美满足时，它会尽力逼近目标分布，而不是像 `GroupKFold` 那样完全忽略类别比例。在医学、金融等领域，当同一患者/客户有多条记录且类别不平衡时，这是首选。

**关键参数**：`n_splits`，`shuffle`，`random_state`；`.split(X, y, groups=groups)`。

**传参数示例**：

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in sgkf.split(X, y, groups=patient_ids):
    ...
```

---

### 交叉验证

#### 1. `KFold` / `StratifiedKFold`

**解释**：K 折交叉验证把数据分成 K 折，轮换用 K-1 折训练、1 折验证。`KFold` 是基础版本；`StratifiedKFold` 在每折中保持类别比例，是分类任务的默认选择。两者都**只产生索引**，不直接切分数据，因此可以配合任意训练框架使用。在 PyTorch 中，交叉验证的本质是**在每折中动态构造 `Subset` 或 `SubsetRandomSampler`**，`DataLoader` 本身不承担划分职责。

**关键参数**：`n_splits`（默认 5），`shuffle`（是否在划分前打乱），`random_state`（`shuffle=True` 时用于复现）。

**传参数示例**：

```python
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_sub = Subset(dataset, train_idx)
    val_sub   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_sub, batch_size=32, shuffle=False)
    # ... 训练与评估 ...
```

---

#### 2. `cross_validate`（高级）

**解释**：scikit-learn 的**元评估核心函数**，比 `cross_val_score` 更强大。它返回一个字典，不仅包含测试分数，还允许收集**拟合时间、评分时间、训练分数**，甚至可以返回每个折上训练好的 estimator。`cv` 参数可以直接传入任何交叉验证分割器（包括 `TimeSeriesSplit`、`GroupKFold`），因此它也是把 sklearn 生态中的划分策略复用到自定义评估流程的入口。

**关键参数**：`estimator`，`X`，`y`，`cv`（分割器或整数），`scoring`（单个或多个指标字符串），`groups`（分组标签），`return_train_score`（是否返回训练分数），`return_estimator`（是否返回每折模型），`n_jobs`（并行进程数）。

**传参数示例**：

```python
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y,
    cv=cv,
    groups=patient_ids,
    scoring=("accuracy", "f1_macro"),
    return_train_score=True,
    return_estimator=False,
    n_jobs=-1,
)
print(scores["test_accuracy"].mean(), scores["test_f1_macro"].mean())
```

`cross_validate` **不保留最终模型**（除非 `return_estimator=True`），它用于评估流程而非产出部署模型。另一个关键陷阱：**预处理器必须在每折内部仅用训练数据拟合**，如果在 `cross_validate` 之前全局拟合了 scaler，泄漏就已经发生了。

---

### 时间序列划分

#### 1. `TimeSeriesSplit`

**解释**：时间序列数据**严禁随机打乱**——用未来信息预测过去会造成直接的时间泄漏。`TimeSeriesSplit` 生成**前向扩展**的折：第 k 折的训练集是前 k 个时间块，验证集是第 k+1 个时间块，保证训练数据在时间上严格早于验证数据。这是 scikit-learn 中处理时序交叉验证的标准入口。

**关键参数**：`n_splits`，`test_size`（每折验证集大小），`gap`（训练集与验证集之间的**禁运间隔**，用于模拟预测延迟或特征回看窗口，防止边界处的自相关泄漏），`max_train_size`（限制训练窗口上限）。

**传参数示例**：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=100,
    gap=10,              # 训练与验证之间留 10 个时间步的间隔
)

for train_idx, val_idx in tscv.split(X_time_ordered):
    # train_idx 全部早于 val_idx，且中间有 gap 隔离
    ...
```

`gap` 的设置原则是**至少等于最大特征回看窗口的长度**——如果特征用到了过去 24 小时的数据，gap 至少应为 24 小时对应的时间步数，否则验证集起点的特征仍然包含了“接近未来”的信息。对于 PyTorch 时间序列训练，同样的索引逻辑可以直接用于构造 `Subset`。

---

#### 2. `TSSplitter`（tsai / fastai 生态，高级）

**解释**：`tsai` 是构建在 PyTorch + fastai 之上的时间序列深度学习库，提供 `TSSplitter`（别名 `TimeSplitter`）专门生成保持时间顺序的 train/val 划分。`RandomSplitter` 则用于非时序场景的随机划分。如果已经在 tsai 的 `DataLoaders` 管道中，这是最自然的入口。

**传参数示例**：

```python
from tsai.data.validation import TSSplitter, RandomSplitter

# 时序划分：保持时间顺序
splitter = TSSplitter(valid_pct=0.2)

# 非时序随机划分
splitter = RandomSplitter(valid_pct=0.2, seed=42)

splits = splitter(items)
train_idx, valid_idx = splits[0], splits[1]
```

---

### 分布偏移感知的划分（高级）

#### 1. `DominoSlicer` + `GroupShuffleSplit`

**解释**：标准随机划分假设训练集和测试集**同分布**，但真实场景中测试分布可能发生偏移。`Domino` 等切片发现工具可以在划分之前识别出模型表现显著下降的**数据切片**，为划分提供“有意义的分组依据”。然后可以用这些切片标签驱动 `GroupShuffleSplit` 或 `StratifiedGroupKFold`，**确保每个脆弱切片在训练/验证中都有代表**，而不是盲目随机切分。这是把“分布偏移诊断”前移到划分阶段的进阶做法。

**传参数示例**：

```python
from domino import DominoSlicer
from sklearn.model_selection import GroupShuffleSplit

slicer = DominoSlicer(n_slices=5)
slicer.fit(embeddings, preds, labels)
slice_labels = slicer.predict(embeddings)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, groups=slice_labels))
```

---

### 高级集成：把划分逻辑收进管道

#### 1. `datasets.train_test_split`（Hugging Face）

**解释**：Hugging Face `datasets` 对象自带 `train_test_split` 方法，直接在 Arrow 表上完成划分，返回 `DatasetDict`，可以无缝接入 `Trainer` 或转换为 torch 格式。支持 `stratify_by_column` 做分层划分；`test_size` 可以是比例（浮点）或绝对数量（整数）。

**关键参数**：`test_size`，`train_size`，`stratify_by_column`，`seed`。

**传参数示例**：

```python
from datasets import load_dataset

ds = load_dataset("imdb", split="train")
split = ds.train_test_split(
    test_size=0.2,
    stratify_by_column="label",
    seed=42,
)
train_ds, test_ds = split["train"], split["test"]
```

需要交叉验证时，可以用 `shuffle(seed=...).select(indices)` 配合 sklearn 的 `KFold` 生成索引后手动 `select`。

---

### 划分策略速查与关键陷阱

| 数据特征 | 推荐基础 API | 推荐高级 API / 集成 | 核心约束 |
|---|---|---|---|
| 通用随机、无结构 | `random_split` / `train_test_split` | `LightningDataModule.setup()` | 固定种子、互不重叠 |
| 类别不平衡 | `train_test_split(stratify=y)` | `StratifiedKFold` | 每折类别比例一致 |
| 分组结构（患者/用户/文档） | `GroupShuffleSplit` / `GroupKFold` | `StratifiedGroupKFold` | 同组不跨折，**每个边界都检查** |
| 时间序列 | 按时间点手动切分 | `TimeSeriesSplit(gap=...)` / `TSSplitter` | 严禁 shuffle，设置禁运间隔 |
| 分布偏移诊断 | — | `DominoSlicer` 切片 + `GroupShuffleSplit` | 脆弱切片在训练/验证均有代表 |
| Hub 数据集快速划分 | `datasets.train_test_split` | 配合 `Trainer` | Arrow 层完成，直通训练 |
| 小数据（< 5,000） | 单次留出不够可靠 | `cross_validate`（5/10 折） | 报告均值与方差 |
| 大数据（> 100,000） | 单次 90/5/5 或 95/2.5/2.5 | — | 测试集比例可压缩 |

**四个最隐蔽的陷阱**：

1. **时间数据随机打乱**：这是最严重的错误，会静默地大幅虚高指标，应视为**阻断性问题**而非建议性提醒。  
2. **分组泄漏**：同一实体的样本跨越训练/验证边界，验证分数虚高。不仅要检查 train/test，**validation 边界也必须检查**。  
3. **预处理在划分之前全局拟合**：scaler、encoder、特征选择必须在**每折内部仅用训练数据**拟合，否则泄漏已经在划分之外发生了。  
4. **测试集被用于调参**：测试集只应在最后评估时触碰一次；调参和早停应使用验证集。

**划分的本质不是“切一刀”，而是“定义一个与数据生成结构一致的评估边界”。** 时间维度、分组结构、类别分布三者中任何一个被忽略，都会让后续所有的训练和调参建立在不可信的性能估计之上。先回答上面的决策树，再选择对应的 API，比默认调用 `random_split` 或 `train_test_split` 稳妥得多。

---

## 加载与预处理

数据预处理在 PyTorch 里没有单一的“万能入口”，而是按**数据模态**和**处理阶段**分散在几个包中。这一篇从 `Dataset` / `DataLoader` 出发，经各模态的变换，最后收束到可复用的数据管道。

### 数据加载容器：`Dataset` 与 `DataLoader`

这是所有预处理流程的骨架。预处理变换通常作为 `Dataset` 的参数注入，`DataLoader` 负责批量化与并行化。

#### 1. `Dataset`（自定义数据集）

**解释**：所有自定义数据集的基类。**映射式**必须实现 `__len__` 和 `__getitem__`，返回单个样本；**迭代式**继承 `IterableDataset` 并实现 `__iter__`，适合流式或随机读取成本极高的数据（数据库、远程日志）。

**传参数示例**：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = load_raw(self.paths[idx])          # 读取原始数据
        if self.transform:
            sample = self.transform(sample)          # 预处理在这里注入
        return sample, self.labels[idx]
```

---

#### 2. `DataLoader`

**解释**：把 `Dataset` 包装成可迭代的批次流，负责打乱、并行加载、锁页内存和批次整理。绝大多数预处理的实际执行位置就在这里（`num_workers > 0` 时在子进程中并行执行变换）。

**关键参数**：`batch_size`，`shuffle`（仅训练集开启），`num_workers`（子进程数），`collate_fn`，`pin_memory`，`drop_last`，`persistent_workers`，`prefetch_factor`。

**传参数示例**：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,              # 训练时打乱；验证/测试设为 False
    num_workers=4,             # 并行子进程，0 表示主进程加载
    pin_memory=True,           # 锁页内存，加速 CPU→GPU 传输
    drop_last=True,            # 丢弃最后一个不完整批次
    persistent_workers=True,   # 复用 worker 进程，避免每个 epoch 重建
    prefetch_factor=2,         # 每个 worker 预取批次数
)
```

`persistent_workers=True` 与 `num_workers > 0` 搭配，可以避免每个 epoch 结束后销毁并重建子进程的开销，在数据集较大时效果明显。

---

### 采样与批次整理

#### 1. `Sampler`

**解释**：控制**索引的产生顺序**。`DataLoader` 默认根据 `shuffle` 选择 `RandomSampler` 或 `SequentialSampler`；也可以传入自定义 `Sampler` 实现类平衡采样、课程学习等策略。

**传参数示例**：

```python
from torch.utils.data import WeightedRandomSampler

# 每个样本一个权重，权重与类别频率成反比
weights = 1.0 / torch.tensor(class_counts)[labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

使用 `sampler` 时不能再同时指定 `shuffle=True`。

---

#### 2. `BatchSampler`

**解释**：在 `Sampler` 之上，控制**每个批次包含哪些索引**。`DataLoader` 内部默认用 `BatchSampler` 把 `batch_size` 个索引捆成一批；自定义 `BatchSampler` 可以实现**动态批次大小**（如按序列长度分桶）。

**传参数示例**：

```python
from torch.utils.data import BatchSampler, SequentialSampler

batch_sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False,
)
loader = DataLoader(dataset, batch_sampler=batch_sampler)
```

---

#### 3. `collate_fn`

**解释**：把 `Dataset.__getitem__` 返回的**样本列表**整理成一个批次。默认的 `default_collate` 只能处理形状一致的张量；变长序列、多模态输入、需要批量级增强（如 MixUp）时，必须自定义。

**传参数示例（变长序列 padding）**：

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return sequences, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
```

如果 `collate_fn` 中需要执行 MixUp/CutMix，直接在返回前对整个批次施加即可。

---

### 图像与视频：`torchvision.transforms.v2`

`torchvision.transforms` 旧接口已逐步让位于 **`v2`**。`v2` 的核心优势是：支持任意输入结构（图像、视频、边界框、掩码、关键点），可以同时变换图像和标注，并且同时接受 PIL Image 和 Tensor（推荐 Tensor 后端以获得更好性能）。

#### 1. `v2.Compose`

**解释**：把多个变换串成流水线，依次执行。是所有图像预处理流程的入口。

**传参数示例**：

```python
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),      # uint8 [0,255] -> float [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
```

验证/推理时使用确定性的 `Resize` + `CenterCrop` + `Normalize`，不加随机增强。

---

#### 2. `v2.ToDtype` / `v2.ToImage` / `v2.ToPureTensor`

**解释**：`v2` 不再依赖单一的 `ToTensor`。`ToImage` 把 PIL/ndarray 转为 `tv_tensors.Image`；`ToDtype` 完成 dtype 转换和值域缩放；`ToPureTensor` 把 `tv_tensors` 剥回普通 `torch.Tensor`（在需要与不支持 `tv_tensors` 的代码交互时使用）。

**传参数示例**：

```python
tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),   # scale=True 同时做 [0,255]->[0,1]
])
img_tensor = tf(pil_image)                   # torch.Tensor, (C, H, W)
```

---

#### 3. `v2.Normalize`

**解释**：逐通道标准化，`output = (input - mean) / std`。图像通常使用 ImageNet 统计量；其他数据需根据自己的数据集计算。

**传参数示例**：

```python
norm = v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
normalized = norm(img_tensor)                # (3, H, W)
```

---

#### 4. 几何变换（`v2.Resize` / `RandomResizedCrop` / `RandomHorizontalFlip` 等）

**解释**：改变空间尺寸或位置。`v2` 的几何变换会**同步调整边界框、掩码和关键点**，这是旧版 `transforms` 不具备的关键能力。

**传参数示例**：

```python
geo_tf = v2.Compose([
    v2.Resize(size=(256, 256), antialias=True),
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
])
```

处理检测任务时，直接把 `(image, boxes)` 同时传入：

```python
img, boxes = geo_tf(img, boxes)              # 图像和框同步变换
```

---

#### 5. `v2.RandAugment` / `v2.AutoAugment` / `v2.TrivialAugmentWide`

**解释**：自动增强策略。`RandAugment` 随机选取 N 个操作并施加统一强度；`AutoAugment` 使用预定义策略（如 ImageNet 策略）；`TrivialAugmentWide` 每次只随机选一个操作，但强度也随机，通常鲁棒性很好。

**传参数示例**：

```python
strong_tf = v2.Compose([
    v2.RandAugment(num_ops=2, magnitude=9),   # 随机 2 个操作，强度 9
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])
```

---

#### 6. `v2.MixUp` / `v2.CutMix`

**解释**：**批次级**增强，必须在 `collate_fn` 或训练循环中对整个 batch 施加，而不是在 `Dataset` 中逐样本调用。`MixUp` 对两张图像和标签做线性混合；`CutMix` 把一张图像的矩形区域粘贴到另一张上，并按面积比例混合标签。

**传参数示例**：

```python
from torchvision.transforms import v2

cutmix = v2.CutMix(num_classes=10)
mixup  = v2.MixUp(num_classes=10)

for images, labels in loader:
    images, labels = cutmix(images, labels)   # 或 mixup，二选一
    # labels 变为软标签，损失函数需使用软标签交叉熵
```

使用 `MixUp`/`CutMix` 后，`CrossEntropyLoss` 的 `target` 不再是类别索引，而是与 `logits` 同形状的概率分布，需要相应调整损失调用方式。

---

### 变换能力的增强

#### 1. `torchvision.transforms.v2` 的自定义变换

**解释**：`v2` 的高级用法不止于使用内置变换，更在于**自定义变换与内置变换无缝组合**。任何继承 `nn.Module` 并重写 `forward` 的类都可以放进 `v2.Compose` 中，并且会自动获得 `v2` 对 `tv_tensors`（图像、边界框、掩码、关键点）的**同步变换语义**——当你的自定义变换修改了空间维度时，标注会跟着一起变，不需要手动处理。这是 `v2` 相对于旧版 `transforms` 最实质的能力升级。

**传参数示例**：

```python
import torch.nn as nn
from torchvision.transforms import v2
from torchvision import tv_tensors

class RandomErasing(nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        super().__init__()
        self.p, self.scale = p, scale

    def forward(self, img, boxes=None):
        # img 可能是 tv_tensors.Image，boxes 可能是 tv_tensors.BoundingBoxes
        # v2 会保证传入的标注与图像空间一致
        if torch.rand(1) > self.p:
            return img, boxes
        # ... 在 img 上随机擦除一块区域 ...
        return img, boxes

train_tf = v2.Compose([
    v2.RandomResizedCrop((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    RandomErasing(p=0.5),                  # 自定义变换直接混入
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

img, boxes = train_tf(img, boxes)           # 图像和标注同步变换
```

内置的 `v2.SanitizeBoundingBoxes` 可以在增强后清理因裁剪而失效的边界框，同样是 `v2` 专为标注数据设计的高阶组件。

---

### 文本：从 `torchtext` 迁移

**`torchtext` 已于 0.18.0 起停止维护，官方推荐用 Hugging Face `tokenizers` / `transformers` 替代**，或使用社区维护的 `touchtext` 作为轻量替代。但如果只是做**分词 + 词表 + 数值化**这条最小链路，仍可以用 `torchtext` 的现有组件理解流程（不依赖其已移除的 `Field` 遗留 API）。

#### 1. `get_tokenizer`

**解释**：获取分词器。`basic_english` 是内置的简单英文分词；多语言场景推荐 spaCy 分词器。

**传参数示例**：

```python
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("PyTorch is a deep learning framework")
# ['pytorch', 'is', 'a', 'deep', 'learning', 'framework']
```

---

#### 2. `Vocab`

**解释**：从 token 迭代器构建词表，把 token 映射为整数 ID，并处理 `<unk>`、`<pad>` 等特殊符号。

**传参数示例**：

```python
from collections import Counter
from torchtext.vocab import Vocab

counter = Counter()
for text in corpus:
    counter.update(tokenizer(text))

vocab = Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
ids = [vocab[token] for token in tokens]
```

文本预处理在 PyTorch 中通常由 `collate_fn` 完成：`Dataset` 返回原始字符串，`collate_fn` 负责分词、转 ID、padding 到批次内最大长度。

---

### 音频：`torchaudio`

#### 1. `torchaudio.load`

**解释**：读取音频文件，返回波形张量（`(channels, time)`）和采样率。

**传参数示例**：

```python
import torchaudio

waveform, sr = torchaudio.load("audio.wav")
# waveform: (1, T) 或 (2, T)，sr: int
```

---

#### 2. `torchaudio.transforms.Resample`

**解释**：重采样到目标采样率。如果数据集中的音频采样率不一致，需要统一。

**传参数示例**：

```python
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
waveform_16k = resampler(waveform)
```

---

#### 3. `torchaudio.transforms.MelSpectrogram`

**解释**：把一维波形转换为 Mel 频谱图，是语音任务最常用的声学特征。`n_mels` 控制 Mel 滤波器数量，`n_fft` 与 `hop_length` 共同决定时间-频率分辨率。

**传参数示例**：

```python
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
)
mel = mel_tf(waveform_16k)                   # (channels, n_mels, time)
```

---

### 预处理流水线的位置

把这些 API 串起来，一个典型流程的职责划分是：

```
Dataset.__getitem__       → 读取单样本，施加逐样本变换（Resize, Augment, Tokenize…）
        ↓
Sampler / BatchSampler     → 决定索引顺序和批次组成
        ↓
collate_fn                 → 批次整理（padding、转张量、MixUp/CutMix）
        ↓
DataLoader                 → 并行执行上述步骤，产出批次
```

**逐样本**的变换放在 `Dataset` 的 `transform` 参数中；**批次级**的操作（padding、MixUp/CutMix）放在 `collate_fn` 中；**顺序/组成**由 `Sampler`/`BatchSampler` 控制。图像用 `transforms.v2`，音频用 `torchaudio.transforms`，文本的分词和词表可以沿用 `torchtext` 的现有组件或迁移到 Hugging Face `tokenizers`，数值特征在 `Dataset` 中手动或借助 `sklearn` 完成标准化。

---

数据预处理的高级 API 可以理解为两个方向的延伸：**向上，把分散在 `Dataset`、`DataLoader`、变换和采样器中的逻辑组织成可复用、可共享的管道；向下，用 GPU 加速、流式格式或内存映射格式突破 CPU 瓶颈和磁盘 I/O 瓶颈。**

### 管道的组织与复用

#### 1. `LightningDataModule`

**解释**：把“数据下载 → 划分 → 变换 → 构建 DataLoader”这一整套逻辑收拢到一个类中，与模型代码彻底解耦。同一个 DataModule 可以被不同模型、不同实验复用，数据处理的逻辑只维护一份。核心方法按执行顺序分为 `prepare_data()`（下载/预处理，只在一个进程执行一次）、`setup()`（划分数据集、构建变换，每个进程执行）、`train_dataloader()` / `val_dataloader()` / `test_dataloader()`（返回 DataLoader）。

**传参数示例**：

```python
import lightning as L
from torch.utils.data import DataLoader, random_split

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32):
        super().__init__()
        self.data_dir, self.batch_size = data_dir, batch_size

    def prepare_data(self):
        # 只在一个进程执行，适合下载数据
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # 每个进程都会执行，适合构建变换和划分
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,)),
        ])
        full = MNIST(self.data_dir, train=True, transform=transform)
        self.train_ds, self.val_ds = random_split(full, [55000, 5000])
        self.test_ds = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

# 与 Trainer 配合：数据逻辑完全从训练脚本中剥离
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=MNISTDataModule())
```

`prepare_data` 和 `setup` 的分工是关键：**下载/写入磁盘的操作只应放在 `prepare_data`，而划分数据集、构建变换这类内存中的操作放在 `setup`**，否则在多卡训练时每个进程都会重复下载。

---

#### 2. `Hugging Face datasets` 的 `with_format("torch")`

**解释**：Hugging Face `datasets` 的 `Dataset` 对象**本身就是 PyTorch `Dataset` 的兼容替代品**，可以直接传给 `DataLoader`。真正的“高级”入口是 `with_format("torch")`：它把底层 Arrow 表中的数据在读取时**零拷贝**转换为 `torch.Tensor`，并支持直接指定 `device` 把张量放到 GPU 上，绕过 CPU 中转。对于已经在 Hub 上的大规模数据集，这是跳过“下载 → 本地解析 → 转张量”整条链路的最短路径。

**传参数示例**：

```python
from datasets import load_dataset

ds = load_dataset("imagenet-1k", split="train")
ds = ds.with_format("torch", columns=["image", "label"], device="cuda")

loader = DataLoader(ds, batch_size=32, num_workers=4)
```

对于形状固定的 N 维数组，显式声明 `Array2D(shape=..., dtype=...)` 等特征类型可以避免逐样本形状比较带来的额外开销。

---

### 加载性能与规模

#### 1. `NVIDIA DALI`：`pipeline_def` + `DALIGenericIterator`

**解释**：把解码、缩放、裁剪、归一化等 CPU 密集操作搬到 **GPU** 上执行，消除 CPU 预处理瓶颈。DALI 用 `@pipeline_def` 声明一条算子图，用 `DALIGenericIterator` 把它包装成与 `DataLoader` 语义一致的迭代器，训练循环几乎不需要改动。最新版本还提供了 `DALI Dynamic`（`nvidia.dali.experimental.dynamic`）和 **DALI Proxy**，后者可以直接替换 `DataLoader` 而保留原有 `Dataset` 逻辑，把 GPU 预处理异步桥接进来。DALI 在大规模图像数据上的加载速度可比原生 `DataLoader` 快数倍。

**传参数示例（`DALIGenericIterator`）**：

```python
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def image_pipeline(data_dir, batch_size):
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    images = fn.decoders.image(jpegs, device="mixed")     # GPU 解码
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels

pipe = image_pipeline(data_dir="/path/to/images", batch_size=64,
                      num_threads=4, device_id=0)
pipe.build()

loader = DALIGenericIterator(pipe, ["images", "labels"])

for batch in loader:
    images = batch[0]["images"]      # 已在 GPU 上，torch.Tensor
    labels = batch[0]["labels"]
```

---

#### 2. `FFCV`：`DatasetWriter` + `Loader`

**解释**：FFCV 的核心思路是**把数据预先转换成高度优化的 `.beton` 格式**，利用内存映射和批量化的页对齐读取，在训练时用 `ffcv.loader.Loader` 直接替代 `torch.utils.data.DataLoader`，训练代码其余部分**不需要改动**。它把“格式转换”放在一次性离线步骤中，训练时只做极轻量的读取和 GPU 传输，因此在视觉任务中常常能显著提升吞吐。

**传参数示例**：

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice

# 1. 离线转换：torchvision Dataset -> .beton
writer = DatasetWriter("/tmp/train.beton", {
    "image": RGBImageField(write_mode="jpg", max_resolution=256),
    "label": IntField(),
})
writer.from_indexed_dataset(train_dataset)

# 2. 训练时替换 DataLoader
loader = Loader(
    "/tmp/train.beton",
    batch_size=64,
    num_workers=8,
    order=OrderOption.RANDOM,
    os_cache=True,
    pipelines={
        "image": [ToTensor(), ToDevice(0)],
        "label": [ToTensor(), ToDevice(0)],
    },
)
for images, labels in loader:
    ...
```

---

#### 3. `WebDataset`：`wds.WebDataset` + `.map` / `.batched`

**解释**：把大规模数据集打包成 **tar 分片**，每个分片内按“相同 basename + 不同扩展名”组织样本（如 `0001.jpg`、`0001.label`）。`wds.WebDataset` 以**流式**方式读取，不需要把全部索引加载到内存，适合数据量远超本地磁盘、或存放在对象存储上的场景。它可以直接把 URL 模板传给 `WebDataset`，自动展开为分片列表，并支持 `resampled=True` 在流式模式下循环采样。在基准中，WebDataset 在“准备成本”和“运行时性能”之间取得了较好的平衡。

**传参数示例**：

```python
import webdataset as wds
from torch.utils.data import DataLoader

URL = "https://hf.co/datasets/.../shard-{0000..0034}.tar"

dataset = (
    wds.WebDataset(URL, resampled=True, shardshuffle=True)
    .decode("pil")
    .map(lambda sample: (transform(sample["jpg"]), sample["cls"]))
    .batched(32)
)

loader = DataLoader(dataset, batch_size=None, num_workers=4)
# batch_size=None 是因为 batched 已在 WebDataset 内部完成
```

---

### 高级数据 API 的定位对照

| 层级 | API / 工具 | 它替代或增强的底层组件 | 核心解决的问题 |
|---|---|---|---|
| **管道组织** | `LightningDataModule` | 手写的 `Dataset` + `DataLoader` 组装逻辑 | 数据代码与模型代码解耦、跨实验复用 |
| **管道组织** | HF `datasets` + `with_format` | 自定义 `Dataset` 的读取与转张量步骤 | 零拷贝 Arrow 读取、Hub 数据集直用 |
| **变换增强** | `v2` 自定义 `nn.Module` | 旧版 `transforms` 的逐样本函数 | 多模态标注同步变换、与内置变换自由组合 |
| **性能加速** | NVIDIA DALI | CPU 上的 `num_workers` 预处理 | 把解码/增强搬到 GPU，消除 CPU 瓶颈 |
| **格式加速** | FFCV | `torch.utils.data.DataLoader` | 离线转换为页对齐 `.beton`，训练时轻量读取 |
| **规模扩展** | WebDataset | 基于索引的映射式 `Dataset` | tar 分片流式读取，支持对象存储/超大语料 |

这些高级 API 没有改变“样本如何被读取、变换、整理成批次”的语义，它们改变的是**管道放在哪里、由谁调度、以及瓶颈被消除在哪一层**。日常原型仍然可以从 `Dataset` + `DataLoader` + `v2` 起步；当数据规模、I/O 成本或实验管理复杂度成为主要矛盾时，再按上表把对应层级替换成高级方案即可。

---

## 特征工程

特征工程在深度学习时代的位置变了，但没有消失。对于图像、音频、原始文本这类**非结构化数据**，特征提取已经基本被端到端学习取代——卷积层和注意力层在做的事，本质上就是自动学习层次化特征。但对于**表格数据、时间序列、高基数类别、业务聚合特征**，手工构造的特征仍然经常比“把原始列直接丢进 MLP”更有效，因为树模型和线性模型无法自动发现比率、滞后、周期、分组统计这类结构化先验。

### 数值特征：缩放、变换、分箱、交互

#### 1. `StandardScaler` / `MinMaxScaler` / `RobustScaler`

**解释**：三种缩放器各有适用场景。`StandardScaler` 适合近似正态分布的数据，是最常见的选择；`MinMaxScaler` 将值压缩到 `[0,1]`，适合对输入范围敏感的模型（如神经网络、KNN）；`RobustScaler` 基于中位数和四分位距而非均值，在存在明显异常值时更稳健。**缩放器只能在训练集上 `fit`**，在完整数据上拟合会引入信息泄漏。

**传参数示例**：

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[["income", "age"]])
X_test_scaled  = scaler.transform(X_test[["income", "age"]])   # 不再 fit

# 异常值较多时
robust = RobustScaler()
X_train_rob = robust.fit_transform(X_train[["salary"]])
```

---

#### 2. `np.log1p`

**解释**：右偏分布（收入、价格、营收、计数）的对数变换，把乘法关系拉成加法关系，压缩尾部。`log1p` 可以安全处理零值。它不保证变成正态分布，主要作用是降低偏度。

**传参数示例**：

```python
import numpy as np
df["revenue_log"] = np.log1p(df["revenue"])
```

---

#### 3. `pd.cut` / `pd.qcut` / `KBinsDiscretizer`

**解释**：分箱把连续变量转为离散类别，用于简化非线性关系、降噪，或让线性模型能够分段拟合。`pd.cut` 生成**等宽**分箱，适合分布均匀的数据；`pd.qcut` 按**分位数**切分，每个箱样本量相等，更适合偏斜分布。`KBinsDiscretizer` 是 sklearn 中可放进 `Pipeline` 的版本，支持 `uniform`、`quantile`、`kmeans` 三种策略。

**传参数示例**：

```python
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

df["age_group"] = pd.cut(
    df["age"], bins=[0, 18, 35, 55, 100],
    labels=["teen", "young_adult", "adult", "senior"],
)

# 可放进 Pipeline 的版本
kbd = KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile")
X_binned = kbd.fit_transform(X[["income"]])
```

---

#### 4. `PolynomialFeatures`

**解释**：生成多项式项和交互项，让线性模型能够捕获非线性关系。`degree=2` 会生成 `x1, x2, x1², x1·x2, x2²`。**特征维度会快速膨胀**，通常只对少量重要特征使用，或配合 `interaction_only=True` 只生成交互项。

**传参数示例**：

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    interaction_only=False,     # True 则只生成交互项，不含平方项
    include_bias=False,
)
X_poly = poly.fit_transform(X[["age", "salary"]])
```

---

#### 5. 交互与比率特征（领域驱动）

**解释**：两个特征的比率或差值往往比各自单独使用更有信息量——`price_per_sqft`、`debt_to_income`、`clicks/impressions` 都是典型例子。这类特征没有通用 API，需要根据领域知识手动构造。

**传参数示例**：

```python
df["price_per_sqft"] = df["price"] / df["sqft"]
df["debt_to_income"] = df["debt"] / df["income"].replace(0, np.nan)
```

---

#### 6. 异常值裁剪

**解释**：与其删除异常值（可能丢失有用信息），不如将极端值截断到合理的百分位范围（如 1% 和 99%）。这对线性模型和基于距离的方法尤其重要。

**传参数示例**：

```python
lower = df["salary"].quantile(0.01)
upper = df["salary"].quantile(0.99)
df["salary_clipped"] = df["salary"].clip(lower=lower, upper=upper)
```

---

PyTorch 没有像 `sklearn` 那样的 `StandardScaler` 类。**图像的标准化用 `v2.Normalize`**；**表格/数值特征的标准化通常有两种做法**：在 `Dataset` 中手动计算，或借助 `sklearn` 预处理后再转为张量。

#### 7. 手动实现 `StandardScaler`（作为 `nn.Module`）

**解释**：把均值和标准差注册为 buffer，`forward` 中执行 `(x - mean) / std`。`buffer` 不会被视为可训练参数，但会随 `state_dict` 保存和加载，适合在推理时复用训练集统计量。

**传参数示例**：

```python
import torch
import torch.nn as nn

class StandardScaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.as_tensor(std,  dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

scaler = StandardScaler(mean=train_mean, std=train_std)
x_scaled = scaler(x_raw)
```

如果特征维度少、数据量不大，直接用 `sklearn.preprocessing.StandardScaler` 拟合后再 `torch.tensor(...)` 也是完全可行的路径。

---

### 类别特征：编码策略的选择取决于基数

#### 1. `OneHotEncoder` / `pd.get_dummies`

**解释**：将名义类别展开为独立的 0/1 列，适合**基数较低**（通常 < 20–50）且无内在顺序的类别。高基数列使用独热编码会造成维度爆炸，应改用目标编码或频率编码。`drop_first=True` 可以避免虚拟变量陷阱（对线性模型有必要，对树模型不必要）。

**传参数示例**：

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(
    handle_unknown="ignore",    # 推理时遇到训练集未见过的类别，不报错
    sparse_output=False,
)
X_train_ohe = ohe.fit_transform(X_train[["city"]])
X_test_ohe  = ohe.transform(X_test[["city"]])
```

---

#### 2. `LabelEncoder` / 有序映射

**解释**：为每个类别赋一个整数。**只适用于存在内在顺序的类别**（如教育程度：高中 < 本科 < 硕士 < 博士）。对城市名等名义数据做标签编码，会让模型错误地推断“London > Mumbai”。

**传参数示例**：

```python
df["education"] = df["education"].map({
    "High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3,
})
```

---

#### 3. `TargetEncoder`（sklearn）

**解释**：用类别的目标均值替换类别值，是处理**高基数类别**最有效的方法之一。核心风险是**目标泄漏**：如果在训练集上直接 `fit().transform()`，每个样本的编码都包含了它自己的目标值，模型会过拟合。sklearn 的 `TargetEncoder` 在 `fit_transform` 内部使用**交叉拟合**（K-Fold out-of-fold）来缓解这个问题，但 `transform()` 在推理时使用的是全量训练数据的统计量。**正确用法是 `fit_transform(X_train, y_train)` 用于训练集，`transform(X_test)` 用于验证/测试集**；不要对同一份训练数据先 `fit` 再 `transform`。

**传参数示例**：

```python
from sklearn.preprocessing import TargetEncoder

enc = TargetEncoder(
    smooth="auto",          # 平滑，避免低频类别的极端估计
    cv=5,                   # 内部交叉拟合折数
    random_state=42,
)
X_train_enc = enc.fit_transform(X_train[["city"]], y_train)
X_test_enc  = enc.transform(X_test[["city"]])
```

如果数据存在**分组结构**（同一患者的重复测量），交叉拟合的折也需要按组划分，否则仍会泄漏；sklearn 较新版本已支持向 `TargetEncoder` 传入 `groups` 参数。

---

#### 4. `category_encoders` 的 `BinaryEncoder`

**解释**：介于标签编码和独热编码之间的折中方案。每个类别先被赋予一个整数，再将该整数转换为二进制位，用多个二进制列表示。100 个类别只需约 7 列，适合中等基数、希望控制维度但又不想完全丢失类别区分度的场景。

**传参数示例**：

```python
from category_encoders import BinaryEncoder

be = BinaryEncoder()
df_encoded = be.fit_transform(df[["city"]])
```

---

#### 5. 频率编码

**解释**：用每个类别的出现频率替换原始值。实现极简，但在树模型中效果常常出人意料，且不依赖目标变量，没有泄漏风险。

**传参数示例**：

```python
freq_map = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq_map)
```

---

### 日期时间特征

#### 1. 标准提取与周期编码

**解释**：原始日期对大多数模型没有意义，需要提取年、月、日、星期、是否周末、季度、距参考日期的天数等。**周期特征（月份、小时、星期）直接用数字编码会丢失循环结构**——十二月（12）和一月（1）在数值上相距很远，但实际上相邻。正弦/余弦编码保留这种周期结构：`month_sin = sin(2π·month/12)`，`month_cos = cos(2π·month/12)`。

**传参数示例**：

```python
df["order_date"] = pd.to_datetime(df["order_date"])
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

节假日、月末、月初等日历特征对业务预测往往有额外信息量，可以通过 `holidays` 库或 pandas 的日期属性补充。

---

### 特征选择：先构造，再筛选

构造特征只是工作的一半；大量无关或冗余特征会拖慢训练、增加过拟合风险。三类方法的选择取决于数据规模和对可解释性的要求。

#### 1. 过滤法（Filter）

**解释**：用统计指标独立评估每个特征与目标的关系，不涉及模型训练，速度快，适合作为第一道粗筛。`VarianceThreshold` 删除方差过低的特征（几乎不变化的列提供不了信息）；`SelectKBest` 配合 `f_classif`（ANOVA F 值）、`chi2`（卡方，要求非负特征）、`mutual_info_classif`（互信息，能捕获非线性关系）选择 top-K。

**传参数示例**：

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

# 删除低方差特征
vt = VarianceThreshold(threshold=0.01)
X_reduced = vt.fit_transform(X)

# 互信息选择 top-K
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X, y)
```

---

#### 2. 包装法（Wrapper）：`RFE`

**解释**：用模型的表现作为评价标准，递归地训练模型、剔除最不重要的特征、再训练，直到达到目标特征数。比过滤法更准确，但计算成本更高，且**必须在每一折训练内部重新执行**，不能在全量数据上先做 RFE 再交叉验证。

**传参数示例**：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(
    estimator=LogisticRegression(max_iter=1000),
    n_features_to_select=20,
    step=1,                     # 每轮剔除的特征数
)
X_rfe = rfe.fit_transform(X_train, y_train)
```

---

#### 3. 嵌入法（Embedded）：`SelectFromModel`

**解释**：利用模型训练过程中产生的特征重要性（L1 正则化的稀疏系数、树模型的特征重要性）来筛选特征。一次训练完成，效率高于 RFE，是实践中常用的折中方案。

**传参数示例**：

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

selector = SelectFromModel(rf, threshold="mean", prefit=True)
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)
```

`threshold="mean"` 保留重要性高于平均值的特征；也可以传入具体数值或 `"median"`。

---

### 文本特征：从 TF-IDF 到上下文嵌入

#### 1. `TfidfVectorizer`

**解释**：将文本转为按词项重要性加权的稀疏矩阵。`max_features` 限制词表规模，`ngram_range=(1,2)` 同时捕获单词和二元短语，`stop_words` 去除高频虚词。它是**词袋模型**，忽略词序和上下文，但在中小规模分类任务中仍然是强基线。

**传参数示例**：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,                   # 忽略出现少于 2 篇文档的词
    stop_words="english",
)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf  = tfidf.transform(test_texts)
```

---

#### 2. `SentenceTransformer.encode`

**解释**：用预训练的 Sentence-BERT 类模型将文本压缩为**稠密语义向量**，捕获上下文和语义相似性，表达能力比 TF-IDF 高出一个量级。`all-MiniLM-L6-v2` 是常用的轻量模型，输出 384 维向量。这是深度学习时代文本特征提取的首选入口，产出的嵌入可以直接作为下游分类器或回归器的输入特征。

**传参数示例**：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    df["review"].tolist(),
    batch_size=64,
    show_progress_bar=True,
)
# embeddings.shape: (n_rows, 384)
```

对于短文本，还可以叠加基础统计特征（词数、平均词长、大写比例、是否含问号等），这些廉价特征在实际任务中经常提供额外信号。

---

### 图像特征：手工算子与 CNN 嵌入

#### 1. 传统算子（SIFT / HOG）

**解释**：在深度学习之前，图像特征提取依赖手工设计的局部算子（SIFT、HOG）配合词袋模型或 SVM。`skimage` 和 OpenCV 提供了这些实现。如今在大多数场景中已被 CNN 取代，但在**数据极少、需要强先验、或计算资源严格受限**的嵌入式场景中仍有价值。

**传参数示例（HOG，skimage）**：

```python
from skimage.feature import hog

fd = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=False,
)
```

---

#### 2. CNN / ViT 倒数第二层嵌入

**解释**：用预训练模型（ResNet、ViT 等）的**倒数第二层输出**作为图像的特征嵌入，是深度学习时代标准的“特征提取器”用法。不需要微调，直接对嵌入做线性探针或聚类，就能获得很强的表示。`timm`（PyTorch Image Models）提供了统一的入口，一行代码输出嵌入向量。

**传参数示例（timm）**：

```python
import timm
import torch

model = timm.create_model("resnet50", pretrained=True, num_classes=0)  # 去掉分类头
model.eval()

with torch.no_grad():
    emb = model(batch_images)    # (batch, 2048)
```

对于多模态场景，CLIP 类模型可以同时产出对齐的图像和文本嵌入，是跨模态特征工程的重要工具。

---

### 时间序列特征：统计描述子与大规模自动提取

#### 1. `tsfresh.extract_features`

**解释**：为每条时间序列自动生成**数百个统计特征**——均值、标准差、偏度、峰度、FFT 系数、熵、自相关、峰值数等（完整集合超过 700 个）。`select_features` 基于 FRESH 算法（可扩展假设检验）筛选与目标显著相关的特征，并控制假发现率。适合**中等规模**的时间序列分类/回归，而不是端到端深度学习。

**传参数示例**：

```python
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

features = extract_features(
    df_long,                    # 长表：id, time, value
    column_id="id",
    column_sort="time",
    column_value="value",
    n_jobs=4,
)
impute(features)               # 填充无法计算的特征（如全常数序列的偏度）
features_selected = select_features(features, y)
```

---

#### 2. `Catch22`（aeon）

**解释**：Catch22 是一组精心筛选的 **22 个规范时间序列特征**，在保持强判别力的同时大幅降低维度，计算效率远高于 tsfresh 的全量提取。`aeon` 提供了 sklearn 兼容的 `Catch22` 变换器，适合需要快速基线或大规模数据集的场景。

**传参数示例**：

```python
from aeon.transformations.collection.feature_based import Catch22

catch22 = Catch22(n_jobs=-1)
X_features = catch22.fit_transform(X_train)   # X_train: (n_cases, n_channels, n_timepoints)
```

---

### 聚合与滞后特征：业务场景中信息量最高的手工特征

#### 1. `groupby().agg()` + `merge`

**解释**：对每个实体（客户、用户、设备）计算历史统计量——总订单数、总消费、平均客单价、最大单笔——再把聚合结果合并回原表。这类特征在风控、推荐、流失预测中往往是最强的信号来源之一，因为单条记录本身的信息量远小于实体的历史行为模式。

**传参数示例**：

```python
stats = (
    df.groupby("customer_id")
    .agg(
        total_orders=("order_id", "count"),
        total_spent=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        max_order=("amount", "max"),
    )
    .reset_index()
)
df = df.merge(stats, on="customer_id", how="left")
```

---

#### 2. `shift` / `rolling`（滞后与滚动窗口）

**解释**：在时间序列或面板数据中，过去 N 个时间段的取值和滚动统计量（滚动均值、滚动求和）是最直接的时序信号。必须**先按实体和时间排序**，再分组执行，否则 `shift` 和 `rolling` 会跨实体泄漏。

**传参数示例**：

```python
df = df.sort_values(["customer_id", "order_date"])

df["prev_order_amount"] = df.groupby("customer_id")["amount"].shift(1)
df["amount_change"] = df["amount"] - df["prev_order_amount"]

df["rolling_3_spend"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda s: s.rolling(3, min_periods=1).sum())
)
```

---

### 自动化特征工程（高级）

#### 1. `Featuretools.dfs`：Deep Feature Synthesis

**解释**：面向**关系型多表数据**的自动特征生成。你定义实体集（EntitySet）——多张表以及表之间的主键/外键关系——DFS 自动在关系图上执行聚合（count、sum、mean、std 等）和变换（year、month、is_weekend 等），生成大量候选特征，并支持 `max_depth` 控制多层关系上的递归派生。

**传参数示例**：

```python
import featuretools as ft

es = ft.EntitySet(id="ecommerce")
es.add_dataframe(dataframe_name="customers", dataframe=df_customers, index="customer_id")
es.add_dataframe(dataframe_name="orders", dataframe=df_orders, index="order_id",
                 time_index="order_date")
es.add_relationship("customers", "customer_id", "orders", "customer_id")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    max_depth=2,
    agg_primitives=["count", "sum", "mean", "max", "min", "std", "n_unique"],
    trans_primitives=["year", "month", "day", "weekday", "is_weekend"],
)
```

生成的 `feature_matrix` 可能包含数十到数百列，后续需要配合特征选择使用。

---

#### 2. `AutoFeatClassifier` / `AutoFeatRegressor`

**解释**：AutoFeat 在给定基础特征上自动生成多项式、比率、对数等非线性变换，并**同时进行特征选择**，输出的特征数量通常远小于暴力多项式展开。`feateng_steps` 控制派生深度，`max_gb` 限制内存使用，适合中小规模表格数据的自动化特征构造。

**传参数示例**：

```python
from autofeat import AutoFeatClassifier

afc = AutoFeatClassifier(
    feateng_steps=2,
    max_gb=1,
    verbose=1,
    n_jobs=4,
)
X_train_new = afc.fit_transform(X_train, y_train)
X_test_new  = afc.transform(X_test)
```

---

#### 3. AutoML 框架中的特征工程

`H2O AutoML`、`Auto-sklearn`、`TPOT` 等在搜索模型和超参数的同时，也会执行一定程度的自动特征预处理和选择。它们的定位是**端到端的流程自动化**，而不是单独的特征工程工具；如果你的目标是快速获得强基线，可以直接使用这些框架，但代价是对特征构造过程的**可解释性和精细控制下降**。

---

### 深度学习时代的表格特征：让模型学习交叉，而非手工枚举

传统表格特征工程的核心痛点之一是**特征交叉**——两个类别特征的组合（如“城市 × 品类”）可能很有信息量，但手工枚举组合在基数高时不可行。深度学习表格模型通过**嵌入 + 注意力/交叉网络**来学习高阶特征交互，部分替代了手工交叉特征的角色。

#### 1. `pytorch_tabular` / `mambular` 中的 `FTTransformer` / `AutoInt`

**解释**：`FTTransformer`（Feature Tokenizer + Transformer）把每个特征（数值或类别）映射为嵌入 token，用 Transformer 编码器学习特征之间的交互；`AutoInt` 通过自注意力机制自动学习特征交互。这些模型的输入仍然是**原始特征列**，类别特征需要嵌入，数值特征通常做标准化或分箱嵌入，但**不需要手工构造交叉项**。`pytorch_tabular` 和 `mambular` 提供了这些架构的统一 API，并与 PyTorch Lightning 集成。

**传参数示例（pytorch_tabular 概念示意）**：

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig

data_config = DataConfig(
    target=["target"],
    continuous_cols=["age", "income"],
    categorical_cols=["city", "product"],
)

model_config = FTTransformerConfig(
    task="classification",
    embedding_dim=32,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=TrainerConfig(max_epochs=50),
)
tabular_model.fit(train=train_df, validation=val_df)
```

这类模型并不意味着手工特征工程完全无用——**聚合、滞后、领域比率特征**仍然是嵌入层无法从单行原始数据中恢复的；它们的价值在于自动处理**同一行内特征之间的交互**。

---

### 特征工程 API 的层级对照与关键原则

| 层级 | 代表 API / 工具 | 核心作用 | 是否自动学习表示 |
|---|---|---|---|
| **数值变换** | `StandardScaler`、`log1p`、`KBinsDiscretizer`、`PolynomialFeatures` | 调整分布、范围、非线性 | 否 |
| **类别编码** | `OneHotEncoder`、`TargetEncoder`、`BinaryEncoder`、频率编码 | 将类别转为数值 | 否 |
| **时间特征** | `dt` 属性、`sin/cos` 周期编码、`holidays` | 提取时间结构与周期 | 否 |
| **特征选择** | `VarianceThreshold`、`SelectKBest`、`RFE`、`SelectFromModel` | 去掉冗余、降低维度 | 否 |
| **文本传统** | `TfidfVectorizer` | 词袋加权表示 | 否 |
| **文本嵌入** | `SentenceTransformer.encode` | 上下文语义向量 | 是（预训练） |
| **图像嵌入** | `timm`（`num_classes=0`） | CNN/ViT 特征 | 是（预训练） |
| **时序统计** | `tsfresh.extract_features`、`Catch22` | 数百/22 个统计描述子 | 否 |
| **业务聚合** | `groupby().agg()`、`shift`、`rolling` | 实体级历史统计 | 否 |
| **关系型自动** | `Featuretools.dfs` | 多表自动聚合/变换 | 部分（规则驱动） |
| **表格深度模型** | `pytorch_tabular`（FTTransformer / AutoInt） | 学习特征交互 | 是（端到端） |

**四个关键原则**：

1. **先划分，再拟合**。所有有状态的变换器（scaler、encoder、`TargetEncoder`、`SelectKBest`、`PolynomialFeatures`）都必须在**训练集上 `fit`**，然后对验证/测试集只 `transform`。在完整数据上拟合是特征工程中最常见的泄漏来源。
2. **目标编码是泄漏高发区**。sklearn 的 `TargetEncoder.fit_transform` 内部做交叉拟合，但 `transform` 在推理时使用全量训练统计量；**不要对训练数据先 `fit` 再 `transform`**，也不要在交叉验证的折外先做目标编码再分折。
3. **聚合和滞后特征必须先排序、再分组**。`groupby().shift()` 和 `rolling()` 如果顺序错误，会跨实体或跨时间泄漏。
4. **深度学习不等于不需要特征工程**。对于表格数据，深度模型可以学习行内交互，但**跨行聚合、时间滞后、领域比率**仍然需要手工构造；对于文本和图像，预训练嵌入是“特征提取”的现代形态，而不是特征工程的替代品。特征工程的目标从“手工设计每一个输入”转向**为模型提供它无法从原始数据中自行恢复的结构化先验**。

---

## 模型构建

用 `nn.Module` 把数据流写下来：容器定结构，层定计算，激活定非线性，初始化定起点。最后一节讨论在任务约束下怎么选模型。

### 模型容器

#### 1. `nn.Module`

**解释**：所有神经网络模块的基类。你在 `__init__` 中赋值的 `nn.Linear`、`nn.Conv2d` 等子模块会被自动注册；`forward` 定义计算流，`backward` 由 autograd 自动完成。`parameters()` 遍历可训练参数，`state_dict()` 用于保存和加载权重。

**传参数示例**：

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                          # 必须调用
        self.fc1 = nn.Linear(in_dim, hidden)        # 自动注册
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = Net(784, 256, 10)
list(model.parameters())        # 两个 Linear 的 weight 和 bias
model.state_dict()              # 用于 torch.save
```

---

#### 2. `nn.Sequential`

**解释**：按顺序链接模块，前一个的输出直接作为后一个的输入，是构建前馈网络最简洁的方式。适合“线性堆叠、不需要在层之间插入自定义逻辑”的结构。

**传参数示例**：

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
```

也支持用 `OrderedDict` 给每层命名，便于调试时定位：

```python
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 256)),
    ("act", nn.ReLU()),
    ("fc2", nn.Linear(256, 10)),
]))
```

---

#### 3. `nn.ModuleList`

**解释**：把子模块存进一个**列表**，支持 `append`、`extend` 和索引，但**不会自动执行前向传播**——必须在 `forward` 中显式遍历调用。适合需要循环堆叠 N 个相同结构的场景。

**传参数示例**：

```python
class Stack(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_layers)
        ])

    def forward(self, x):
        for blk in self.blocks:          # 必须自己写循环
            x = torch.relu(blk(x))
        return x
```

---

#### 4. `nn.ModuleDict`

**解释**：把子模块存进**字典**，通过 key 访问，同样不会自动前向传播。适合存在可选分支、或多条并行通路的模型（如多任务头）。

**传参数示例**：

```python
class MultiHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.heads = nn.ModuleDict({
            f"task{i}": nn.Linear(in_dim, n_classes) for i in range(3)
        })

    def forward(self, x, task):
        return self.heads[task](x)       # 按名字选头
```

---

### 基础层

#### 1. `nn.Linear`

**解释**：全连接层，计算 \( y = xW^T + b \)。输入形状 `(..., in_features)`，输出形状 `(..., out_features)`。

**参数**：`in_features`，`out_features`，`bias`（默认 `True`）。

**传参数示例**：

```python
fc = nn.Linear(in_features=784, out_features=256, bias=True)
x = torch.randn(32, 784)
out = fc(x)                              # (32, 256)
```

---

#### 2. `nn.Conv2d`

**解释**：二维卷积层，在 `(N, C_in, H, W)` 的输入上滑动卷积核提取局部特征。输出形状 `(N, C_out, H_out, W_out)`。

**关键参数**：`in_channels`，`out_channels`（卷积核个数），`kernel_size`，`stride`（默认 1），`padding`（默认 0），`dilation`（默认 1）。

**传参数示例**：

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,          # padding=1 + k=3 时空间尺寸不变
)
x = torch.randn(16, 3, 224, 224)
out = conv(x)            # (16, 64, 224, 224)
```

---

#### 3. `nn.MaxPool2d`

**解释**：最大池化，对每个窗口取最大值，降低空间分辨率。默认 `stride = kernel_size`。

**传参数示例**：

```python
pool = nn.MaxPool2d(kernel_size=2)       # H, W 各减半
x = torch.randn(16, 64, 224, 224)
out = pool(x)                             # (16, 64, 112, 112)
```

---

#### 4. `nn.AdaptiveAvgPool2d`

**解释**：自适应平均池化，指定**输出空间尺寸**，自动适配任意输入尺寸。常用于分类头前统一到 `1×1`。

**传参数示例**：

```python
gap = nn.AdaptiveAvgPool2d((1, 1))
x = torch.randn(16, 512, 7, 7)
out = gap(x)                              # (16, 512, 1, 1)
```

---

#### 5. `nn.BatchNorm2d`

**解释**：对 `(N, C, H, W)` 的二维特征图做批归一化，在 `N, H, W` 上统计均值和方差，按通道归一化，加速训练并提升稳定性。

**关键参数**：`num_features`（通道数），`eps`（默认 `1e-5`），`momentum`（默认 `0.1`）。

**传参数示例**：

```python
bn = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1)
x = torch.randn(16, 64, 56, 56)
out = bn(x)
```

---

#### 6. `nn.LayerNorm`

**解释**：对最后若干维做归一化，不依赖 batch 维度，常用于 Transformer。输入最后一维必须等于 `normalized_shape` 指定的形状。

**传参数示例**：

```python
ln = nn.LayerNorm(normalized_shape=128)
x = torch.randn(8, 50, 128)               # (batch, seq_len, hidden)
out = ln(x)                               # 对最后一维 128 归一化
```

---

#### 7. `nn.Dropout`

**解释**：训练时以概率 `p` 随机将元素置零，推理时关闭（`model.eval()` 后失效），用于缓解过拟合。

**传参数示例**：

```python
drop = nn.Dropout(p=0.5)
x = torch.randn(32, 256)
out_train = drop(x)                       # model.train() 状态下随机置零
model.eval()
out_eval = drop(x)                        # eval 状态下原样返回
```

---

#### 8. `nn.Embedding`

**解释**：将离散索引映射为稠密向量。输入形状 `(..., seq_len)` 的整数张量，输出在最后追加一维 `embedding_dim`。

**传参数示例**：

```python
emb = nn.Embedding(num_embeddings=10000, embedding_dim=256)
idx = torch.randint(0, 10000, (32, 50))   # (batch, seq_len)
out = emb(idx)                             # (32, 50, 256)
```

---

#### 9. `nn.LSTM`

**解释**：长短期记忆网络，用于序列建模。`batch_first=True` 时输入形状为 `(batch, seq_len, input_size)`，比默认的 `(seq_len, batch, input_size)` 更符合直觉。

**传参数示例**：

```python
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.1,             # 仅 num_layers > 1 时生效
)
x = torch.randn(32, 50, 128)              # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
# output: (32, 50, 256)
```

---

### 激活函数

#### 1. `nn.ReLU` / `F.relu`

**解释**：\(\max(0, x)\)。模块形式适合放入 `Sequential`，函数形式适合在 `forward` 中直接调用。

**传参数示例**：

```python
relu = nn.ReLU(inplace=False)             # inplace=True 会直接修改输入
x = torch.randn(4, 8)
out = relu(x)

import torch.nn.functional as F
out = F.relu(x)                            # 函数形式
```

---

#### 2. `nn.LeakyReLU`

**解释**：负半轴给一个小斜率，缓解 ReLU 的“神经元死亡”问题。

**传参数示例**：

```python
act = nn.LeakyReLU(negative_slope=0.01)
out = act(torch.randn(4, 8))
```

---

#### 3. `nn.GELU`

**解释**：高斯误差线性单元，平滑版 ReLU，Transformer 系列模型（BERT、GPT）的默认选择。

**传参数示例**：

```python
act = nn.GELU()
out = act(torch.randn(4, 8))

# 或函数形式
out = F.gelu(x)
```

---

#### 4. `nn.SiLU`

**解释**：\(x \cdot \text{sigmoid}(x)\)，也称 Swish，常见于现代 CNN（EfficientNet 等）和门控结构。

**传参数示例**：

```python
act = nn.SiLU()
out = act(torch.randn(4, 8))
```

---

#### 5. `nn.Sigmoid`

**解释**：将实数压缩到 \((0, 1)\)，用于二分类输出。如果使用 `BCEWithLogitsLoss`，输出层**不应**再加 Sigmoid，因为损失函数内部已经包含。

**传参数示例**：

```python
act = nn.Sigmoid()
out = act(torch.randn(4, 1))
```

---

#### 6. `nn.Softmax`

**解释**：沿指定维度归一化为概率分布。如果使用 `CrossEntropyLoss`，输出层**不应**再加 Softmax，因为损失函数期望原始 logits。

**传参数示例**：

```python
act = nn.Softmax(dim=1)                   # 在类别维度上归一化
logits = torch.randn(32, 10)
probs = act(logits)                        # 每行和为 1
```

---

### 权重初始化

`nn.init` 中的函数都**以 `_` 结尾**（表示原地修改张量），并且自动在 `torch.no_grad()` 下运行，不会影响 autograd。

#### 1. `nn.init.kaiming_uniform_`（He 初始化）

**解释**：配合 ReLU 族激活函数，保持前向/反向方差稳定。

**传参数示例**：

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

model.apply(init_weights)                  # 递归应用
```

---

#### 2. `nn.init.xavier_uniform_`（Glorot 初始化）

**解释**：配合 `tanh` 或 `sigmoid`，使输入输出方差尽量一致。

**传参数示例**：

```python
linear = nn.Linear(128, 256)
nn.init.xavier_uniform_(linear.weight)
nn.init.zeros_(linear.bias)
```

---

#### 3. `nn.init.zeros_` / `nn.init.normal_`

**解释**：偏置通常初始化为零；`normal_` 用于需要指定均值/标准差的自定义初始化。

**传参数示例**：

```python
nn.init.zeros_(linear.bias)

w = torch.empty(3, 5)
nn.init.normal_(w, mean=0.0, std=0.01)
```

---

### 模型选择：先定约束，再定基线

模型选择不是“哪个架构最新就用哪个”，而是先回答三个问题：**任务类型、数据模态与规模、可用算力与时间**。2026年的多项基准研究反复确认同一个结论：**任务特定的选择始终优于任何跨任务的默认方案**。不存在一个“通用最佳模型”。

#### 1. 按模态的基线推荐

**图像任务**：有预训练权重时，**ResNet 系列（如 ResNet-50）仍是最稳健的起点**；需要更强表示且算力允许时，ViT 系列（DeiT、Swin）是合理升级。ImageNet 规模以上的 CNN 训练中，**SGD + Momentum + Nesterov 仍然是金标准**；Transformer 架构则默认 **AdamW** 作为事实标准。不要从零训练大模型——先冻结骨干、只训练分类头，再逐步解冻微调。

**文本任务**：中小规模分类任务，TF-IDF + 线性模型或轻量微调模型（如 DistilBERT、MiniLM）足以建立有意义的基线；生成或复杂理解任务再引入更大规模的预训练模型。2026年的系统评估指出，**经典 ML 模型（LR、LightGBM、ResNet-50）在多个文本/图像医学分类任务中仍然具有竞争力**，并非所有场景都需要 LLM。

**表格任务**：**LightGBM / XGBoost 是表格数据的默认强基线**，在中小数据上经常优于深度模型。深度学习表格模型（FTTransformer、AutoInt）的价值在于**自动学习特征交互**，当手工特征工程已经充分、但行内交叉仍未被捕捉时再引入。

**时间序列**：先建立季节性朴素基线（seasonal naive）和线性模型基线，再考虑梯度提升或时序深度模型。

#### 2. 模型选择的核心原则

**从最简单能工作的基线开始，而不是从最复杂的模型开始。** 一个在验证集上可信的 LightGBM 或逻辑回归基线，比一个没有调好的大模型更有决策价值——它告诉你“这个任务的下限在哪里”，后续所有复杂模型都必须证明自己**显著超过这个下限**才值得被保留。

**评估指标决定选择方向。** 类别不平衡时用 macro-F1 或 AUC 而非 accuracy；排序任务用 NDCG；回归任务同时看 MAE 和 R²。指标选择错误会让模型选择本身失去意义。2026年的基准中，**验证集 macro-F1 被明确用于最佳 epoch 检查点的选择**，而不是训练损失。

**固定评估协议，再比较模型。** 统一的预处理、增强、优化器（通常 AdamW）和验证指标，是模型之间可比的前提。如果每个模型用不同的训练配置，比较结果不可信。

---

## 训练

模型的选择与训练，核心不在于“选一个最强的模型”，而在于**在任务约束下找到可验证、可复现、可收敛的基线，然后用训练组件把它推到该任务的上限**。模型选择决定起点，训练配置决定能否到达终点。

### 训练循环骨架：五步是不变的核心

无论使用原生 PyTorch 还是任何高级封装，单步训练永远是这五步：

```
前向 → 算损失 → 清梯度 → 反向传播 → 更新参数
```

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                   # 1. forward
        loss = criterion(logits, y)         # 2. loss
        optimizer.zero_grad()               # 3. 清梯度（默认累积）
        loss.backward()                     # 4. backward
        optimizer.step()                    # 5. update

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...
```

`optimizer.zero_grad()` 是最常见的遗漏项：PyTorch 默认**累积梯度**，不清零会导致梯度叠加，训练行为完全错误。

---

### 损失函数

#### 1. `nn.CrossEntropyLoss`

**解释**：多分类交叉熵。输入必须是**未经过 softmax 的原始 logits**，`target` 是类别索引的 `LongTensor`（形状 `(batch,)` 或 `(batch, d1, d2, ...)`）。

**关键参数**：`reduction`（`'mean'` / `'sum'` / `'none'`，默认 `'mean'`），`weight`（类别权重）。

**传参数示例**：

```python
criterion = nn.CrossEntropyLoss()
logits = model(x)                          # (32, 10)，未 softmax
target = torch.randint(0, 10, (32,))       # (32,)，类别索引
loss = criterion(logits, target)
```

---

#### 2. `nn.BCEWithLogitsLoss`

**解释**：二分类交叉熵，**内部自动做 Sigmoid + BCE**，数值更稳定。输入是原始 logits，target 是形状相同的浮点张量（0 或 1）。

**传参数示例**：

```python
criterion = nn.BCEWithLogitsLoss()
logits = model(x)                          # (32, 1)，未 sigmoid
target = torch.randint(0, 2, (32, 1)).float()
loss = criterion(logits, target)
```

---

#### 3. `nn.MSELoss`

**解释**：均方误差，用于回归。输入和 target 形状一致。

**关键参数**：`reduction`（默认 `'mean'`）。

**传参数示例**：

```python
criterion = nn.MSELoss(reduction="mean")
pred = model(x)                            # (32, 1)
target = torch.randn(32, 1)
loss = criterion(pred, target)
```

---

#### 4. `nn.L1Loss`

**解释**：平均绝对误差，对异常值比 MSE 更鲁棒。

**传参数示例**：

```python
criterion = nn.L1Loss()
loss = criterion(pred, target)
```

---

### 优化器：起点决定收敛路径

优化器不计算梯度，它**读取** `.grad` 并决定如何更新参数。选择取决于任务类型、内存约束和调参预算。

#### 1. `optim.AdamW`（默认起点）

**解释**：Adam 的**解耦权重衰减**版本——weight decay 不混入梯度中，而是直接作用于参数本身。这是 Transformer（GPT、Llama 等）和大模型微调的**事实标准**，也是大多数任务的**最稳妥默认起点**。它开箱即用，对超参数不敏感，适合调参时间有限的场景。

**关键参数**：`lr`（常用 `3e-4` 到 `1e-3` 作为起点），`weight_decay`（常用 `0.01`），`betas=(0.9, 0.999)`，`eps=1e-8`。

**传参数示例**：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)
```

#### 2. `optim.SGD`（追求最终精度时的选择）

**解释**：随机梯度下降 + 动量。在**有充分调参预算**的情况下，SGD + Momentum 往往能达到**更好的最终泛化精度**，尤其是在 ImageNet 规模的 CNN 训练中仍然是金标准。代价是需要精心设置学习率调度，且收敛更慢。

**关键参数**：`lr`（比 AdamW 大一个量级，常用 `0.1` 起步配合调度），`momentum`（通常 `0.9`），`weight_decay`，`nesterov=True`。

**传参数示例**：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
)
```

#### 3. `optim.Lion`（内存受限时的替代）

**解释**：Google 提出的**符号动量**优化器，只维护一个状态（AdamW 维护两个），**优化器显存约为 AdamW 的 50%**。在优化器状态成为显存瓶颈时，Lion 是直接替代方案。代价是对学习率更敏感，通常需要设为 AdamW 的 **1/10 左右**。

**传参数示例**：

```python
optimizer = torch.optim.Lion(
    model.parameters(),
    lr=3e-5,                # 约为 AdamW 的 1/10
    weight_decay=0.01,
)
```

#### 4. 优化器选择速查

| 场景 | 推荐 | 理由 |
|---|---|---|
| 默认起点 / 调参时间有限 | **AdamW** | 开箱即用，对超参数不敏感 |
| Transformer（NLP / ViT） | **AdamW** | 事实标准 |
| ImageNet 级 CNN，有调参预算 | **SGD + Momentum + Nesterov** | 最终泛化精度通常更好 |
| 优化器显存紧张 | **Lion** | 状态减半，lr 降至 1/10 |
| LLM 微调显存不足 | **AdamW 8-bit** | 优化器状态量化，显存降为 1/4 |

---

### 学习率调度：训练中最具杠杆的超参数

固定学习率几乎从来不是最优选择。调度器的作用是**在训练初期允许大步快速下降，在后期缩小步长做精细收敛**。调度器的 `step()` 调用位置因类型而异：**按 epoch 更新的调度器（Cosine、Step）在 epoch 末尾调用；按 batch 更新的调度器（OneCycle）在每个 batch 后调用；`ReduceLROnPlateau` 在验证指标计算后调用**。

#### 1. `CosineAnnealingLR`（配合 Warmup 使用）

**解释**：学习率按余弦曲线从初始值平滑衰减到接近零，是当前最常用的基础调度策略。**Warmup 在训练最初若干步将学习率从极小值线性升到目标值**，用于稳定 Transformer 和大模型的早期训练，避免初始大步长破坏预训练权重。两者组合（Warmup + Cosine Decay）是 Transformer 训练的标准配置。

**关键参数**：`T_max`（余弦周期长度，通常等于总 epoch 数或总 step 数），`eta_min`（最低学习率）。

**传参数示例**：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)     # 前 5 个 epoch warmup
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# 每个 epoch 末尾调用
scheduler.step()
```

#### 2. `OneCycleLR`（单周期策略）

**解释**：学习率在一个训练周期内先**从低升到 `max_lr`，再余弦衰减到远低于起点的值**。它内置了 warmup 和退火两个阶段，在固定的训练步数下通常能带来**更快的收敛和较强的正则化效果**，适合训练预算明确、不需要中途调整的快速实验。

**关键参数**：`max_lr`（峰值学习率），`total_steps`（总训练步数，必须精确等于 `epochs * len(train_loader)`），`pct_start`（上升阶段占比，默认 0.3）。

**传参数示例**：

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.3,
)

# 每个 batch 之后调用（不是 epoch 末尾）
scheduler.step()
```

#### 3. `ReduceLROnPlateau`（反馈驱动）

**解释**：**不按预设曲线衰减，而是监控验证指标**——当验证损失在连续 `patience` 个 epoch 内没有改善时，将学习率乘以 `factor`。它引入了“反馈闭环”，适合验证指标噪声较大、训练长度不固定的场景，也适合作为早停的互补手段。

**关键参数**：`mode`（`"min"` 监控损失 / `"max"` 监控指标），`factor`（衰减系数，如 0.1），`patience`（容忍轮数），`threshold`。

**传参数示例**：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5,
)

# 每个 epoch 验证后，传入验证指标
scheduler.step(val_loss)
```

#### 4. 调度器选择速查

| 场景 | 推荐调度器 | 调用频率 |
|---|---|---|
| Transformer / 大模型 | **Warmup + Cosine** | epoch 末尾 |
| 固定预算的快速实验 | **OneCycleLR** | **batch 末尾** |
| 验证指标噪声大 / 训练长度不定 | **ReduceLROnPlateau** | 验证后传入指标 |
| 简单 CNN 基线 | **StepLR**（每 N 轮 ×0.1） | epoch 末尾 |

---

### 正则化与稳定训练：防止过拟合与训练崩溃

#### 1. 权重衰减（Weight Decay）

**解释**：在损失中隐式或显式地惩罚大权重，限制模型对个别特征的依赖。**AdamW 的 weight decay 是解耦的**（不混入梯度），比原始 Adam 的 L2 正则更正确。常用值 `0.01`，大模型可达 `0.1`。它与优化器绑定，不单独调用 API。

#### 2. `nn.Dropout`（随机屏蔽）

**解释**：训练时以概率 `p` 随机将元素置零，推理时关闭，迫使网络分散表征、不依赖单一神经元。Transformer 中常用 `0.1`，CNN 全连接层常用 `0.2–0.5`。注意 `model.eval()` 后自动失效。

**传参数示例**：

```python
drop = nn.Dropout(p=0.3)
```

#### 3. 标签平滑（Label Smoothing）

**解释**：将硬标签 `[0, 1, 0]` 转为软标签（如 `[0.05, 0.9, 0.05]`），**缓解模型过度自信**，改善校准性。在分类任务中，`CrossEntropyLoss` 直接支持 `label_smoothing` 参数，无需手动修改标签。

**传参数示例**：

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### 4. 早停（Early Stopping）

**解释**：监控验证集指标，当连续若干轮没有改善时停止训练，并**恢复最佳检查点**。这是最朴素但极其实用的正则化手段，避免把随机波动误判为进步、避免在过拟合区间继续消耗算力。原生 PyTorch 需要手动实现早停逻辑；Keras 和 Lightning 提供现成回调。

**原生 PyTorch 手动实现**：

```python
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load("best.pt"))
```

**Lightning 回调**：

```python
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
]
trainer = L.Trainer(callbacks=callbacks, max_epochs=100)
```

#### 5. 梯度裁剪（Gradient Clipping）

**解释**：限制梯度的范数，防止梯度爆炸，是训练 RNN、Transformer 和深层网络的标准手段。必须在 `backward()` 之后、`step()` 之前调用。

**传参数示例**：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

#### 6. 梯度累积（Gradient Accumulation）

**解释**：在显存放不下大 batch 时，用多个小 batch 累积梯度，再执行一次更新，**等效于更大的有效 batch size**。关键：**只在累积边界上 `step` 和 `zero_grad`**，中间步只 `backward`；损失除以累积步数以保持梯度量级一致。

**传参数示例**：

```python
accum_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accum_steps
    loss.backward()

    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

### 精度与显存

#### 1. 混合精度：`autocast` + `GradScaler`

**解释**：自动混合精度（AMP）在选定区域内为每个算子选择 FP16/BF16 或 FP32，`GradScaler` 对 loss 做动态缩放防止 FP16 下梯度下溢。在 A100 等 GPU 上，AMP 能**显著提升吞吐并降低显存占用**。**梯度裁剪前需要 `scaler.unscale_(optimizer)`**，因为此时梯度仍处于缩放状态。

**传参数示例**：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")

for x, y in loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

#### 2. 梯度检查点（Gradient Checkpointing）

**解释**：**用计算换显存**——在前向时不保存中间激活，反向时重新计算。适合单卡放不下大模型、又无法使用参数分片的场景。代价是训练速度降低约 20–30%。PyTorch 原生通过 `torch.utils.checkpoint` 使用，但更常见的是在 Hugging Face `Trainer` 中开启 `gradient_checkpointing=True`，或对 Transformer 模块逐层包装。

**传参数示例（概念示意）**：

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(module, x):
    return checkpoint(module, x, use_reentrant=False)
```

---

### 分布式训练

#### 1. `DistributedDataParallel`（DDP）

**解释**：标准多 GPU 数据并行。每个 GPU 持有一份完整模型副本，处理不同的数据子集，反向传播后自动在所有进程间对梯度做 All-Reduce。你的训练循环几乎不变，只需启动方式和模型包装两处改动。

**关键参数**：`device_ids`（本进程绑定的 GPU 列表），`output_device`。

**传参数示例**：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 训练循环与单卡相同
for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

启动方式：`torchrun --nproc_per_node=4 train.py`。

---

#### 2. `no_sync()`：DDP 下的梯度累积优化

**解释**：在 DDP 中做梯度累积时，中间步骤的 `backward()` 默认也会触发跨 GPU 梯度同步，造成不必要的通信开销。`model.no_sync()` 上下文管理器**在内部跳过同步**，只在退出上下文后的那次反向传播中执行 All-Reduce。

**传参数示例**：

```python
accum_steps = 4

for i, (x, y) in enumerate(loader):
    ctx = model.no_sync() if (i + 1) % accum_steps != 0 else nullcontext()
    with ctx:
        loss = criterion(model(x), y) / accum_steps
        loss.backward()

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

#### 3. FSDP2 的 `fully_shard`

**解释**：当模型大到单卡放不下时，FSDP 对参数、梯度和优化器状态进行**分片**：前向/反向之前 all-gather 出完整参数，计算后立即释放；反向时 reduce-scatter 梯度。FSDP1 已废弃，当前使用 FSDP2，公共 API 通过 `fully_shard` 暴露。

**传参数示例**：

```python
from torch.distributed.fsdp import fully_shard

# 先对子模块逐层分片，再对根模型分片
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# 训练循环不变，启动方式与 DDP 相同
```

对子模块逐一 `fully_shard` 可以在前向计算某层时，让其余层保持分片状态，进一步压低峰值显存。

---

### 编译加速

#### 1. `torch.compile`

**解释**：PyTorch 2.x 的 JIT 编译器，通过 TorchDynamo 捕获计算图、TorchInductor 生成优化 kernel。最小用法只是把模型对象包一层，首次运行有编译开销，后续迭代显著加速。

**关键参数**：
- `mode`：`"default"`（平衡），`"reduce-overhead"`（降低 Python 开销），`"max-autotune"`（最长编译、最快运行）。
- `dynamic`：`True` 时允许变化的输入形状；如果形状固定，保持默认 `False` 能获得更好的优化。
- `fullgraph`：`True` 时要求整个前向完整捕获，遇到图断点直接报错而非回退。

**传参数示例**：

```python
model = MyModel().cuda()
compiled_model = torch.compile(
    model,
    mode="max-autotune",
    dynamic=False,
)
output = compiled_model(input)   # 首次编译，后续加速
```

与 DDP 组合时，建议先 `torch.compile(model)`，再用 `DDP` 包装编译后的模块。

---

### 训练流程控制

#### 1. `LightningModule` + `Trainer`

**解释**：把模型、优化器、训练/验证步骤组织进 `LightningModule`，`Trainer` 接管循环、设备放置、精度、日志、检查点和分布式策略。你保留对 `training_step`、`configure_optimizers` 的完全控制，其余工程细节交给 `Trainer`。

**关键参数**：`accelerator`（`"auto"`/`"gpu"`/`"cpu"`），`devices`，`max_epochs`，`precision`（如 `"16-mixed"`），`gradient_clip_val`。

**传参数示例**：

```python
import lightning as L
import torch
import torch.nn as nn

class LitMLP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.net(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=10,
    precision="16-mixed",
    gradient_clip_val=1.0,
)
trainer.fit(LitMLP(), train_dataloaders=train_loader, val_dataloaders=val_loader)
```

`self.log` 自动接入 TensorBoard 等日志器；`Trainer` 内部仍然执行那五步，只是不再由你手写。

---

#### 2. Hugging Face `Trainer` + `TrainingArguments`

**解释**：面向 Transformers 微调的专用高级封装。`TrainingArguments` 是一个集中配置的 dataclass，`Trainer` 接管训练、评估、保存、分布式、混合精度、梯度累积和 Hub 推送。它比 Lightning 更“开箱即用”，但定制自由度相应收窄。

**关键参数**：`output_dir`，`learning_rate`，`per_device_train_batch_size`，`num_train_epochs`，`evaluation_strategy`，`save_strategy`，`load_best_model_at_end`，`fp16`/`bf16`，`gradient_accumulation_steps`。

**传参数示例**：

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./out",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.train()
```

如果你的模型不是 Transformer，`Trainer` 仍然可用，但收益不如 Lightning 直接。

---

### 超参数搜索

#### 1. `Optuna` 的 `trial.suggest_*`

**解释**：把训练过程包装成一个 `objective(trial)` 函数，在函数内用 `trial.suggest_float` / `suggest_int` / `suggest_categorical` 声明搜索空间，Optuna 负责采样和剪枝。它与 PyTorch 训练循环完全解耦，侵入性极低。

**传参数示例**：

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden = trial.suggest_int("hidden", 64, 512, step=64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    model = build_model(hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # ... 训练若干 epoch，返回验证 loss ...
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

---

#### 2. Ray Tune + `OptunaSearch`

**解释**：Ray Tune 适合**并行、分布式**的超参搜索。它可以把 `OptunaSearch` 作为搜索算法，同时利用 Ray 的并行调度同时跑多个 trial。与 Lightning 组合时，还有 `TuneReportCheckpointCallback` 把指标回报给 Tune。

**传参数示例**：

```python
from ray import tune
from ray.tune.search.optuna import OptunaSearch

search_alg = OptunaSearch(metric="val_loss", mode="min")

tune.run(
    trainable,
    search_alg=search_alg,
    config={"lr": tune.loguniform(1e-5, 1e-2)},
    num_samples=50,
)
```

---

### 训练配置模板与避坑清单

#### 1. 一个可直接复用的训练配置骨架

```python
# ---------- 模型选择 ----------
# 图像: ResNet-50 (timm, pretrained) | 文本: MiniLM 微调 | 表格: LightGBM 基线
model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)

# ---------- 优化器 ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# ---------- 调度器（Warmup + Cosine） ----------
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

# ---------- 损失（含标签平滑） ----------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---------- AMP ----------
scaler = GradScaler("cuda")

# ---------- 早停与检查点 ----------
best_val, patience_counter = float("inf"), 0

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    val_loss = evaluate(model, val_loader)
    scheduler.step()

    if val_loss < best_val:
        best_val, patience_counter = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

#### 2. 训练中最隐蔽的五个陷阱

1. **忘记 `optimizer.zero_grad()`**：PyTorch 默认累积梯度，不清零会让训练静默错误。这是最常被遗漏的一行。

2. **`scheduler.step()` 调用位置错误**：`OneCycleLR` 必须在**每个 batch 后**调用，`CosineAnnealingLR` 在 **epoch 末尾**调用，`ReduceLROnPlateau` 在**验证指标计算后**调用。位置错误会让学习率曲线完全偏离预期。

3. **在 `model.train()` 状态下做验证**：忘记 `model.eval()` 和 `torch.no_grad()` 会让 Dropout 和 BatchNorm 在验证时仍然生效，验证指标虚高且不可复现。

4. **AMP 下先裁剪再反缩放**：梯度裁剪前如果没有 `scaler.unscale_(optimizer)`，裁剪的是被放大过的梯度，阈值失去意义。

5. **早停监控训练损失而非验证损失**：训练损失几乎总是在下降，用它做早停永远不会触发有意义的停止。早停必须监控**验证集**指标。

**模型选择提供的是“在哪个方向上优化”的判断，训练配置提供的是“能否沿这个方向有效到达”的工程能力。** 一个可信的基线 + 稳定的训练循环 + 正确的调度和正则化，其价值远大于一个未经调好的最新架构。先让基线跑通、跑稳、跑出可信的验证数字，再考虑架构升级和高级训练技巧。

---

### 高级 API 与底层 API 的映射

| 高级 API | 替你做的事 | 仍然暴露给你的控制点 |
|---|---|---|
| `Trainer.fit` / `LightningModule` | 循环、设备、日志、检查点 | `training_step`、`configure_optimizers` |
| `autocast` + `GradScaler` | 算子级精度选择、梯度缩放 | 前向区域边界、缩放策略 |
| 梯度累积 + `no_sync` | 等效大 batch、减少通信 | 累积步数、step 边界 |
| DDP / `fully_shard` | 梯度同步 / 参数分片 | 进程组、分片粒度 |
| `torch.compile` | 图捕获、kernel 融合 | `mode`、`dynamic`、后端选择 |
| `torch.export` / ONNX | 图序列化、跨运行时部署 | 输入签名、动态轴 |
| Optuna / Ray Tune | 搜索空间采样、并行调度 | `objective` 内部完全自定义 |

这些高级 API 没有改变你之前那五步的语义；它们改变的是**谁来写、以什么粒度写、以及额外叠加了哪些工程保障**。当你需要排查问题时，仍然可以随时退回 `torch.autograd`、手写循环或 `no_grad` 层面——高级封装的存在，是为了让常规路径更短，而不是取代底层可见性。

---

## 评估

评估回答的是“这个模型在未见数据上表现如何，这个结论有多可信”。它远比调一个指标函数复杂：**指标选择决定你看不看得见问题，评估协议决定你的结论是否有效，持续评估决定模型上线后是否还在正常工作**。一个在测试集上准确率 95% 的模型，如果评估集与训练集存在泄漏、指标不匹配业务代价、或缺乏校准，这个数字可能毫无决策价值。

### 基础任务指标：选对指标比调高指标更重要

#### 1. 分类任务：`torchmetrics` 与 `sklearn.metrics`

PyTorch 生态中，**`torchmetrics` 是评估指标的事实标准**，提供 100+ 指标，核心优势是**自动跨 batch 累积、自动多设备同步**，与训练循环和 DDP 无缝集成。`sklearn.metrics` 则适合在预测结果已经收集为 numpy 数组后的离线分析。

**`torchmetrics.classification`**

**解释**：模块化指标对象。在训练或评估循环中逐 batch 调用 `metric(preds, target)` 累积状态，循环结束后调用 `metric.compute()` 得到全集指标。分类任务必须显式指定 `task="multiclass"` 或 `"binary"` 和 `num_classes`。

**传参数示例**：

```python
import torchmetrics

metric = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=10
).to(device)

model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        metric.update(preds, y)          # 逐 batch 累积

acc = metric.compute()                    # 全集准确率
metric.reset()
```

**`confusion_matrix` / `classification_report`（sklearn）**

**解释**：`confusion_matrix` 给出 TP/FP/TN/FN 矩阵，`classification_report` 输出每个类别的 precision、recall、F1 和支持样本数。**类别不平衡时，accuracy 会严重误导**——一个把所有样本判为多数的模型可以有 90% 的 accuracy，但少数类的 recall 为 0。此时应看 macro-F1 或每个类别的指标。

**传参数示例**：

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
```

**`roc_auc_score` / `average_precision_score`**

**解释**：两者都是**排序质量指标**，不依赖于固定阈值。ROC-AUC 在类别平衡时含义清晰；**类别严重不平衡时，PR-AUC（`average_precision_score`）比 ROC-AUC 更敏感**，因为大量真负例会掩盖假正例的影响。

**传参数示例**：

```python
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_true, y_score)           # y_score 为概率或 logits
pr_auc  = average_precision_score(y_true, y_score)
```

**关键选择原则**：`accuracy` 只在类别平衡且误判代价对称时使用；不平衡分类用 **macro-F1** 或 **PR-AUC**；需要阈值无关的排序评估用 **ROC-AUC**；需要权衡 precision/recall 时看 `precision_recall_curve` 并选择符合业务代价的阈值。

---

#### 2. 回归任务：`torchmetrics.regression` / `sklearn.metrics`

**解释**：MAE 对异常值鲁棒，MSE/RMSE 对大误差惩罚更重。**RMSE 与目标同量纲**，比 MSE 更容易解释；**RMSE 显著大于 MAE 说明存在个别大误差**。R² 表示模型解释的方差比例，但**在样本量小或分布外测试时可能为负**，不应单独作为唯一指标。MAPE 在目标接近零时不稳定，sMAPE 有一定缓解但仍有偏差。

**传参数示例**：

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2   = r2_score(y_true, y_pred)
```

`torchmetrics.regression.MeanSquaredError(squared=False)` 等价于 RMSE，`R2Score` 同样可用。

---

#### 3. 分割与检测：`torchmetrics.detection` / `torchmetrics.segmentation`

**解释**：分割任务的核心指标是 **IoU（Jaccard）和 Dice**。Dice 对重叠更“宽容”（同样重叠下 Dice 数值高于 IoU），两者在医学分割中常同时报告。检测任务的核心是 **mAP**：先按 IoU 阈值判定预测框是否为命中，再按置信度排序计算每个类别的 Precision-Recall 曲线面积，最后对所有类别和 IoU 阈值取平均。**COCO 的 mAP@[.5:.95] 是最严格的标准**，torchmetrics 的 `MeanAveragePrecision` 默认后端为 `pycocotools`，与官方协议一致。

**传参数示例（分割）**：

```python
from torchmetrics.segmentation import MeanIoU, DiceScore

miou = MeanIoU(num_classes=21).to(device)
dice = DiceScore(num_classes=21, average="macro").to(device)

for images, masks in val_loader:
    preds = model(images).argmax(dim=1)
    miou.update(preds, masks)
    dice.update(preds, masks)

print(miou.compute(), dice.compute())
```

**传参数示例（检测）**：

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

for images, targets in val_loader:
    preds = model(images)
    metric.update(preds, targets)        # preds/targets 需为 dict 列表

results = metric.compute()
# results 包含 map, map_50, map_75, mar_100 等
```

---

#### 4. 排序与推荐：`NDCG` / `MRR` / `MAP`

**解释**：排序任务的评估关注**相关结果是否排在前面**，而不是分类是否正确。NDCG 按位置折损收益，MRR 只看第一个相关结果的位置，MAP 对每个查询的平均精度再取平均。这类指标通常没有统一的“一行 API”，需要在评估脚本中按查询分组计算，或借助 `torchmetrics.retrieval` 中的检索指标。

---

### 高级评估维度：指标之外的可信度

#### 1. 模型校准：ECE 与可靠性图

**解释**：**准确率回答“模型对不对”，校准回答“模型说 90% 信心时，是否真的 90% 正确”**。一个准确率 91% 的模型可以同时严重过度自信。**期望校准误差（ECE）** 按置信度分箱，计算每个箱内准确率与平均置信度之差的加权平均，0 表示完美校准。**温度缩放（Temperature Scaling）** 是最常用的后处理校准方法：在验证集上学习一个标量温度 \(T\)，推理时用 \( \text{softmax}(z/T) \) 替代原始 softmax，不改变预测类别，只调整置信度。

**传参数示例（概念示意）**：

```python
# 计算 ECE（需按置信度分箱）
from netcal.metrics import ECE

ece = ECE(bins=15)
ece_score = ece.measure(y_prob, y_true)   # y_prob: (N, C) 概率

# 温度缩放：在验证集上优化 T
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.T

scaler = TemperatureScaler()
optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.01)
# 在验证集上最小化 NLL 来学习 T
```

---

#### 2. 可解释性：特征归因与局部解释

**解释**：可解释性评估回答“模型为什么做出这个预测”。**Captum 是 PyTorch 原生的归因库**，提供 Integrated Gradients、GradientSHAP、DeepLIFT 等方法；**SHAP** 提供基于 Shapley 值的全局和局部特征贡献；**LIME** 适合单样本的局部解释。这些方法的输出不应被当作“因果真相”，而是**诊断工具**：如果归因结果与领域知识严重冲突，模型可能依赖了伪相关特征。

**传参数示例（Captum Integrated Gradients）**：

```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(
    inputs,
    baselines=torch.zeros_like(inputs),
    target=predicted_class,
    n_steps=50,
)
# attributions 形状与 inputs 相同，表示每个输入维度对预测的贡献
```

---

#### 3. 公平性与切片评估

**解释**：**聚合指标会掩盖子群体上的性能差异**。一个整体 AUC 0.85 的模型，在某个少数群体上可能只有 0.65。切片评估的做法是：按敏感属性（性别、年龄、地区）或业务维度将测试集分组，**分别计算每组指标，并报告组间最大差异**。`fairlearn` 提供了 sklearn 兼容的 `MetricFrame`，专门做分组指标对比。

**传参数示例**：

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "recall": recall_score},
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=group_labels,
)
print(mf.by_group)          # 每组指标
print(mf.difference())      # 组间最大差异
```

---

#### 4. 鲁棒性与不确定性

**解释**：鲁棒性评估包括**分布偏移下的性能保持**（在合成噪声、对抗扰动、时间偏移数据上的指标下降幅度）和**不确定性质量**（预测置信度是否与错误率一致）。对于安全关键场景，仅报告干净测试集指标是不够的——模型在轻微扰动下崩溃意味着评估结论不稳健。

---

### 统计比较：两个模型的差距是真实的吗

当模型 A 的测试准确率比模型 B 高 0.5% 时，这个差距可能完全来自测试集的随机抽样。**统计显著性检验是区分“真实改进”与“评估噪声”的必要步骤**。

**McNemar 检验**：用于比较两个分类器在**同一测试集**上的**成对决策差异**。它关注的是“A 对 B 错的样本数”与“B 对 A 错的样本数”是否显著不对称，而不是比较两个独立的准确率数值。

**DeLong 检验**：用于比较两个模型的 **AUC 差异**是否显著，是医学和机器学习论文中 AUC 比较的标准方法。

**Bootstrap 置信区间**：对测试集进行有放回重采样（通常 1000 次），计算指标的经验分布，给出 95% 置信区间。它不依赖分布假设，适用于任意指标（包括 mAP、校准误差），但计算成本较高。

**传参数示例（概念示意）**：

```python
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# 构造列联表：[[A对B对, A对B错], [A错B对, A错B错]]
table = [[both_correct, a_correct_b_wrong],
         [a_wrong_b_correct, both_wrong]]
result = mcnemar(table, exact=True)
print(result.pvalue)

# Bootstrap 置信区间
scores = []
for _ in range(1000):
    idx = np.random.choice(len(y_true), len(y_true), replace=True)
    scores.append(metric_fn(y_true[idx], y_pred[idx]))
print(np.percentile(scores, [2.5, 97.5]))
```

**报告原则**：如果两个模型的置信区间重叠且 McNemar/DeLong 检验不显著，不应声称“模型 A 更好”。**“没有显著差异”不等于“等价”**，但在模型选择中，它至少意味着“当前证据不足以支持更换模型”。

---

### 持续评估：模型上线后，评估没有结束

**解释**：离线测试集评估的是“模型在历史数据上表现如何”，但生产环境中数据分布会漂移、概念会变化、上游管道会故障。**持续评估（生产监控）回答的是“模型今天是否还在正常工作”**。

**Evidently** 提供数据漂移、目标漂移和模型性能的报告与监控。需要注意 **Evidently 在 0.6/0.7 版本后 API 发生了重大变化**，旧的 `from evidently.report import Report` 已不再推荐，应使用当前版本的 `Report` / `Dataset` 接口。

**NannyML** 的核心能力是**在没有真实标签的情况下估计模型性能**。它通过置信度分布和漂移检测来推断性能是否下降，适合标签延迟到达的场景（如信用违约需要数月才能观察到）。

**传参数示例（Evidently 概念示意）**：

```python
# 注意：API 随版本变化，以下为新版概念结构
from evidently import Report
from evidently.presets import DataDriftPreset, TargetDriftPreset

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=ref_df, current_data=prod_df)
report.save_html("drift_report.html")
```

**监控分层原则**：**数据漂移**是最早的预警信号，但不等于性能下降；**性能下降**（在有标签后计算）是最终确认；**业务指标**（转化率、坏账率）是最终影响。三者应分层监控，而不是只盯数据漂移。

---

### 评估协议与避坑清单

#### 1. 评估协议的三条铁律

**第一，测试集只触碰一次。** 调参、早停、模型选择、校准温度学习全部在**验证集**上完成；测试集只在最终报告时使用。反复在测试集上比较模型，测试集就变成了验证集，其上的指标不再是无偏估计。

**第二，所有拟合了数据的步骤必须在训练集上 fit。** 包括 scaler、encoder、`TargetEncoder`、特征选择、校准温度。如果在划分之前全局拟合，泄漏已经发生，后续任何指标都不可信。

**第三，报告分布而不是单点。** 单次划分的指标有随机性。至少报告**多次随机种子下的均值与标准差**，或交叉验证的折间分布。如果资源允许，给出 Bootstrap 置信区间。

#### 2. 五个最常见的评估陷阱

1. **用 accuracy 评估不平衡分类**：多数类基线可能就有 90%+ 的 accuracy，但模型毫无用处。改用 macro-F1 或 PR-AUC。

2. **在测试集上调阈值或早停**：测试集指标会被乐观偏置。所有决策应在验证集上完成。

3. **把校准和准确率混为一谈**：准确率高不代表置信度可信。安全关键场景必须单独评估 ECE 和可靠性图。

4. **只看聚合指标，不做切片分析**：整体指标可能掩盖某个子群体的严重退化。敏感属性和关键业务维度必须分组报告。

5. **声称微小提升而不做显著性检验**：0.3% 的准确率差异在测试集规模下很可能不显著。报告置信区间或成对检验结果，避免把噪声当作改进。

**评估的终点不是“跑出一个数字”，而是“这个数字在给定协议下可以被信任，并且指向正确的决策”。** 指标选择匹配任务代价，校准和切片暴露聚合指标的盲区，统计检验区分真实改进与噪声，持续评估在部署后守住底线。一个经过严格评估的简单模型，其决策价值远高于一个指标漂亮但评估协议不可信的复杂模型。

---

## 可视化

绘图在深度学习工作流中不是装饰，而是**诊断工具**：训练曲线暴露过拟合与学习率问题，模型图确认数据流，混淆矩阵和特征归因揭示模型实际依赖什么。绘图的选择由**你要回答的问题**驱动，而不是由哪个库“好看”驱动。

### 训练监控：曲线是第一个诊断信号

训练曲线回答的是“模型是否在学、是否在过拟合、学习率是否合适”。这是绘图最高频的用途。

#### 1. `matplotlib.pyplot.plot`（静态训练曲线）

**解释**：最基础的训练曲线绘制方式。`history` 中通常包含 `loss`、`val_loss`（以及 `accuracy`、`val_accuracy` 等），直接绘制即可观察收敛趋势。**关键点：损失和准确率应分开子图**，因为它们的量纲和变化方向不同，混在一张图上会压缩其中一条曲线的可读性。

**传参数示例**：

```python
import matplotlib.pyplot as plt

epochs = range(1, len(history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history["loss"], label="Train Loss", color="steelblue")
axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="darkorange", linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(epochs, history["accuracy"], label="Train Acc", color="steelblue")
axes[1].plot(epochs, history["val_accuracy"], label="Val Acc", color="darkorange", linestyle="--")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].set_title("Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
```

训练和验证曲线之间的**间隙**本身就是信息：验证损失在训练损失继续下降时开始上升，是过拟合的典型信号；两条曲线都高且接近，可能是欠拟合或学习率过小。

---

#### 2. EMA 曲线平滑

**解释**：逐 batch 或逐 epoch 的损失通常噪声很大，直接绘制会掩盖趋势。**指数移动平均（EMA）** 用 `ema_t = momentum * ema_{t-1} + (1 - momentum) * value_t` 平滑曲线，只需维护一个变量，实现极简，适合日常监控；滑动平均（moving average）在正式图表中也可用，但需要窗口对齐。平滑后的曲线用于观察趋势，**原始数据应以低透明度保留为背景**，避免平滑掩盖异常。

**传参数示例**：

```python
def smooth_ema(values, momentum=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        last = momentum * last + (1 - momentum) * v
        smoothed.append(last)
    return smoothed

plt.plot(epochs, history["loss"], alpha=0.3, label="Raw Loss")
plt.plot(epochs, smooth_ema(history["loss"], momentum=0.9),
         color="steelblue", label="EMA Loss")
plt.legend()
```

`momentum` 越大越平滑，但响应越滞后。日常监控常用 `0.9`；多卡训练中由于通信噪声更大，可下调至 `0.95–0.98` 以减少噪声放大。

---

#### 3. `plotly.graph_objects.Figure`（交互式训练曲线）

**解释**：当需要**缩放、悬停查看具体数值、叠加多个实验的曲线**时，Plotly 的交互能力明显优于静态图。`go.Figure` + `go.Scatter` 是基础入口，支持 `spline` 平滑和 `markers` 标记。适合在 Notebook 中探索，也可以直接导出为独立的 HTML 文件分享。

**传参数示例**：

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=epochs, y=history["train_loss"],
    mode="lines", name="Train Loss",
    line=dict(shape="spline", smoothing=1.3),
))
fig.add_trace(go.Scatter(
    x=epochs, y=history["val_loss"],
    mode="lines+markers", name="Val Loss",
    line=dict(dash="dash"),
))
fig.update_layout(
    title="Training Curves",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    template="plotly_white",
)
fig.write_html("training_curves.html")   # 交互式文件
```

`go.Scatter` 的 `line.shape="spline"` 可以生成平滑曲线，`mode="lines+markers"` 适合数据点较少的场景，便于看清每个 epoch 的实际值。

---

### 模型结构：确认数据流与参数

模型结构可视化的价值不在于“画得漂亮”，而在于**验证形状变化是否正确、定位参数集中的层、确认分支和跳跃连接符合设计**。

#### 1. `torchviz.make_dot`（计算图可视化）

**解释**：基于 PyTorch 的动态计算图生成可视化图。它展示的是**前向传播的实际计算图**，节点包括张量、模块和函数，边表示数据依赖。适合调试复杂的分支逻辑和自定义 `autograd.Function`。**依赖 Graphviz**，Linux 上需 `sudo apt install graphviz`，否则 `render` 会失败。

**传参数示例**：

```python
from torchviz import make_dot

y = model(dummy_input)
dot = make_dot(
    y,
    params=dict(model.named_parameters()),
    show_attrs=True,
    show_saved=True,
)
dot.render("model_graph", format="png")
```

`params=dict(model.named_parameters())` 让图中标注参数名，便于对应到具体层。`show_attrs` 和 `show_saved` 显示张量属性和被保存的中间变量，适合深入调试。

---

#### 2. `torchview.draw_graph`（模块级结构图）

**解释**：与 torchviz 关注计算图不同，`torchview` 生成的是**模块级的有向图**，展示 `nn.Module` 之间的连接关系和输入/输出形状。**关键优势：`device='meta'` 时只做形状推导，不消耗实际显存和内存**，因此可以对大模型做结构可视化而不需要真正加载权重。支持 `depth` 控制展开层级，`resize_graph(scale=...)` 调整分辨率。

**传参数示例**：

```python
from torchview import draw_graph

model_graph = draw_graph(
    model,
    input_size=(batch_size, 3, 224, 224),
    depth=3,
    device="meta",          # 零显存消耗，只做结构推导
    expand_nested=True,
    save_graph=True,
    filename="model_structure",
)
model_graph.visual_graph
```

`expand_nested=True` 会展开嵌套的 `Sequential` 和自定义模块，`depth` 控制展开深度，避免图过于庞大。

---

#### 3. `Netron`（ONNX 可视化）

**解释**：Netron 是一个**桌面/网页端的模型查看器**，支持 ONNX、TorchScript 等格式，以交互式方式展示算子级结构。它不通过代码调用，而是导出模型文件后用 `netron model.onnx` 打开。优势在于**算子粒度的清晰布局和交互式缩放**，适合在部署前检查导出模型的结构是否与预期一致。

**传参数示例**：

```python
# 先导出 ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
                  dynamo=True,
                  input_names=["input"],
                  output_names=["logits"])

# 然后在终端运行
# netron model.onnx
```

---

### 数据与预测诊断：混淆矩阵、ROC、散点

训练曲线告诉你“模型在学”，但不告诉你“**模型在哪类样本上错、错成什么样**”。

#### 1. `seaborn.heatmap`（混淆矩阵）

**解释**：混淆矩阵是分类诊断的基础。`seaborn.heatmap` 将矩阵可视化为颜色编码的热图，并可在单元格中标注数值。**类别不平衡时，只看对角线会误导**——需要结合每类的 support 和归一化（`normalize='true'` 按真实标签归一化，`'pred'` 按预测标签归一化）来观察召回率和精确率的结构。

**传参数示例**：

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
```

---

#### 2. `sns.lmplot` / `sns.countplot` / `sns.histplot`（数据分布与关系）

**解释**：在建模之前和评估阶段，seaborn 的统计图用于理解**数据分布和模型输出与输入的关系**。`countplot` 查看类别分布，`histplot` 查看数值分布，`lmplot` 可以按某个维度（如 Fold、模型版本）分组展示趋势线。`sns.set(style="whitegrid")` 适合作为深度学习报告的统一风格。

**传参数示例**：

```python
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# 类别分布
sns.countplot(data=df, x="label", hue="split")

# 模型输出与某个特征的局部关系
sns.lmplot(data=result_df, x="feature_value", y="shap_value",
           lowess=True, hue="fold", scatter_kws={"alpha": 0.4})
```

---

#### 3. ROC 与 Precision-Recall 曲线

**解释**：`sklearn.metrics.RocCurveDisplay` 和 `PrecisionRecallDisplay` 提供了现成的绘图接口，不需要手动计算 FPR/TPR 再 `plot`。**类别不平衡时应同时看 PR 曲线**，因为 ROC 曲线在负例极多时会被“稀释”，PR 曲线对少数类的表现更敏感。

**传参数示例**：

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
```

---

### 特征归因与注意力：模型在看哪里

#### 1. `captum.attr.visualization`（特征归因热图）

**解释**：Captum 是 PyTorch 原生的可解释性库，提供 Integrated Gradients、Grad-CAM、Occlusion 等归因算法。`captum.attr.visualization` 模块提供了**开箱即用的可视化工具**，特别是 `visualize_image_attr` 可以直接将归因结果叠加在原始图像上，生成热图，显示哪些区域支持了模型的分类决策。

**传参数示例**：

```python
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

ig = IntegratedGradients(model)
attributions = ig.attribute(
    input_img,
    target=predicted_class,
    n_steps=50,
    internal_batch_size=10,
)

# 将归因叠加在原始图像上
viz.visualize_image_attr(
    attributions[0].permute(1, 2, 0).detach().cpu().numpy(),
    original_image=original_img,
    method="blended_heat_map",
    sign="positive",
    show_colorbar=True,
    title="Integrated Gradients",
)
```

`sign="positive"` 只显示正贡献（支持预测的区域），`sign="all"` 同时显示正负贡献。`method="heat_map"` 只显示归因，`"blended_heat_map"` 将归因叠加在原图上，便于对照语义区域。

---

#### 2. 注意力图可视化（ViT / Transformer）

**解释**：对于 Vision Transformer 和 NLP Transformer，注意力权重是理解模型关注位置的重要信号。`vit-pytorch` 等库在 `forward` 中可以直接返回 attention maps；`transformer-attention-hooker` 提供了轻量工具，通过 hook 提取注意力权重并用 `plot_attention_grid` 绘制每个 head 的热图。**注意**：注意力权重高不等于“因果重要”，它只是模型分配计算资源的一种方式，应结合归因方法交叉验证。

**传参数示例（vit-pytorch 概念示意）**：

```python
from vit_pytorch import ViT

model = ViT(..., return_attention=True)
preds, attn_maps = model(img)
# attn_maps: 每个 Transformer 层的注意力矩阵
# 可视化某个 head 的 CLS token 对 patch 的注意力
plt.imshow(attn_maps[layer][0, head, 0, 1:].reshape(patch_h, patch_w))
```

---

#### 3. 特征降维可视化：`sklearn.manifold.TSNE`

**解释**：t-SNE 将高维特征嵌入降到二维或三维，用于**观察类别在特征空间中是否分离**。在评估阶段，对模型倒数第二层输出做 t-SNE，可以判断模型学到的表示是否具有判别性；在数据诊断阶段，可以观察是否存在聚类混杂或异常簇。

**传参数示例**：

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vis = tsne.fit_transform(features.cpu().numpy())

plt.scatter(vis[:, 0], vis[:, 1], c=labels, cmap="tab10", s=5)
plt.colorbar()
```

`perplexity` 通常设在 `5–50` 之间，值越大越关注全局结构，越小越关注局部邻域。

---

### 实验追踪与交互式面板（高级）

#### 1. `torch.utils.tensorboard.SummaryWriter`

**解释**：PyTorch 官方集成的实验追踪工具。`SummaryWriter` 将标量、图像、直方图、模型图、嵌入等记录到日志目录，TensorBoard 前端负责可视化。**核心方法**：`add_scalar` 记录损失/准确率，`add_histogram` 记录梯度或权重分布，`add_graph` 记录模型结构，`add_image` 记录输入或生成图像。适合**多实验对比**和**长时间训练的远程监控**（通过 SSH 端口转发在本地浏览器查看）。

**传参数示例**：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp1")

for epoch in range(epochs):
    # ... 训练 ...
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)

# 记录模型图（需要一个 dummy input）
writer.add_graph(model, dummy_input)
writer.flush()
writer.close()
```

启动：`tensorboard --logdir runs --port 6006`。**无界面 Linux 环境避免使用 `plt.show()`**，TensorBoard 和保存图片是更稳妥的选择。

---

#### 2. Weights & Biases（`wandb`）

**解释**：W&B 是云端实验管理平台，除了训练曲线，还支持**媒体记录**（图像、音频、视频）、**交互式表格**、**超参数平行坐标图**和**多实验对比面板**。`wandb.log` 是核心 API，`wandb.Histogram` 记录分布，`wandb.Table` 记录样本级预测结果。适合需要跨团队共享和多实验系统管理的场景。

**传参数示例**：

```python
import wandb

wandb.init(project="my-project", config={"lr": 3e-4})

for epoch in range(epochs):
    wandb.log({"loss": loss, "val_loss": val_loss, "epoch": epoch})

# 记录图像和表格
wandb.log({"examples": [wandb.Image(img) for img in sample_images]})
table = wandb.Table(columns=["id", "pred", "true"])
wandb.log({"predictions": table})
```

---

### 出版级图表的规范（高级）

#### 1. 配色与可访问性

**解释**：学术出版和正式报告中的图表需要满足**色盲安全**和**灰度可读**两条硬约束。不应仅靠颜色区分曲线——必须同时使用不同的**线型**（实线/虚线/点线）或**标记**（○/△/□）。推荐的色盲安全调色板在各类色觉缺陷下均可区分；避免红绿对比作为唯一区分手段。

**传参数示例（色盲安全配色）**：

```python
# 色盲安全的对比色（高对比度，适合线和柱）
COLORS = {
    "ours":   "#D55E00",   # 深橙红，高亮本方方法
    "base1":  "#0072B2",   # 蓝
    "base2":  "#009E73",   # 绿
    "base3":  "#CC79A7",   # 粉
    "baseline_gray": "#999999",
}

plt.plot(x, y_ours, color=COLORS["ours"], linestyle="-", marker="o", label="Ours")
plt.plot(x, y_base, color=COLORS["baseline_gray"], linestyle="--", marker="s", label="Baseline")
```

如果本方方法在多个基线中表现最好，**用更高对比度的颜色高亮本方曲线，其余基线用灰色或浅色**，这是顶会论文中常见的视觉策略。

#### 2. 矢量格式与导出

**解释**：LaTeX 论文优先导出 **PDF/EPS** 矢量格式，保证任意缩放不失真；位图仅用于无法矢量化的场景（如 AI 生成示意图）。`bbox_inches="tight"` 去除多余白边，`dpi` 仅在位图导出时影响清晰度。

**传参数示例**：

```python
plt.savefig("figure.pdf", bbox_inches="tight")      # 矢量，用于 LaTeX
plt.savefig("figure.png", dpi=300, bbox_inches="tight")  # 位图，用于网页/PPT
```

---

### 绘图 API 的选择对照

| 你要回答的问题 | 推荐入口 | 基础/高级 |
|---|---|---|
| 训练损失/准确率趋势 | `matplotlib.plot` + EMA 平滑 | 基础 |
| 交互式查看曲线细节 | `plotly.graph_objects.Figure` | 高级 |
| 模型计算图与形状流 | `torchviz.make_dot` | 基础 |
| 模块级结构（零显存） | `torchview.draw_graph(device="meta")` | 高级 |
| 算子级结构检查（部署前） | 导出 ONNX → `Netron` | 高级 |
| 混淆矩阵 / 分类错误结构 | `seaborn.heatmap` + `confusion_matrix` | 基础 |
| 数据分布与特征关系 | `seaborn.countplot/histplot/lmplot` | 基础 |
| ROC / PR 曲线 | `sklearn.metrics.*Display` | 基础 |
| 图像特征归因热图 | `captum.attr.visualization` | 高级 |
| Transformer 注意力图 | `vit-pytorch` / attention hooker | 高级 |
| 特征空间分离度 | `sklearn.manifold.TSNE` | 高级 |
| 多实验对比与远程监控 | `SummaryWriter` + TensorBoard | 高级 |
| 云端实验管理与媒体记录 | `wandb.log` / `wandb.Table` | 高级 |
| 出版级图表（色盲安全、矢量） | `matplotlib` + 色盲安全调色板 + PDF 导出 | 高级 |

**绘图的价值在于让不可见的训练过程变得可诊断。** 训练曲线是第一道警报，混淆矩阵和归因图回答“错在哪、为什么错”，模型结构图确认实现与设计一致，实验追踪让多次运行可比较。基础图表足以支撑日常调试；当需要多实验管理、深层归因或出版级输出时，再引入高级工具。**先让图能回答问题，再让它好看。**

---

## 导出与部署

训练结束之后，模型要离开 Python 进程：导出一张只含张量计算的图，再交给下游运行时；显存或延迟吃紧时，再叠加量化与推理期编译。

### 模型导出

#### 1. `torch.export.export()`

**解释**：PyTorch 2.x 的新一代导出入口，以 AOT（提前）方式将 `nn.Module` 和样本输入追踪为 `ExportedProgram`——一个只含张量计算的单图表示，可序列化、可在不同输入上执行，也是后续量化、ONNX 导出的基础。

**传参数示例**：

```python
import torch
from torch.export import export

exported = export(model, (example_input,))   # ExportedProgram
# 之后可序列化或交给下游工具
```

---

#### 2. `torch.onnx.export(..., dynamo=True)`

**解释**：导出为 ONNX。从 PyTorch 2.5 起，基于 TorchScript 的旧路径已不推荐；**推荐使用 `dynamo=True`**，它走 `torch.export` 图捕获，是当前的新默认方向。

**关键参数**：`dynamo=True`，`input_names`，`output_names`，`dynamic_axes`。

**传参数示例**：

```python
torch.onnx.export(
    model,
    (example_input,),
    "model.onnx",
    dynamo=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}},
)
```

如果你的目标运行时对算子支持有限，建议先在 `torch.export` 层确认图可导出，再走 ONNX 路径。

---

### 推理期加速

#### 1. `torch.inference_mode()`

**解释**：`torch.no_grad()` 只关闭梯度记录，`torch.inference_mode()` 还关闭了 autograd 的版本计数（version counter）和张量视图追踪，因此推理循环里更快。代价是它产生的张量不能再参与需要梯度的计算——推理脚本应默认用它替代 `no_grad`。

**传参数示例**：

```python
model.eval()
with torch.inference_mode():
    for x, _ in test_loader:
        logits = model(x.to(device))
```

如果在同一个进程里先推理再训练，注意 `inference_mode` 下得到的张量不能直接拿去反向传播。

---

#### 2. `torch.compile(mode="reduce-overhead")`

**解释**：推理场景优先固定输入形状并关闭梯度，`mode="reduce-overhead"` 会用 CUDA Graphs 把多次 kernel 启动合并，显著降低小 batch、高频调用时的启动开销。代价是显存占用略增，且对动态形状不友好——形状一变就会触发重编译。

**传参数示例**：

```python
model = torch.compile(model, mode="reduce-overhead", dynamic=False)

@torch.inference_mode()
def predict(batch):
    return model(batch)
```

CUDA Graphs 会复用固定缓冲区，输入需要拷贝进去；如果遇到 CUDA graphs 相关报错，先退回 `mode="default"` 确认问题不在图捕获。

---

### 量化

#### 1. `torchao.quantization.quantize_`（权重量化）

**解释**：`torchao` 是 PyTorch 官方的量化/稀疏化库。`quantize_` 原地把 `nn.Linear` 的权重替换为 int8 / int4 存储，反量化在 kernel 内完成，**不需要校准数据**，对模型结构改动最小。适合显存或带宽受限、而算力尚可的推理场景。

**关键参数**：量化方案（`int8_weight_only()` / `int4_weight_only()`）、可选的 `filter_fn` 用来只量化部分层。

**传参数示例**：

```python
from torchao.quantization import quantize_, int8_weight_only

model = model.eval().to("cuda")
quantize_(model, int8_weight_only())        # 原地替换权重

with torch.inference_mode():
    logits = model(x)
```

权重量化省的是显存和带宽，算力不是瓶颈时收益有限；精度损失通常很小，但仍应在验证集上复测一遍。

---

#### 2. PT2E 静态量化：`prepare_pt2e` / `convert_pt2e`

**解释**：PyTorch 2.x 推荐的量化路径。先用 `torch.export` 捕获图，`prepare_pt2e` 插入观察器，用校准数据跑一遍收集激活值范围，`convert_pt2e` 把观察器替换成量化/反量化算子。相比旧的 eager 模式 `torch.ao.quantization.quantize`，它按图改写，对自定义算子和动态形状更友好，也是 `torch.compile` 的 Inductor 后端能真正吃到 int8 加速的路径。

**关键参数**：`quantizer` 决定量化配置（x86 用 `X86InductorQuantizer`，移动端用 `XNNPACKQuantizer`）；校准集通常几十到几百个 batch。

**传参数示例**：

```python
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer, get_default_x86_inductor_quantization_config,
)

exported = torch.export.export(model.eval(), (torch.randn(1, 3, 224, 224),))

quantizer = X86InductorQuantizer().set_global(
    get_default_x86_inductor_quantization_config()
)
prepared = prepare_pt2e(exported.module(), quantizer)

with torch.inference_mode():                 # 校准：收集激活范围
    for x, _ in calib_loader:
        prepared(x)

quantized = convert_pt2e(prepared)
compiled = torch.compile(quantized)          # Inductor 生成 int8 kernel
```

PT2E 的 API 在小版本间仍在演进，升级 PyTorch 后建议对照官方 PT2E 教程确认签名。

---

### 导出与部署的选择对照

| 你要解决的问题 | 入口 | 产物 / 代价 |
|---|---|---|
| 换框架、换语言部署 | `torch.onnx.export(dynamo=True)` | ONNX 文件，算子覆盖取决于目标运行时 |
| 留在 PyTorch 生态、跨进程加载 | `torch.export` + `torch.export.save` / `load` | `.pt2` 归档，保留 ExportedProgram |
| 显存 / 带宽吃紧 | `torchao` 权重量化 | int8 / int4 权重，精度需复测 |
| 算力吃紧、要 int8 kernel | PT2E + `torch.compile` | 量化后的 Inductor kernel，需要校准集 |
| 小 batch、低延迟 | `torch.compile(mode="reduce-overhead")` | CUDA Graphs，形状必须固定 |
| 推理脚本省开销 | `torch.inference_mode()` | 不记录梯度，版本计数也关闭 |

**导出的第一原则是先在 `torch.export` 层确认图能导出，再挑下游格式。** 图捕获失败往往不是导出工具的问题，而是模型里有数据依赖的控制流（Python 分支、`.item()`、动态 shape）——这些要回到模型定义里解决，换成任何导出路径都绕不过去。

---

## 索引

- **数据清洗与划分**
  - 基础清洗操作：去重、缺失值、异常值 ｜ `torcharrow.DataFrame`（PyTorch 原生表格清洗） · `polars` 的惰性清洗
  - 算法驱动的智能清洗（高级） ｜ `cleanlab.Datalab` · `PyOD` / `alibi-detect`
  - 数据质量校验与管道集成（高级） ｜ `Great Expectations` · `DVC` 数据质量门禁
  - 清洗 API 的层级对照
  - 先看决策逻辑：你的数据该用什么划分
  - 基础随机划分 ｜ `torch.utils.data.random_split` · `sklearn.model_selection.train_test_split`
  - 分层划分：分类任务的基本约束 ｜ `stratify` 参数（sklearn）
  - 分组划分：防止实体级泄漏 ｜ `GroupKFold` / `GroupShuffleSplit` · `StratifiedGroupKFold`
  - 交叉验证 ｜ `KFold` / `StratifiedKFold` · `cross_validate`（高级）
  - 时间序列划分 ｜ `TimeSeriesSplit` · `TSSplitter`（tsai / fastai 生态，高级）
  - 分布偏移感知的划分（高级） ｜ `DominoSlicer` + `GroupShuffleSplit`
  - 高级集成：把划分逻辑收进管道 ｜ `datasets.train_test_split`（Hugging Face）
  - 划分策略速查与关键陷阱

- **加载与预处理**
  - 数据加载容器：`Dataset` 与 `DataLoader` ｜ `Dataset`（自定义数据集） · `DataLoader`
  - 采样与批次整理 ｜ `Sampler` · `BatchSampler` · `collate_fn`
  - 图像与视频：`torchvision.transforms.v2` ｜ `v2.Compose` · `v2.ToDtype` / `v2.ToImage` / `v2.ToPureTensor` · `v2.Normalize` · 几何变换（`v2.Resize` / `RandomResizedCrop` / `RandomHorizontalFlip` 等） · `v2.RandAugment` / `v2.AutoAugment` / `v2.TrivialAugmentWide` · `v2.MixUp` / `v2.CutMix`
  - 变换能力的增强 ｜ `torchvision.transforms.v2` 的自定义变换
  - 文本：从 `torchtext` 迁移 ｜ `get_tokenizer` · `Vocab`
  - 音频：`torchaudio` ｜ `torchaudio.load` · `torchaudio.transforms.Resample` · `torchaudio.transforms.MelSpectrogram`
  - 预处理流水线的位置
  - 管道的组织与复用 ｜ `LightningDataModule` · `Hugging Face datasets` 的 `with_format("torch")`
  - 加载性能与规模 ｜ `NVIDIA DALI` · `FFCV` · `WebDataset`
  - 高级数据 API 的定位对照

- **特征工程**
  - 数值特征：缩放、变换、分箱、交互 ｜ `StandardScaler` / `MinMaxScaler` / `RobustScaler` · `np.log1p` · `pd.cut` / `pd.qcut` / `KBinsDiscretizer` · `PolynomialFeatures` · 交互与比率特征（领域驱动） · 异常值裁剪 · 手动实现 `StandardScaler`（作为 `nn.Module`）
  - 类别特征：编码策略的选择取决于基数 ｜ `OneHotEncoder` / `pd.get_dummies` · `LabelEncoder` / 有序映射 · `TargetEncoder`（sklearn） · `category_encoders` 的 `BinaryEncoder` · 频率编码
  - 日期时间特征 ｜ 标准提取与周期编码
  - 特征选择：先构造，再筛选 ｜ 过滤法（Filter） · 包装法（Wrapper） · 嵌入法（Embedded）
  - 文本特征：从 TF-IDF 到上下文嵌入 ｜ `TfidfVectorizer` · `SentenceTransformer.encode`
  - 图像特征：手工算子与 CNN 嵌入 ｜ 传统算子（SIFT / HOG） · CNN / ViT 倒数第二层嵌入
  - 时间序列特征：统计描述子与大规模自动提取 ｜ `tsfresh.extract_features` · `Catch22`（aeon）
  - 聚合与滞后特征：业务场景中信息量最高的手工特征 ｜ `groupby().agg()` + `merge` · `shift` / `rolling`（滞后与滚动窗口）
  - 自动化特征工程（高级） ｜ `Featuretools.dfs` · `AutoFeatClassifier` / `AutoFeatRegressor` · AutoML 框架中的特征工程
  - 深度学习时代的表格特征：让模型学习交叉，而非手工枚举 ｜ `pytorch_tabular` / `mambular` 中的 `FTTransformer` / `AutoInt`
  - 特征工程 API 的层级对照与关键原则

- **模型构建**
  - 模型容器 ｜ `nn.Module` · `nn.Sequential` · `nn.ModuleList` · `nn.ModuleDict`
  - 基础层 ｜ `nn.Linear` · `nn.Conv2d` · `nn.MaxPool2d` · `nn.AdaptiveAvgPool2d` · `nn.BatchNorm2d` · `nn.LayerNorm` · `nn.Dropout` · `nn.Embedding` · `nn.LSTM`
  - 激活函数 ｜ `nn.ReLU` / `F.relu` · `nn.LeakyReLU` · `nn.GELU` · `nn.SiLU` · `nn.Sigmoid` · `nn.Softmax`
  - 权重初始化 ｜ `nn.init.kaiming_uniform_`（He 初始化） · `nn.init.xavier_uniform_`（Glorot 初始化） · `nn.init.zeros_` / `nn.init.normal_`
  - 模型选择：先定约束，再定基线 ｜ 按模态的基线推荐 · 模型选择的核心原则

- **训练**
  - 训练循环骨架：五步是不变的核心
  - 损失函数 ｜ `nn.CrossEntropyLoss` · `nn.BCEWithLogitsLoss` · `nn.MSELoss` · `nn.L1Loss`
  - 优化器：起点决定收敛路径 ｜ `optim.AdamW`（默认起点） · `optim.SGD`（追求最终精度时的选择） · `optim.Lion`（内存受限时的替代） · 优化器选择速查
  - 学习率调度：训练中最具杠杆的超参数 ｜ `CosineAnnealingLR`（配合 Warmup 使用） · `OneCycleLR`（单周期策略） · `ReduceLROnPlateau`（反馈驱动） · 调度器选择速查
  - 正则化与稳定训练：防止过拟合与训练崩溃 ｜ 权重衰减（Weight Decay） · `nn.Dropout`（随机屏蔽） · 标签平滑（Label Smoothing） · 早停（Early Stopping） · 梯度裁剪（Gradient Clipping） · 梯度累积（Gradient Accumulation）
  - 精度与显存 ｜ 混合精度 · 梯度检查点（Gradient Checkpointing）
  - 分布式训练 ｜ `DistributedDataParallel`（DDP） · `no_sync()` · FSDP2 的 `fully_shard`
  - 编译加速 ｜ `torch.compile`
  - 训练流程控制 ｜ `LightningModule` + `Trainer` · Hugging Face `Trainer` + `TrainingArguments`
  - 超参数搜索 ｜ `Optuna` 的 `trial.suggest_*` · Ray Tune + `OptunaSearch`
  - 训练配置模板与避坑清单 ｜ 一个可直接复用的训练配置骨架 · 训练中最隐蔽的五个陷阱
  - 高级 API 与底层 API 的映射

- **评估**
  - 基础任务指标：选对指标比调高指标更重要 ｜ 分类任务 · 回归任务 · 分割与检测 · 排序与推荐
  - 高级评估维度：指标之外的可信度 ｜ 模型校准 · 可解释性 · 公平性与切片评估 · 鲁棒性与不确定性
  - 统计比较：两个模型的差距是真实的吗
  - 持续评估：模型上线后，评估没有结束
  - 评估协议与避坑清单 ｜ 评估协议的三条铁律 · 五个最常见的评估陷阱

- **可视化**
  - 训练监控：曲线是第一个诊断信号 ｜ `matplotlib.pyplot.plot`（静态训练曲线） · EMA 曲线平滑 · `plotly.graph_objects.Figure`（交互式训练曲线）
  - 模型结构：确认数据流与参数 ｜ `torchviz.make_dot`（计算图可视化） · `torchview.draw_graph`（模块级结构图） · `Netron`（ONNX 可视化）
  - 数据与预测诊断：混淆矩阵、ROC、散点 ｜ `seaborn.heatmap`（混淆矩阵） · `sns.lmplot` / `sns.countplot` / `sns.histplot`（数据分布与关系） · ROC 与 Precision-Recall 曲线
  - 特征归因与注意力：模型在看哪里 ｜ `captum.attr.visualization`（特征归因热图） · 注意力图可视化（ViT / Transformer） · 特征降维可视化
  - 实验追踪与交互式面板（高级） ｜ `torch.utils.tensorboard.SummaryWriter` · Weights & Biases（`wandb`）
  - 出版级图表的规范（高级） ｜ 配色与可访问性 · 矢量格式与导出
  - 绘图 API 的选择对照

- **导出与部署**
  - 模型导出 ｜ `torch.export.export()` · `torch.onnx.export(..., dynamo=True)`
  - 推理期加速 ｜ `torch.inference_mode()` · `torch.compile(mode="reduce-overhead")`
  - 量化 ｜ `torchao.quantization.quantize_`（权重量化） · PT2E 静态量化
  - 导出与部署的选择对照

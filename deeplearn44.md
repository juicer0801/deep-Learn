---
title: 综合电力预测项目总结
date: 2026-10-14 21:27:47
tags:
---

# 电力负荷预测 · 工程笔记

> 基于本次从“基础 XGBoost”走到“MAE 2379”的全过程整理。重点是**通用逻辑、防泄漏原则、踩过的坑**，代码只保留关键片段。

---

## 一、项目分层

```
项目根/
├── data/                  # 原始数据 + 生成的图
├── model/                 # 模型文件
├── log/                   # 训练日志
├── src/
│   └── train.py           # 训练主流程：读数据 → 特征工程 → 训练评估
└── utils/
    ├── common.py          # 数据预处理 + 通用解析函数
    └── log.py             # 日志封装
```

**分层原则**：

- `utils/` 只放**无业务状态**的工具函数，可被训练、推理、测试复用；
- `train.py` 负责**编排流程**，不写具体解析细节；
- 原始数据、模型、日志、图**分开目录**，不混在一起。

---

## 二、数据预处理层（common.py）

### 2.1 标准流程

```python
def data_preprocessing(path):
    data = pd.read_csv(path, encoding='utf-8-sig')
    data = data.rename(columns={'数据时间': 'time', '总有功功率（kw）': 'power_load'})
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time')
    data = data.drop_duplicates()
    # 异常值 → NaN → 插值
    data['power_load'] = data['power_load'].interpolate(method='linear')
    data = data.reset_index(drop=True)
    return data
```

### 2.2 三条通用原则

| 原则 | 说明 |
|---|---|
| **rename 必须赋值** | `data = data.rename(...)`，不能只写 `data.rename(inplace=True)`，新版 pandas 会废弃 inplace |
| **先排序再去重再插值** | `shift` 特征依赖顺序，顺序错了全错 |
| **异常值不要硬编码阈值** | 用 `quantile(0.001/0.999)` 或业务下限，硬编码 10 万会误杀春节/夜间真实低负荷 |

### 2.3 踩过的坑

- ❌ `result.dropna(axis=0)` 不赋值 → 等于没执行 → 前 96 行 NaN 进训练；
- ✅ `result = result.dropna(axis=0).reset_index(drop=True)`；
- ❌ 硬编码 `< 100000` 当异常 → 春节、疫情低负荷被插值拉平；
- ✅ 先用 `describe()` + 比例统计确认，再改分位数阈值。

---

## 三、特征工程层（核心）

### 3.1 特征分五类

| 类别 | 代表特征 | 作用 |
|---|---|---|
| **时间** | `hour / month / weekday` one-hot | 告诉模型“现在是什么时段” |
| **滞后** | `power_load_t-1 / yesterday_load / last_week_load` | 告诉模型“刚才是多少” |
| **滚动** | `rolling_mean_onehour / oneday / oneweek` | 告诉模型“最近一段的平均水平” |
| **外部** | 天气 7 列、节假日 2 列 | 解释“为什么今天不一样” |
| **行业**（未接入） | 行业日负荷 lag/roll | 解释“上下游在干什么” |

### 3.2 时间特征写法

```python
result['hour'] = result['time'].dt.hour
result['month'] = result['time'].dt.month
result['weekday'] = result['time'].dt.dayofweek

hour_encoding = pd.get_dummies(result['hour'], prefix='hour')
month_encoding = pd.get_dummies(result['month'], prefix='month')
weekday_encoding = pd.get_dummies(result['weekday'], prefix='weekday')

result = pd.concat([result, hour_encoding, month_encoding, weekday_encoding], axis=1)
```

**为什么 one-hot**：星期几、小时是**类别**不是连续值。不做 one-hot，模型会以为“周一 < 周日”，学到错误关系。

### 3.3 滞后特征：三个尺度

```python
for i in range(1, 4):
    result[f'power_load_t-{i}'] = result['power_load'].shift(i * 4)   # 1/2/3 小时前
result['yesterday_load'] = result['power_load'].shift(96)             # 昨天同时刻
result['last_week_load'] = result['power_load'].shift(672)            # 上周同时刻
```

**为什么三个尺度**：短期（小时级）抓连续性，日级抓同日规律，周级抓周期规律。

### 3.4 滚动特征：`shift` 和 `window` 要配对

```python
# 正确：严格"过去 1 小时 / 1 天 / 1 周"，不含当前点
result['rolling_mean_onehour'] = result['power_load'].shift(4).rolling(4, min_periods=1).mean()
result['rolling_mean_oneday'] = result['power_load'].shift(96).rolling(96, min_periods=1).mean()
result['rolling_mean_oneweek'] = result['power_load'].shift(672).rolling(672, min_periods=1).mean()
```

**踩过的坑**：

- ❌ `shift(1).rolling(4)` → 和 `t-1` 高度共线，重要性虚高到 0.85，MAE 变差；
- ✅ `shift(4).rolling(4)` → 和 `t-1` 隔开，信息互补；
- **规律**：`shift` 的步数 ≈ `window` 的大小，表示“一个完整周期之前的那段”。

### 3.5 外部数据融合：三段式

```python
# 1. 构造日频特征表（一天一行）
weather = build_weather_features('../data/附件3-气象数据.csv', logger=logger)

# 2. 按 date 左连接
result['date'] = result['time'].dt.normalize()
result = result.merge(weather, on='date', how='left')

# 3. 缺失处理：天气可 ffill+bfill，行业只 ffill
result[weather_cols] = result[weather_cols].ffill().bfill().fillna(0)
```

**规则**：

- **天气**：前后都能补，因为气象数据本身独立于负荷；
- **行业**：**只向前补**，不能用未来补过去，否则泄漏；
- **节假日**：`chinese_calendar` 的 `is_holiday` / `is_workday`，注意调休日。

### 3.6 防泄漏三条铁律

1. **任何 rolling 都要 `shift(1)` 起步**，不能包含当前目标值；
2. **任何外部数据的“当天极值/汇总”不能直接 merge**，只能用 `shift(1)` 后的 lag/roll（附件2 行业最大最小值就是典型）；
3. **任何 `ffill` 都不能反转为 `bfill`**，除非确认该数据与目标无关（如天气）。

---

## 四、模型训练与评估

### 4.1 切分策略

```python
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]   # 时间序列，不能随机打乱
```

- 时间序列**必须按时间顺序切**，不能 `train_test_split(shuffle=True)`；
- 进阶用 `TimeSeriesSplit(n_splits=5)` 做滚动验证，确认单次切分不是运气。

### 4.2 调参经验

```python
XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    eval_metric='mae',
)
```

- **`early_stopping_rounds` 是关键**，让模型自己决定停在哪里；
- `best_iteration=930` 说明 2000 是够的，继续加树收益很小；
- `eval_metric='mae'` 和业务指标一致，比 `rmse` 更稳。

### 4.3 评估指标组合

| 指标 | 看什么 |
|---|---|
| **MAE** | 整体平均误差，业务最直观 |
| **MSE / RMSE** | 对极端值敏感，反映“最差能差多少” |
| **bias（真实-预测）** | 正=低估，负=高估，判断系统性偏差 |
| **分月 / 分小时 MAE** | 定位哪段时间差 |
| **误差最大 10 天** | 定位异常事件 |

**关键经验**：**bias 接近 0 但 MAE 高 → 不是水平问题，是形状或异常日问题**。这个判断能省掉大量无效调参。

### 4.4 特征重要性分析

```python
importance = pd.Series(model.feature_importances_, index=time_feature_names).sort_values(ascending=False)
print(importance.head(20))
```

**读法**：

- 单个特征重要性 **> 0.5** → 模型高度依赖它，有泄漏或共线嫌疑；
- 一类特征（如天气 7 列）**全部 < 0.001** → 该类特征当前形态贡献极小，不必再投入；
- **重要性高 ≠ 有用**：`rolling_mean_onehour` 曾到 0.85 但 MAE 变差，因为它和 `t-1` 共线。

### 4.5 模型保存要带特征列

```python
joblib.dump({
    'model': model,
    'features': time_feature_names,
    'target': 'power_load',
    'freq': '15min',
}, '../model/xgb_bundle.pkl')
```

**为什么**：推理时列顺序、列名必须和训练一致，只存 `.pkl` 模型不带特征列表，下次推理容易错位。

---

## 五、踩坑清单（按严重程度）

| # | 坑 | 后果 | 修复 |
|---|---|---|---|
| 1 | `rolling` 不含 `shift(1)` | 训练时偷看答案，指标虚高 | `shift(1).rolling(...)` 或 `shift(n).rolling(n)` |
| 2 | `dropna` 不赋值 | 前 N 行 NaN 进训练 | `result = result.dropna(...)` |
| 3 | rename 不赋值 | `KeyError: 'time'` | `data = data.rename(...)` |
| 4 | 行业当天极值直接 merge | 严重泄漏 | 只用 lag/roll |
| 5 | 硬编码异常阈值 | 误杀真实低负荷 | 分位数 / 业务下限 |
| 6 | `shift` 和 `window` 不对齐 | 语义混乱，共线 | `shift(n).rolling(n)` |
| 7 | 共线特征全塞进去 | MAE 变差 | A/B 对比，删掉冗余 |
| 8 | 只存模型不存特征列 | 推理错位 | 存 bundle |
| 9 | 随机切分时间序列 | 未来信息泄漏 | 按时间顺序切 |

---

## 六、本次关键指标演变

| 阶段 | MAE | MSE | 备注 |
|---|---|---|---|
| 基础版（t-1/2/3 + yesterday） | 2525 | 1724 万 | 有 `dropna` bug |
| 加 lag672 + rolling（有泄漏） | 2597 | 1522 万 | rolling 含当前点，指标虚高 |
| 修泄漏 + 加天气 | 2586 | 1731 万 | 真实基线 |
| rolling 改成 `shift(4)` | 2574 | 1721 万 | 语义修正 |
| 加节假日 | 2608 | 1763 万 | 2 月降 29，其他微升 |
| **调参 + 去共线** | **2379** | **1539 万** | **当前最优** |

**结论**：**调参 + 去共线**是单次收益最大的改动（MAE -229），远大于加天气（-0）和加节假日（+34）。

---

## 七、当前特征清单（定稿版）

```python
time_feature_names = (
    list(hour_encoding.columns)          # 24 列
    + list(month_encoding.columns)       # 12 列
    + list(weekday_encoding.columns)     # 7 列
    + weather_cols                       # 7 列：tmax,tmin,temp_diff,is_rain,is_cloud,is_sunny,wind_day
    + ['power_load_t-1', 'power_load_t-2', 'power_load_t-3']
    + ['yesterday_load', 'last_week_load']
    + ['rolling_mean_onehour', 'rolling_mean_oneday', 'rolling_mean_oneweek']
    + ['is_holiday']                     # 已去掉 is_workday_adjusted（共线，去掉更好）
)
```

---

## 八、尚未做 / 下一步候选

1. **`TimeSeriesSplit` 滚动验证**：确认 2379 是稳定水平还是单次运气；
2. **异常日处理**：2021-04-26 等区域性停工日，纯数据驱动预测不了，评估时单独报告“剔除 N 个异常日后 MAE”；
3. **行业日负荷（附件2）**：只用 `lag1/lag7/roll7`，先做行业日负荷预测，再输入 15 分钟模型（两级模型）；
4. **分时段建模**：早高峰、晚高峰、夜间分别建模，看是否比全局模型好；
5. **模型对比**：LightGBM / CatBoost 与 XGBoost 对比，通常 15 分钟负荷上树模型差距不大，但 CatBoost 对小样本异常日更稳。

---

## 九、一句话总结

> **防泄漏 > 调参 > 加特征**。先保证 rolling/lag/merge 没有偷看答案，再用早停调参，最后才考虑加外部特征。特征重要性高不代表有用，共线特征会互相干扰，A/B 对比是唯一标准。

# 电力负荷预测 · 工程级优化笔记

> 聚焦**代码工程化**：分层、配置、函数设计、可观测性、契约、可维护性。算法逻辑（特征语义、调参经验）见上一份笔记，这里不重复。

---

## 一、分层架构：职责边界

```
src/train.py          →  编排流程：读数据 → 特征工程 → 训练评估
utils/common.py       →  数据工具：预处理 + 原子解析函数
utils/log.py          →  基础设施：日志封装
data/ model/ log/     →  产物隔离：原始数据、模型、日志、图
```

**已做到**：

- `utils/` 只放**无业务状态**的工具函数，可被训练、推理、测试复用；
- `train.py` 不写具体解析细节，只调用；
- 数据 / 模型 / 日志 / 图分目录，不混放。

**判断标准**：如果明天要写 `predict.py`，`utils/` 里的东西能不能直接 import 复用？能，就分层对了。

---

## 二、配置外置：消灭魔法值

**已做到**：把散落在代码里的字符串/数字抽成模块级常量。

```python
# ---------------- 气象特征工程配置 ----------------
DATE_COL = '日期'
WEATHER_COL = '天气状况'
TMAX_COL = '最高温度'
TMIN_COL = '最低温度'
WIND_DAY_COL = '白天风力风向'

WEATHER_RULES = {
    'is_rain':  r'雨|雷|暴雨|阵雨',
    'is_cloud': r'阴|多云|雾',
    'is_sunny': r'晴',
}

DEFAULT_WIND_LEVEL = 2.0

WEATHER_FEATURE_COLS = [
    'date', 'tmax', 'tmin', 'temp_diff',
    'is_rain', 'is_cloud', 'is_sunny', 'wind_day',
]
```

**收益**：

| 常量 | 好处 |
|---|---|
| `*_COL` | 列名变化只改一处，不用全文搜索替换 |
| `WEATHER_RULES` | 以后加 `is_snow`，只加一行，主流程不动 |
| `WEATHER_FEATURE_COLS` | 输出列顺序固定，下游 merge 不会错位 |

**还没做**：特征工程里的魔法数字仍然散落。

```python
step = 4
window_size = 3
split_idx = int(len(X) * 0.7)
result['power_load'].shift(96)     # 96 是什么？一天 96 个 15 分钟点
result['power_load'].shift(672)    # 672 是什么？一周
```

**建议**：抽到顶部统一管理。

```python
FREQ_PER_HOUR = 4
FREQ_PER_DAY = 96
FREQ_PER_WEEK = 672
LAG_STEPS = [1, 2, 3]        # 小时级滞后
ROLL_WINDOWS = [4, 96, 672]  # 滚动窗口
TRAIN_RATIO = 0.7
```

这样 `shift(96)` 变成 `shift(FREQ_PER_DAY)`，代码自解释，改频率（比如换 5 分钟数据）只改一处。

---

## 三、函数设计：单一职责 + 可组合

**已做到**：把气象解析拆成 4 个原子函数，`build_weather_features` 只负责编排。

```python
_parse_date(series)                     # 中文日期 → datetime
_parse_temp(series)                     # "22℃" → 22.0
_classify_weather(series, rules)        # 天气文字 → 0/1 特征
_parse_wind(series, default)            # "北风3～4级" → 3.5
```

**好处**：

- 每个函数**可用一行输入单测**，不用读整个 csv；
- 出问题能定位到具体哪一步，而不是“整个函数报错”；
- 规则变化只动 `WEATHER_RULES`，不动函数。

**还没做**：`feature_engineering` 已经 100+ 行，承担了 5 件事：

```python
def feature_engineering(data, logger):
    result = data.copy()                    # 1
    result = result.merge(weather, ...)     # 2 天气
    result['hour'] = ...                    # 3 时间
    result['power_load_t-1'] = ...          # 4 滞后
    result['rolling_mean_*'] = ...          # 5 滚动
    flags = result['date'].apply(...)       # 6 节假日
    ...
```

**建议拆成**：

```python
def feature_engineering(data, logger):
    result = data.copy()
    result = _add_weather(result, logger)
    result = _add_time_features(result)
    result = _add_lag_features(result)
    result = _add_rolling_features(result)
    result = _add_holiday_features(result)
    result = result.dropna().reset_index(drop=True)
    feature_cols = _collect_feature_names(...)
    return result, feature_cols
```

**判断标准**：一个函数能一句话说清楚“它做什么”，就是单一职责；如果说成“它做 A 又做 B 还做 C”，就该拆。

---

## 四、可观测性：日志 + 数据质量报告

**已做到**：`Logger` 类封装 + 阶段性日志 + 数据质量报告。

```python
class Logger:
    level_relations = {...}
    def __init__(self, root_path, log_name, level='info', fmt=...):
        ...
    def get_logger(self):
        ...
        return self.logger
```

```python
def build_weather_features(path, logger=None):
    def log(msg):
        if logger is not None:
            logger.info(msg)
    ...
    report = {
        'rows': len(df),
        'date_min': ..., 'date_max': ...,
        'tmax_missing': ..., 'wind_missing': ...,
        'temp_abnormal': ...,
    }
    log(f"[weather] 数据质量: {report}")
```

**好处**：

- 训练挂了能翻日志定位到哪一步；
- `logger=None` 时静默，函数可在 Jupyter 里单独调试，不污染日志；
- 数据质量报告让“数据有没有问题”从“猜”变成“看数字”。

**还没做**：

- `Logger.get_logger()` 每次 `addHandler`，同一个 `log_name` 实例化两次会重复写日志；
- 训练日志里有**特征重要性、分月 MAE、误差最大 10 天**，但这些只写了 `logger.info`，没有**结构化落盘**（比如存成 `metrics.json`），后面做模型对比要手工复制。

**建议**：

```python
# 防重复 handler
if not self.logger.handlers:
    self.logger.addHandler(rotate_handler)

# 指标结构化
metrics = {'mae': mae, 'mse': mse, 'best_iter': model.best_iteration,
           'features': len(time_feature_names)}
with open('../log/metrics.json', 'w') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
```

---

## 五、契约与防御：类型提示 + 入参校验

**已做到**：

```python
def build_weather_features(path: str, logger=None) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str
    logger : logging.Logger, optional
    Returns
    -------
    pd.DataFrame
        列固定为 WEATHER_FEATURE_COLS，一天一行，按 date 升序
    """
    required = [DATE_COL, WEATHER_COL, TMAX_COL, TMIN_COL, WIND_DAY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"气象数据缺少列: {missing}，实际列: {list(df.columns)}")
```

**三个契约点**：

| 契约 | 做法 | 收益 |
|---|---|---|
| 输入契约 | `required` 列检查 | 数据变化时**立刻报错**，而不是后面 merge 出一堆 NaN |
| 输出契约 | 返回列固定 `WEATHER_FEATURE_COLS` | 下游按列名取，不依赖位置 |
| 类型契约 | `-> pd.DataFrame` + docstring | 别人（包括未来的你）知道返回什么 |

**对比**：没有校验的版本，如果附件3 改了一列名，会静默 merge 出一列 NaN，模型指标变差但不知道原因。**防御性编程的核心是“早点报错”**。

**还没做**：`feature_engineering` 和 `model_train_and_evaluate` 没有输入校验。比如传进来的 `data` 缺 `power_load` 列，要跑到 `y = data['power_load']` 才报错。**建议**在入口加：

```python
def feature_engineering(data, logger):
    assert {'time', 'power_load'}.issubset(data.columns), \
        f"输入缺少必需列，实际: {data.columns.tolist()}"
```

---

## 六、防泄漏的工程约束

**已做到**（通过代码结构保证，而不是靠人记）：

| 约束 | 工程手段 |
|---|---|
| 滚动不含当前点 | 统一用 `shift(n).rolling(n)` 的写法，不写 `shift(1).rolling(n)` |
| 时间序列不随机切 | `X.iloc[:split_idx]` 而不是 `train_test_split(shuffle=True)` |
| 外部数据只 lag | 行业数据只 merge `lag1/lag7/roll7`，不 merge 当天极值 |
| 天气可 bfill | `result[weather_cols].ffill().bfill()`，因为气象独立于负荷 |
| 行业只 ffill | 行业数据 `ffill()` 不 `bfill()`，避免未来信息补过去 |

**为什么这是“工程”而不是“算法”**：算法讨论“为什么不能泄漏”，工程讨论“怎么用代码结构保证不泄漏”。**好的工程是用 API 设计让人不可能写出泄漏代码**，而不是靠 code review 抓。

---

## 七、评估工程化：指标 / 诊断 / 可视化分离

**已做到**：三层评估各司其职。

```python
# 1. 指标层：量化好坏
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 2. 诊断层：定位问题
month_bias = error_df.groupby('month').agg(
    mae=('abs_error', 'mean'),
    bias=('error', 'mean'),
    true_mean=('y_true', 'mean'),
    pred_mean=('y_pred', 'mean'),
)
daily_mae = error_df.groupby(error_df['time'].dt.date)['abs_error'].mean()

# 3. 可视化层：肉眼确认
plt.plot(sub['time'], sub['y_true'], label='真实值')
plt.plot(sub['time'], sub['y_pred'], label='预测值')
```

**`error_df` 是评估的基石**：一张表同时支持指标、诊断、可视化。

```python
error_df = pd.DataFrame({
    'time': ..., 'y_true': ..., 'y_pred': ...,
    'error': ...,      # 带符号，看 bias
    'abs_error': ...,  # 绝对值，看 MAE
})
```

**还没做**：评估代码和训练代码混在 `model_train_and_evaluate` 里，函数 200+ 行。**建议**：

```python
def model_train_and_evaluate(data, feature_cols, logger):
    model = _train(data, feature_cols)
    _save_bundle(model, feature_cols)
    y_pred = model.predict(data[feature_cols].iloc[split_idx:])
    _report_metrics(y_test, y_pred, logger)
    _report_diagnostics(y_test, y_pred, data, logger)
    _plot_diagnostics(y_test, y_pred, data)
    return model
```

**判断标准**：函数超过 100 行，或者包含 `print` + `plt` + `logger` + `joblib` 四种副作用，就该拆。

---

## 八、已做到 vs 待改进（打分表）

| 维度 | 状态 | 说明 |
|---|---|---|
| 分层架构 | ✅ | 三层清晰，utils 可复用 |
| 配置外置 | 🟡 | 气象配置已外置，特征工程魔法数字散落 |
| 函数单一职责 | 🟡 | 原子解析函数做得好，主流程函数太长 |
| 类型提示 + docstring | 🟡 | `build_weather_features` 有，其余函数没有 |
| 输入契约校验 | 🟡 | 气象有 `required` 检查，其他函数没有 |
| 可观测性 | 🟡 | 日志有，指标未结构化落盘 |
| 防泄漏 | ✅ | 通过 `shift(n).rolling(n)` 和切分方式保证 |
| 评估分层 | 🟡 | 指标/诊断/可视化逻辑都在，但混在一个函数 |
| 模型保存 | ❌ | 只存 `.pkl`，没存 `feature_cols` |
| 单元测试 | ❌ | 无 |
| 路径管理 | ❌ | `'../data/附件3-气象数据.csv'` 硬编码 |

---

## 九、重构建议（按性价比排序）

| 优先级 | 改动 | 成本 | 收益 |
|---|---|---|---|
| **P0** | `joblib.dump` 存 bundle（model + features + 配置） | 5 行 | 推理不踩列错位的坑 |
| **P0** | 指标存 `metrics.json` | 5 行 | 模型对比不用手抄 |
| **P1** | 特征工程魔法数字抽常量 | 10 行 | 换频率只改一处 |
| **P1** | `feature_engineering` 拆成 5 个子函数 | 30 行 | 每加一类特征只改一个函数 |
| **P2** | 评估代码从训练函数拆出 | 50 行 | 训练/评估独立演进 |
| **P2** | `Logger` 加 `if not handlers` 防重复 | 2 行 | 推理脚本复用不重复写 |
| **P3** | 路径配置化（`config.py`） | 20 行 | 换数据目录不改代码 |
| **P3** | 给 `_parse_*` 函数写单测 | 50 行 | 解析逻辑坏了立刻发现 |

---

## 十、一句话总结

> **工程化的本质是“让错误早点暴露、让改动只发生在一处、让逻辑可被单独验证”。**
>
> 你现在已经做到：**配置外置、原子函数拆分、输入契约校验、防泄漏的代码结构**。
> 下一步该做：**模型 bundle、指标落盘、主流程函数拆分**——这三件事做完，代码就从“能跑”变成“能交接”。

# 电力负荷预测 · 切片与向量化优化笔记（含 `dt` 延申）

> 切片在 pandas 里不只是“取几个元素”，它同时决定了**索引对不对齐、内存有没有复制、逻辑有没有隐式 bug**。而 `dt` / `.str` 这类**访问器**是切片的延申：**切片是“取哪一块”，访问器是“从这一块里提取什么成分”**，两者共同决定代码是“能跑”还是“跑得快、不出错”。

---

## 一、先搞清楚三种切片

| 写法 | 按什么切 | 什么时候用 |
|---|---|---|
| `df.iloc[a:b]` | **位置**（第 a 行到第 b-1 行） | 时间序列切分、按行号取 |
| `df.loc[a:b]` | **标签**（索引值从 a 到 b，**含 b**） | 按时间/日期取、按条件取 |
| `df['col']` / `df[['c1','c2']]` | **列名** | 取列、取多列 |

**最容易错的两点**：

1. `iloc` **不含右端点**，`loc` **含右端点**；
2. `loc` 是按**索引值**切，不是按行号。如果 `reset_index` 过，`loc[0:10]` 和 `iloc[0:10]` 一样；如果没 reset，`loc[0:10]` 可能一行都没有。

---

## 二、这份代码里的 6 处切片

### 1. 训练/测试切分：`iloc[:split_idx]` ✅

```python
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

**为什么用 `iloc`**：

- 时间序列必须**按位置切**，不能按标签切；
- `iloc[:split_idx]` 是**视图**（view），不复制数据，内存友好；
- 如果用 `train_test_split(shuffle=True)`，会打乱时间顺序，未来信息泄漏。

**注意**：`X` 和 `y` 必须来自**同一个 `data`**，且中间没有对 `X` 或 `y` 单独 `reset_index`，否则 `iloc` 位置对不上。

---

### 2. 对齐取时间列：`data.loc[y_test.index, 'time']` ⚠️ 有风险

```python
error_df = pd.DataFrame({
    'time': data.loc[y_test.index, 'time'].values,
    'y_true': y_test.values,
    ...
})
```

**这行能跑，但依赖一个隐含前提**：`y_test.index` 是 `data` 的原始索引。

- 如果 `feature_engineering` 里 `result = result.dropna().reset_index(drop=True)`，`y_test.index` 就是 `data` 的行号，`data.loc[...]` 能对上；
- 如果某一步 `reset_index` 之后又 `merge`，索引可能变，`data.loc[y_test.index]` 就错位了。

**更稳的写法**：在切分时**同时切 `time` 列**，不依赖索引对齐。

```python
time_all = data['time']
time_train, time_test = time_all.iloc[:split_idx], time_all.iloc[split_idx:]

error_df = pd.DataFrame({
    'time': pd.to_datetime(time_test.values),
    'y_true': y_test.values,
    'y_pred': y_pred,
})
```

**收益**：

- 不依赖 `y_test.index` 和 `data.index` 是否一致；
- `time_test.values` 和 `y_test.values` 长度天然一致；
- 少一次 `data.loc` 查表，快一点点。

---

### 3. 字符串切片：`.str.split('/').str[0]` ✅

```python
day_weather = series.astype(str).str.split('/').str[0]
```

**做的事**：把 `"多云/小雨"` 拆成 `["多云", "小雨"]`，取第一个 `"多云"`。

**为什么不用 `apply(lambda x: x.split('/')[0])`**：

- `.str` 是 pandas 的**向量化字符串方法**，底层是 C 实现，比 `apply` + Python 循环快 5～10 倍；
- `.str` 对 NaN 自动返回 NaN，不用手写 `if pd.isna`；
- `apply(lambda ...)` 每行调一次 Python 函数，12 万行就是 12 万次函数调用。

**规律**：**只要能用 `.str.`，就不要用 `apply`**。

---

### 4. 列表推导取元组：`[x[0] for x in flags]` 🟡 可优化

```python
flags = result['date'].apply(_holiday_flags)   # 每行返回 (0/1, 0/1)
result['is_holiday'] = [x[0] for x in flags]
result['is_workday_adjusted'] = [x[1] for x in flags]
```

**能跑，但有两个问题**：

1. `apply` 返回的是 `Series of tuple`，再列表推导拆开，**遍历了两次**；
2. 列表推导的结果是 **Python list**，赋给 DataFrame 列时 pandas 要再转一次 `Series`。

**优化写法**：`apply` 直接返回 `Series`，一次拆分。

```python
flags = result['date'].apply(
    lambda d: pd.Series(_holiday_flags(d), index=['is_holiday', 'is_workday_adjusted'])
)
result = pd.concat([result, flags], axis=1)
```

**更快的写法**：`.str` 直接取。

```python
flags = result['date'].apply(_holiday_flags)
result['is_holiday'] = flags.str[0]
result['is_workday_adjusted'] = flags.str[1]
```

`flags.str[0]` 是向量化取元组第 0 位，比列表推导快，且自动对齐索引。

**但注意**：`chinese_calendar` 本身是**按行调用 Python 函数**，`apply` 无法避免逐行。如果数据量再大，可以**先对 `date` 去重再 apply**，再 merge 回去——因为一天只有一个节假日标记，96 个 15 分钟点重复算 96 次是浪费。

```python
unique_dates = result['date'].drop_duplicates().reset_index(drop=True)
flags = unique_dates.apply(_holiday_flags)
holiday_map = pd.DataFrame({
    'date': unique_dates,
    'is_holiday': flags.str[0],
    'is_workday_adjusted': flags.str[1],
})
result = result.merge(holiday_map, on='date', how='left')
```

**收益**：计算量从“12 万行”降到“约 1000 天”，快 100 倍以上。

---

### 5. 条件赋值：`.loc[mask, col]` ✅

```python
data.loc[data['power_load'] < 100000, 'power_load'] = np.nan
```

**为什么用 `.loc` 而不是 `data[mask]['col'] = ...`**：

- `data[mask]['col'] = ...` 是**链式索引**，pandas 可能返回一个 copy，赋值丢失，还会报 `SettingWithCopyWarning`；
- `data.loc[mask, 'col'] = ...` 是**原子操作**，直接改原 DataFrame，行为确定。

**规律**：**条件赋值一律用 `.loc[mask, col]`**，不要链式。

---

### 6. `reset_index(drop=True)`：切完必做 ✅

```python
result = result.dropna(axis=0).reset_index(drop=True)
```

**为什么**：`dropna` 之后索引是**不连续的**（比如 `0,1,2,5,6,9...`）。

- 不 reset：后面 `iloc[:split_idx]` 仍然按位置切，没问题；但 `loc[y_test.index]` 就可能对不上；
- reset 后：索引变成 `0,1,2,...,n-1`，`iloc` 和 `loc` 行为一致，调试时也好看。

**代价**：`reset_index` 会**复制一份数据**，内存翻倍。12 万行还好，几百万行就要考虑。

**规律**：**`dropna` / `merge` / `concat` 之后，统一 `reset_index(drop=True)`**，让索引始终干净。

---

## 三、延申：`dt` / `.str` 是切片的“下游伙伴”

切片是“取哪一块”，`dt` / `.str` 是“从这一块里提取什么成分”。**两者是同一类优化思路：能用 pandas 内置的向量化访问器，就不用 `apply` + Python 循环。**

### 3.1 `dt` 和 `.str` 是同一类访问器

| 访问器 | 用于 | 例子 | 底层 |
|---|---|---|---|
| `.str` | 字符串列 | `series.str.split('/')` | C 实现 |
| `.dt` | datetime 列 | `series.dt.hour` | C 实现 |

它们都比 `apply(lambda)` 快 5～10 倍，因为**不逐行调 Python 函数**。

```python
# ❌ 慢：12 万行调 12 万次 lambda
result['hour'] = result['time'].apply(lambda x: x.hour)

# ✅ 快：向量化，一次算完
result['hour'] = result['time'].dt.hour
```

你代码里 `result['time'].dt.hour / dt.month / dt.dayofweek / dt.normalize()` 全是向量化写法，这块已经是优化过的，不用改。

### 3.2 `dt` 能取的所有常用属性（备查）

```python
s.dt.year          # 年
s.dt.month         # 月
s.dt.day           # 日
s.dt.hour          # 时
s.dt.minute        # 分
s.dt.second        # 秒
s.dt.dayofweek     # 星期几 0-6（新版也叫 day_of_week，旧版只认 dayofweek）
s.dt.dayofyear     # 一年第几天
s.dt.quarter       # 季度
s.dt.is_month_start # 是否月初
s.dt.is_month_end   # 是否月末
s.dt.normalize()    # 归一化到当天 00:00
s.dt.floor('15min') # 向下取整到 15 分钟
s.dt.strftime('%Y-%m-%d')  # 格式化字符串
```

**规律**：想从 datetime 列里取任何时间成分，**先查 `dt` 有没有，没有再用 `apply`**。90% 的情况 `dt` 都有。

### 3.3 一处可优化：按天聚合用 `dt.normalize()` 而不是 `dt.date`

```python
# 你现在的写法
daily_mae = error_df.groupby(error_df['time'].dt.date)['abs_error'].mean()

# 更快：保持 datetime64，groupby 走 C 实现
daily_mae = error_df.groupby(error_df['time'].dt.normalize())['abs_error'].mean()
```

**原因**：

- `dt.date` 每行生成一个 Python `date` 对象，12 万行就是 12 万个对象；
- `dt.normalize()` 保持 `datetime64[ns]` 类型，groupby 走 C 层哈希，快很多。

**数据量小时差别不明显，数据量大时差距明显**。这是 `dt` 访问器里少数需要“选对具体方法”的地方。

### 3.4 `dt` / `.str` 与切片的关系

| 场景 | 切片 + 访问器 |
|---|---|
| 训练/测试切分后取小时 | `t_test.dt.hour` 向量化，一次算完 |
| 按月聚合误差 | `error_df['time'].dt.month` 做 groupby key，比 `apply(lambda)` 快 |
| 按天看 MAE | `error_df['time'].dt.normalize()` 做 groupby key |
| 天气字符串拆白天/夜晚 | `series.str.split('/').str[0]` |

**一句话**：**切片负责“取哪一段数据”，`dt` / `.str` 负责“从这段数据里快速提取成分”，两者组合起来才是完整的向量化写法。**

---

## 四、切片优化的三个通用原则

### 原则 1：能 `.str` / `.dt` 不 `apply`，能 `apply` 不去重算

| 写法 | 相对速度 |
|---|---|
| `.str.xxx` / `.dt.xxx` | 最快（C 实现） |
| `.apply(内置函数)` | 中 |
| `.apply(lambda)` | 慢 |
| `for` 循环 + `iterrows` | 最慢 |

### 原则 2：切分时同时切“特征 + 标签 + 辅助列”

不要只切 `X` 和 `y`，把 `time` 也一起切：

```python
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
t_train, t_test = data['time'].iloc[:split_idx], data['time'].iloc[split_idx:]
```

**收益**：后面画图、算 bias、按天聚合，都用 `t_test`，不用再 `data.loc[y_test.index]`，彻底摆脱索引对齐风险。

### 原则 3：重复计算前先问“能不能去重”

`_holiday_flags` 一天只算一次就够，96 个 15 分钟点重复算 96 次是浪费。**先 `drop_duplicates` 算，再 `merge` 回去**，是这类“日频特征挂到 15 分钟数据”的标准优化。

### 原则 4（延申）：访问器选对方法

同一个 `dt` 访问器下，不同方法代价不同。按天聚合用 `dt.normalize()` 而不是 `dt.date`，保持 `datetime64` 类型，groupby 快一个量级。

---

## 五、这段代码的切片打分

| 切片 / 访问器 | 写法 | 评价 |
|---|---|---|
| `X.iloc[:split_idx]` | 位置切，不 shuffle | ✅ 正确 |
| `data.loc[y_test.index, 'time']` | 依赖索引对齐 | ⚠️ 能跑，建议改成同时切 `time` |
| `.str.split('/').str[0]` | 向量化字符串 | ✅ 正确 |
| `[x[0] for x in flags]` | 列表推导拆元组 | 🟡 可改 `.str[0]`，最好去重后再算 |
| `data.loc[mask, 'col'] = ...` | 原子条件赋值 | ✅ 正确 |
| `reset_index(drop=True)` | 切完清索引 | ✅ 正确 |
| `dt.hour / dt.month / dt.dayofweek` | 向量化时间属性 | ✅ 正确 |
| `dt.date` 做 groupby key | 生成 Python 对象 | 🟡 可改 `dt.normalize()` |

---

## 六、一句话总结

> **切片优化的核心不是“写得更短”，而是“让索引永远对齐、让向量化替代循环、让重复计算只算一次”。**
>
> **切片是“取哪一块”，`dt` / `.str` 是“从这一块里提取什么成分”**——两者是同一类向量化思路的两端。
>
> 当前代码在 `iloc` 切分、`.str` / `.dt` 向量化、`.loc` 条件赋值上已经是对的；下一步优化三处：**切分时同时切 `time`，摆脱 `data.loc[y_test.index]` 的索引依赖**；**节假日特征先去重再算，再 merge 回去**；**按天聚合用 `dt.normalize()` 替代 `dt.date`**。
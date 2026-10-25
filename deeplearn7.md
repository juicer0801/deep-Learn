---
title: pandas与csv
date: 2026-09-26 19:35:06
tags:
---
# 用 pandas 处理 CSV：安装与常用操作

pandas 是 Python 里处理表格数据最常用的库，CSV 又是最常见的表格格式。如果你已经会 Python，但还没用过 pandas，下面这些内容可以帮你快速上手。

## 安装

直接 pip：

```
pip install pandas
```

如果网络慢，可以换国内源：

```
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

建议在 Jupyter Notebook 或 IPython 里交互式地敲代码，比在命令行里方便。装 Jupyter 也是 pip 一条命令（之前有介绍）：

```
pip install jupyter
jupyter notebook
```

用 VS Code、PyCharm 也完全可以，看个人习惯。

## 读取 CSV

先准备一个 CSV 文件，比如 `test.csv`：

```
姓名,年龄,城市
张三,25,北京
李四,30,上海
```

读取：

```python
import pandas as pd

df = pd.read_csv('test.csv')
print(df)
```

输出：

```
   姓名  年龄  城市
0  张三  25  北京
1  李四  30  上海
```

`df` 是 DataFrame 的惯例缩写，你可以把它理解成一张内存里的表格。`pd` 是 pandas 的通用别名。

`read_csv` 有一些常用参数：

- `encoding='gbk'` 或 `encoding='utf-8-sig'`：解决中文乱码
- `sep='\t'`：处理制表符分隔的文件
- `header=None`：文件没有表头时使用
- `names=['a','b','c']`：手动指定列名
- `dtype={'年龄': int}`：指定列的数据类型
- `parse_dates=['日期']`：把某列解析成日期

路径方面，Windows 下反斜杠需要转义，或者直接用正斜杠，或者加 `r` 前缀：

```python
df = pd.read_csv(r'C:\Users\me\data\test.csv')
```

## 查看数据概况

拿到一张表，先看几眼：

```python
df.head()        # 前5行，可传参指定行数
df.tail()        # 后5行
df.shape         # (行数, 列数)
df.columns       # 列名索引
df.info()        # 每列类型、非空数量、内存占用
df.describe()    # 数值列的统计摘要
df['年龄'].value_counts()  # 某列的值分布
```

`df.info()` 尤其有用，能快速发现数据类型不对或大量缺失值。

## 选择数据

选择列：

```python
df['姓名']              # 单列，返回 Series
df[['姓名', '城市']]    # 多列，返回 DataFrame
```

选择行，用 `loc`（按标签）和 `iloc`（按位置）：

```python
df.loc[0]              # 第一行（索引为0）
df.loc[0:2]            # 索引0到2的行
df.iloc[0]             # 第一行
df.iloc[0:2]           # 前两行
df.iloc[0, 1]          # 第0行第1列的值
```

条件筛选：

```python
df[df['年龄'] > 25]
df[(df['年龄'] > 25) & (df['城市'] == '北京')]
df[df['城市'].isin(['北京', '上海'])]
```

多个条件时，每个条件要用括号包起来，用 `&` 表示与，`|` 表示或。

## 修改数据

新增一列：

```python
df['明年年龄'] = df['年龄'] + 1
```

删除列：

```python
df = df.drop(columns=['明年年龄'])
```

排序：

```python
df.sort_values('年龄', ascending=False)
df.sort_values(['城市', '年龄'], ascending=[True, False])
```

对某列应用函数：

```python
df['城市'] = df['城市'].apply(lambda x: x.upper())
```

或者用 `map` 做值映射：

```python
df['城市'] = df['城市'].map({'北京': 'BJ', '上海': 'SH'})
```

## 处理缺失值（pytorch里面讲了）

实际数据常有缺失，pandas 里显示为 `NaN`。检查：

```python
df.isnull()          # 每个位置是否为空
df.isnull().sum()    # 每列缺失数量
```

处理方式：

```python
df.dropna()                         # 删除任何含空值的行
df.dropna(subset=['年龄'])          # 只考虑某列
df.fillna(0)                        # 用0填充
df['年龄'].fillna(df['年龄'].mean()) # 用均值填充
df['城市'].fillna(method='ffill')    # 用前一个值填充
```

选择哪种方式取决于数据含义。

## 保存 CSV

```python
df.to_csv('result.csv', index=False)
```

`index=False` 避免把行索引写进文件。其他常用参数：

- `encoding='utf-8-sig'`：让 Excel 正确识别中文
- `columns=['姓名', '年龄']`：只保存指定列
- `sep='\t'`：保存为制表符分隔

## 一些实用技巧

- **分块读取大文件**：`pd.read_csv('big.csv', chunksize=10000)` 返回一个迭代器，每次处理一万行。
- **只读部分列**：`pd.read_csv('data.csv', usecols=['姓名', '年龄'])` 节省内存。
- **跳过行**：`skiprows=3` 跳过前3行，`nrows=100` 只读100行。
- **合并多个 CSV**：`pd.concat([pd.read_csv(f) for f in files])`。
- **快速绘图**：`df['年龄'].plot.hist()` 需要 matplotlib。

## 最后

pandas 的功能远不止这些，但日常处理 CSV 时，上面这些操作已经能覆盖大部分场景。剩下的遇到再查文档即可(其实是我也不会了）。官方文档写得很清楚，搜索引擎也能解决多数问题。



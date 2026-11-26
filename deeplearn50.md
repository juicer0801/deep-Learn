---
title: 工具包捏
date: 2026-10-14 21:28:08
tags:
---

# 以下为测试数据
#替换工程版中的读取数据部分为：
```py
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=23)
```
后续代码的 df.iloc[:, feature_slice] 替换为直接使用 X

```python
import os
os.environ['OMP_NUM_THREADS'] = '4'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_kmeans_evaluation(data_path, feature_slice, k_range):
    """
    通用 KMeans 聚类评估函数，自动绘制 SSE 和 SC 曲线
    :param data_path: CSV 数据文件路径
    :param feature_slice: 特征切片的列索引，如 slice(3, 5) 对应 iloc[:, 3:5]
    :param k_range: K 值的遍历范围，如 range(2, 20)
    """
    # 1. 加载与预处理数据
    try:
        df = pd.read_csv(data_path)
        X = df.iloc[:, feature_slice]
    except FileNotFoundError:
        print(f"错误：找不到文件 {data_path}，请检查路径。")
        return

    sse_list, sc_list = [], []

    # 2. 循环训练与评估
    for k in k_range:
        # 训练模型
        model = KMeans(n_clusters=k, max_iter=100, random_state=23)
        model.fit(X)
        labels = model.labels_  # 或者使用 fit_predict(X)
        
        # 收集指标
        sse_list.append(model.inertia_)
        sc_list.append(silhouette_score(X, labels))

    # 3. 统一绘图设置
    def _draw_plot(y_data, title, y_label, color):
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, y_data, marker='o', label=y_label, color=color)
        plt.title(title)
        plt.xlabel('K (Number of Clusters)')
        plt.ylabel(y_label)
        plt.xticks(k_range)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

    # 4. 渲染图像 (SSE找拐点，SC找最高点)
    _draw_plot(sse_list, 'SSE Evaluation (Elbow Method)', 'SSE', 'blue')
    _draw_plot(sc_list, 'Silhouette Coefficient Evaluation', 'SC', 'orange')


# ================= 使用示例 =================
if __name__ == '__main__':
    # 只需传入这三个参数，即可复用整套流程
    PLOT_kmeans_evaluation(
        data_path='./data/customers.csv',
        feature_slice=slice(3, 5),  # 相当于 iloc[:, 3:5]
        k_range=range(2, 20)
    )


```

# NLP文本规范化
```py
# 数据清洗
def normalizeString(s):
    """
    字符串规范化函数
    :param s: 需要处理的字符串
    :return: 处理后的字符串
    """
    # 转为规范形式
    s = unicodedata.normalize("NFD", s)
    # 移除变音符号
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('Mn'))
    # 转为小写
    s = s.casefold()
    # 只保留字母和空白，和你原逻辑对齐
    s = ''.join(c for c in s if c.isalpha() or c.isspace())
    s = ' '.join(s.split())
    return s
```
基本适用所有数据集，有其他需求可补充。

## 保留？！. 版本
```py
import unicodedata

import unicodedata

def normalizeString(s):
    s = unicodedata.normalize("NFD", s)
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('Mn'))
    s = s.casefold()
    # 放行：字母、空白、指定标点 !?.
    s = ''.join(c for c in s if c.isalpha() or c.isspace() or c in "!?.")
    s = ' '.join(s.split())
    return s

```

## 案例
### re版
```py
import re

def normalizeString(s):
    """
    字符串规范化函数
    :param s: 需要处理的字符串
    :return: 处理后的字符串
    """
    # 1. 将字符串转成小写形式, 并去除收尾空白字符.
    s = s.lower().strip()

    # 2. 在 .!? 前加1个空格, 使用正则表达式的捕获组替换.
    # 参1: 正则表达式(即: 要被替换的内容), 参2: (用来)替换的内容, 参3: 要操作的字符串
    s = re.sub('([.!?])', r' \1', s)
    # print(f's: {s}')

    # 3. 过滤非标准字符: 保留大小写字母 和 基本的标点符号, 其它字符替换为: 空格.
    # [^a-zA-Z.!?]解释:  除了大小写字母, .!? 符号之外, 任意的1个字符
    # +解释: 数量词, 代表前边的内容至少出现1次, 至多出现无数次.
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)
    # print(f's: {s}')

    # 4. 返回处理后的字符串.
    return s
```

### Unicode 友好版本
```py
import unicodedata

def normalizeString(s):
    """
    字符串规范化函数
    Unicode友好、无正则实现，保留原代码全部功能：
    去重音、casefold小写折叠、. ! ? 前加空格、仅保留字母/空白/.!?、其余转空格、合并连续空格
    :param s: 需要处理的字符串
    :return: 处理后的字符串
    """
    # 1. Unicode NFD分解 + 移除变音标记(Mn) + casefold + 去除首尾空白
    s = unicodedata.normalize("NFD", s)
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('Mn'))
    s = s.casefold()
    s = s.strip()

    # 2. 在 . ! ? 前面插入空格（无正则，遍历实现捕获组效果）
    buf = []
    for c in s:
        if c in ('.', '!', '?'):
            buf.append(' ')
            buf.append(c)
        else:
            buf.append(c)
    s = ''.join(buf)

    # 3. 过滤字符：字母/空白/.!? 直接保留，其他字符替换成空格
    buf = []
    for c in s:
        if c.isalpha() or c.isspace() or c in ('.', '!', '?'):
            buf.append(c)
        else:
            buf.append(' ')
    s = ''.join(buf)

    # 4. 合并连续多个空白为单个空格（等价 re.sub(r'\s+',' ',s)，无正则写法）
    s = ' '.join(s.split())

    return s

```

### 融合版
```py
import re
import unicodedata

def normalizeString(s):
    """
    融合版：Unicode友好 + 适度使用re简化代码
    ✅ NFD、移除变音符号、casefold（流畅Python）
    ✅ 用re做标点前置空格（复用你原来熟悉的捕获组写法）
    ✅ isalpha() 保留所有Unicode拉丁字母，不局限a-z
    """
    s = unicodedata.normalize("NFD", s)
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('Mn'))
    s = s.casefold().strip()
    # 复用你原版的正则：. ! ? 前面插入空格
    s = re.sub(r'([.!?])', r' \1', s)
    # 过滤：合法字符保留，其余替换空格（循环方式，维持Unicode友好）
    s = ''.join(c if (c.isalpha() or c.isspace() or c in ".!?") else ' ' for c in s)
    # 合并连续空白
    return ' '.join(s.split())

```
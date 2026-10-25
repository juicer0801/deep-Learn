---
title: 线性代数（基本需求）
date: 2026-09-26 19:35:09
tags:
---

# 第 5 节：线性代数，深度学习的地基

上一节把数据预处理过了一遍，你可能已经发现，不管怎么处理，最后数据都变成了张量。模型里所有的运算，本质上都是张量之间的运算。

这一节讲线性代数。从深度学习用到的角度讲。请参照李沐《动手学深度学习》2.3 节，按会补充一些他没展开、实际写代码时又经常碰到的东西。

后面不管看什么模型——全连接、卷积、注意力——底层全是这些。

---

## 5.1 标量

标量就是单独的一个数， `3.0`、`-1`、`2.5`，都是标量。

在 PyTorch 里，标量是 0 维张量（只有1个元素）：

```python
import torch

x = torch.tensor(3.0)
print(x)         # tensor(3.)
print(x.shape)   # torch.Size([])
print(x.ndim)    # 0
```

**注意** `shape` 是 `torch.Size([])`，是空的。不是 `(1,)`，`(1,)` 是一维张量，shape里面有一个元素，注意区分

标量的运算就是普通算术：加、减、乘、除、乘方。

```python
a = torch.tensor(2.0)
b = torch.tensor(3.0)

print(a + b)    # tensor(5.)
print(a * b)    # tensor(6.)
print(a ** b)   # tensor(8.)
```

写模型的时候，标量用得很多：学习率、损失值、温度系数，这些都是标量。

---

## 5.2 向量

向量可视为标量值组成的列表，这些标量就是向量的元素：**向量是一个有序的数字列表，每个位置上的数描述某个特征。**
向量也可视为具有一个轴的张量

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)   # torch.Size([3])
print(x.ndim)    # 1
```

访问元素用下标（以列向量为默认方向），从 0 开始：

```python
print(x[0])   # tensor(1.)
print(x[1])   # tensor(2.)
```

向量的长度叫**维度**（dimension）。这里 `x` 是 3 维向量。注意跟“张量的维度数”区分开——张量的维度数是它有几个轴，向量的维度是它那个轴有多长。向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。而张量的维度用来表示张量具有的轴数。在这个意义上，张量的某个轴的维数就是这个轴的长度。

在深度学习里，一个样本通常就是一个向量。比如一张 28×28 的灰度图，展平之后是 784 维向量。一段文本用词袋表示，可能是几千维向量。

---

## 5.3 矩阵

矩阵是一张二维表，有行有列。

矩阵是具有2轴的张量
m行n列的矩阵，其形状为mxn 或者（m，n），当m = n时，它被称为方阵,对称矩阵是方阵的一种，A == `$A^T$`

同样创建矩阵时也可指定行与列（轴0与轴1）
```python
A = torch.arange(20).reshape(5,4)
A
```

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(A.shape)   # torch.Size([2, 3])
```

这个矩阵有 2 行 3 列。访问元素用 `A[i, j]`，第 i 行第 j 列：

```python
print(A[0, 1])   # tensor(2.)
```

如果没有给出标量元素的话，可以使用矩阵A的小写字母索引下标a<sub>ij</sub>来引用[A]<sub>ij</sub>,为了简单表示，必要时会将**逗号**插入单独的索引中a<sub>2,3j</sub>和[A]<sub>2i-1,3</sub>


矩阵的转置是行列互换，记作 `$A^T$`：
 
```python
print(A.T)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
print(A.T.shape)   # torch.Size([3, 2])
```

矩阵在深度学习里的意义：**一批样本。** 比如一个 batch 有 32 个样本，每个样本是 784 维向量，那这一批数据就是 32×784 的矩阵。每一行是一个样本。

---

## 5.4 张量

张量是标量、向量、矩阵的推广。0 维是标量，1 维是向量，2 维是矩阵，3 维及以上统称张量。
数学上的索引机制与矩阵类似x<sub>ijk</sub> [X]<sub>1,2i-1,3</sub>

```python
T = torch.arange(24).reshape(2, 3, 4)
print(T.shape)   # torch.Size([2, 3, 4])
print(T.ndim)    # 3
```

这个张量有 3 个轴，第 0 轴长 2，第 1 轴长 3，第 2 轴长 4。

在深度学习里，张量的维度通常有明确的物理含义。比如图像数据常用 `(batch, channel, height, width)`：

- batch：一批有多少张图
- channel：每张图有几个通道（RGB 是 3，灰度是 1）
- height、width：图像的高和宽

文本数据常用 `(batch, sequence_length, embedding_dim)`：

- batch：一批有多少条文本
- sequence_length：每条文本有多少个 token
- embedding_dim：每个 token 用多少维向量表示

理解这些形状的含义，比记住“张量是 n 维数组”重要得多。

---

## 5.5 张量算法的基本性质

### 按元素运算

两个同形状张量之间的运算，默认都是按元素（elementwise）来的。

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print(A + B)
# tensor([[ 6.,  8.],
#         [10., 12.]])

print(A * B)   # 按元素乘，不是矩阵乘法
# tensor([[ 5., 12.],
#         [21., 32.]])
```

`*` 是按元素乘，矩阵乘法用 `@` 或 `torch.matmul()`。

### 二元运算定律
给定具有相同形状的任意两个张量，按元素的加法和乘法满足以下性质：

交换律：A + B = B + A，A * B = B * A。
结合律：(A + B) + C = A + (B + C)，(A * B) * C = A * (B * C)。
分配律：A * (B + C) = A * B + A * C5。

张量加上或乘以一个标量不会改变张量的形状，其中张量的每个元素都与标量相加或相乘
任何按元素的一元运算都不会改变其操作数的形状。同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。

### 广播

不同形状的张量运算时，PyTorch 会自动广播。规则上一节讲过：从右往左比，对应维度要么相等，要么其中一个是 1，要么一方没有那个维度。

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])   # (2, 2)
b = torch.tensor([10.0, 20.0])   # (2,)

print(A + b)
# tensor([[11., 22.],
#         [13., 24.]])
```

`b` 被广播成了 `[[10, 20], [10, 20]]`。

### 哈达玛积

按元素乘有个正式名字，叫哈达玛积（Hadamard product），记作 `$A \odot B$`。深度学习里用得很频繁，比如门控机制、注意力权重，都是按元素乘。
示例：

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = A * B  # [[5, 12], [21, 32]]
```

---

## 5.6 降维

降维就是沿某个轴做聚合，让那个轴消失


### 求和

sum默认是沿所有

```python
A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(A)
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

print(A.sum())          # tensor(66.)  所有元素求和
print(A.sum(axis=0))    # tensor([12., 15., 18., 21.])  沿第 0 轴求和，形状 (4,)
print(A.sum(axis=1))    # tensor([ 6., 22., 38.])       沿第 1 轴求和，形状 (3,)
```

`A.sum(axis=0)` 把每一列加起来，第 0 轴消失。`A.sum(axis=1)` 把每一行加起来，第 1 轴消失。

**沿哪个轴操作，哪个轴就消失。** 所有降维操作都是这个逻辑。

如果想保留那个轴，长度变成 1，加 `keepdim=True`：

```python
print(A.sum(axis=0, keepdim=True).shape)   # torch.Size([1, 4])
print(A.sum(axis=1, keepdim=True).shape)   # torch.Size([3, 1])
```

`keepdim=True` 在广播的时候有用。比如你想把每行的和除回原矩阵，保持形状一致才能广播。

### 求平均
```python
A.mean()或
A.sum()/A.numel()
#同样可沿轴降维:
A.mean(axis=0)或
A.sum(axis=0)/A.shape[0]
```

```python
print(A.mean())          # 全局平均
print(A.mean(axis=0))    # 每列平均
print(A.mean(axis=1))    # 每行平均
```

### 求最大值和最小值

```python
print(A.max())           # 全局最大
print(A.max(axis=0))     # 每列最大，返回 values 和 indices
print(A.max(axis=1))     # 每行最大
```

`A.max(axis=0)` 返回的是一个 namedtuple，包含 `values` 和 `indices`。如果你只需要值：

```python
values, indices = A.max(axis=1)
print(values)    # tensor([ 3.,  7., 11.])
print(indices)   # tensor([3, 3, 3])
```

### 累加和

```python
print(A.cumsum(axis=0))
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.]])
```

`cumsum` **不降维**，保留原形状，每个位置是沿该轴到当前位置的累加和。

### axis方向问题

`A.sum(axis=0)` 到底是在“对行求和”还是“对列求和”？

看结果：`A.sum(axis=0)` 得到 `[12, 15, 18, 21]`，这是每一列的和。所以它是**把行方向压掉，对每一列求和**。

我自己的记法是：**axis=0 是往下缩压，把每一列的数加起来。axis=1 是往右压缩，把每一行的数加起来。**

---

## 5.7 点积

点积（dot product）是两个向量之间的运算，结果是一个标量。

给定两个向量 `$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$`，它们的点积是：

$$
\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i
$$

对应位置相乘，再全部加起来。

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(torch.dot(x, y))   # tensor(32.)  1*4 + 2*5 + 3*6 = 32
```

点积衡量两个向量的“相似程度”。如果两个向量方向接近，点积大；如果垂直，点积为 0；如果方向相反，点积为负。

按你的格式如下：

在深度学习中，点积无处不在。全连接层的计算本质上就是输入向量和权重向量的点积。注意力机制里的 query 和 key 做相似度计算，也是点积。在很多场景都很有用

$$
\boldsymbol{x}^{\mathrm{T}}\boldsymbol{y}
=\sum_{i=1}^{d}\boldsymbol{x}_i y_i.
$$

例如，给定一组由向量 `$\boldsymbol{x}\in\mathbb{R}^{d}$` 表示的值，和一组由向量 `$\boldsymbol{w}\in\mathbb{R}^{d}$` 表示的权重，`$\boldsymbol{x}$` 中的值根据权重 `$\boldsymbol{w}$` 的加权和可以表示为点积 `$\boldsymbol{x}^{\mathrm{T}}\boldsymbol{w}$`。当权重为非负且和为 1，即

$$
\sum_{i=1}^{d}w_i=1
$$

时，点积表示加权平均（weighted average）。将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。本节后面的内容将正式介绍长度（length）的概念。

## 5.8 矩阵-向量积

矩阵 `$A \in \mathbb{R}^{m \times n}$` 和向量 `$\mathbf{x} \in \mathbb{R}^n$` 相乘，得到一个 `$m$` 维向量。

$$
A\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^\top \\ \mathbf{a}_2^\top \\ \vdots \\ \mathbf{a}_m^\top \end{bmatrix} \mathbf{x} = \begin{bmatrix} \mathbf{a}_1^\top \mathbf{x} \\ \mathbf{a}_2^\top \mathbf{x} \\ \vdots \\ \mathbf{a}_m^\top \mathbf{x} \end{bmatrix}
$$

在代码中使用张量表示矩阵 - 向量积，我们使用 mv 函数。当我们为矩阵 A 和向量 x 调用`torch.mv(A, x)`时，会执行矩阵 - 向量积。注意，A 的列维数（沿轴 1 的长度）必须与 x 的维数（其长度）相同。

```python
A.shape, x.shape, torch.mv(A, x)
```

```
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

每个输出元素是矩阵对应行和向量的点积。

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])   # (3, 2)
x = torch.tensor([1.0, 1.0])     # (2,)

print(A @ x)   # tensor([ 3.,  7., 11.])
```

全连接层 `nn.Linear(in_features, out_features)` 原理同上：输入向量乘以权重矩阵，再加上偏置向量。

---

## 5.9 矩阵-矩阵乘法

矩阵 `$A \in \mathbb{R}^{m \times n}$` 和矩阵 `$B \in \mathbb{R}^{n \times p}$` 相乘，得到 `$C \in \mathbb{R}^{m \times p}$`。

\[
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
\]

`C` 的第 i 行第 j 列，是 `A` 的第 i 行和 `B` 的第 j 列的点积。

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])   # (2, 2)
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])   # (2, 2)

print(A @ B)
# tensor([[19., 22.],
#         [43., 50.]])
```

矩阵乘法不满足交换律，`$AB \neq BA$` 。但它满足结合律，`$(AB)C = A(BC)$`。

矩阵乘法的形状规则：**前一个的列数必须等于后一个的行数。** 结果形状是 `(前一个的行数, 后一个的列数)`。

```python
(3, 4) @ (4, 5) → (3, 5)   # OK
(3, 4) @ (3, 5)             # 报错，4 != 3
```

矩阵乘法在深度学习里的意义：**批量变换。** 一批数据 `(batch, in_features)` 经过权重矩阵 `(in_features, out_features)`，得到 `(batch, out_features)`。一个矩阵乘法就处理了整个 batch。

---

## 5.10 范数

范数（norm）衡量向量的大小。你可以把它理解成“向量的长度”。

### L2 范数

### 2.3.10 范数
## 李沐大神说
在线性代数中，向量范数是将向量映射到标量的函数$f$。给定任意向量$\boldsymbol{`$\boldsymbol{x}$`}$，向量范数具有一些性质。第一个性质是：如果我们按常数因子$\alpha$缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放：
$$
f(\alpha \boldsymbol{`$\boldsymbol{x}$`})=|\alpha| f(\boldsymbol{`$\boldsymbol{x}$`}) \tag{2.10}
$$

第二个性质是熟悉的三角不等式：
$$
f(\boldsymbol{`$\boldsymbol{x}$`}+\boldsymbol{y}) \leqslant f(\boldsymbol{`$\boldsymbol{x}$`})+f(\boldsymbol{y}) \tag{2.11}
$$

第三个性质简单地说就是范数必须是非负的：
$$
f(\boldsymbol{`$\boldsymbol{x}$`}) \geqslant 0 \tag{2.12}
$$

这是有道理的，因为在大多数情况下，任何数的最小的大小是0。最后一个性质要求范数最小为0，当且仅当向量全由0组成：
$$
\forall i,[\boldsymbol{`$\boldsymbol{x}$`}]_i=0 \Leftrightarrow f(\boldsymbol{`$\boldsymbol{x}$`})=0 \tag{2.13}
$$

范数很像距离的度量。欧几里得距离和毕达哥拉斯定理中的非负性概念以及三角不等式可能会给出一些启发。在代码中，我们可以按如下方式计算向量的$L_2$范数：
```
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```
`tensor(5.)`

深度学习中经常使用$L_2$范数的平方，也会经常遇到$L_1$范数，它表示为向量元素的绝对值之和：
$$
\|\boldsymbol{`$\boldsymbol{x}$`}\|_1=\sum_{i=1}^n |`$\boldsymbol{x}$`_i| \tag{2.15}
$$

与$L_2$范数相比，$L_1$范数受异常值的影响较小。为了计算$L_1$范数，我们将绝对值函数和按

最常用的是 L2 范数，也叫欧几里得范数：

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

就是每个元素平方，加起来，再开根号。

```python
x = torch.tensor([3.0, 4.0])
print(torch.norm(x))   # tensor(5.)  因为 sqrt(9+16)=5
```

L2 范数在深度学习里最常见的用途是**权重衰减**（weight decay），也叫 L2 正则化。它在损失函数里加上权重的 L2 范数，惩罚过大的权重，防止过拟合。

### L1 范数

L1 范数：

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|
$$

每个元素取绝对值相加。

```python
x = torch.tensor([3.0, -4.0])
print(torch.norm(x, p=1))   # tensor(7.)
```

L1 范数也用于正则化，叫 L1 正则化。与`$L_2$`范数相比，`$L_1$`范数受异常值的影响较小。

### Lp 范数

更一般地，Lp 范数定义为：

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}
$$

L1 和 L2 是 p=1 和 p=2 的特例。

### Frobenius 范数

矩阵的范数叫 Frobenius 范数，就是所有元素平方和开根号：

$$
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}
$$

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print(torch.norm(A))   # tensor(5.4772)  sqrt(1+4+9+16)
```

### 范数的性质

范数满足三条性质：

- 非负性：`$\|\mathbf{x}\| \geq 0$`，且 `$\|\mathbf{x}\| = 0$` 当且仅当 `$\mathbf{x} = \mathbf{0}$`
- 齐次性：`$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$`
- 三角不等式：`$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$`

这三条是范数的定义性质。任何满足这三条的函数都可以叫范数。

---

## 5.11 特殊矩阵

有几个特殊矩阵，看到要知道是什么。

**对称矩阵**：`$A = A^\top$`，转置等于自己。

**单位矩阵**：对角线是 1，其他是 0，记作 `$I$`。任何矩阵乘单位矩阵都不变：`$AI = IA = A$`。

```python
I = torch.eye(3)
print(I)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
```

**对角矩阵**：除了对角线，其他都是 0。

**正交矩阵**：`$A^\top A = I$`，转置就是逆矩阵。正交矩阵保持向量长度不变，只旋转方向。

**正定矩阵**：对所有非零向量 `$\mathbf{x}$`，都有 `$\mathbf{x}^\top A \mathbf{x} > 0$`。协方差矩阵就是正定的。


---

## 本节小结

有了理解深度学习模型的数学基础。

- **标量**是 0 维张量，**向量**是 1 维，**矩阵**是 2 维，更高维统称**张量**。
- **按元素乘**用 `*`，**矩阵乘法**用 `@`，别搞混。
- **降维**就是沿某个轴聚合，沿哪个轴操作，哪个轴消失。
- **点积**是两个向量的对应元素相乘再求和，衡量相似度。
- **矩阵乘法**是批量变换，形状规则是前列等于后行。
- **范数**衡量向量大小，L2 最常用，L1 产生稀疏性。
- **特殊矩阵**：对称、单位、对角、正交、正定，后面会反复碰到。

后面看全连接层，你要知道那是矩阵乘法；看注意力机制，你要知道那是点积和 softmax；看正则化，你要知道那是范数。

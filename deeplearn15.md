---
title: 人工神经网络
date: 2026-10-05 20:14:16
tags:
---
# 从零到精通：神经网络入门完全指南（详解版）

{% asset_img Snipaste_2026-10-05_20-48-26.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_21-02-33.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_21-04-20.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_21-06-23.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_21-23-56.png "Hexo 博客封面示例" %}

神经网络中信息只向一个方向移动，即从输入节点向前移动，通过隐藏节点，再向输出节点移动。其中的基本部分是:
1. 输入层（Input Layer）: 即输入x的那一层（如图像、文本、声音等）。每个输入特征对应一个神经元。输入层将数据传递给下一层的神经元。
2. 输出层（Output Layer）: 即输出y的那一层。输出层的神经元根据网络的任务（回归、分类等）生成最终的预测结果。
3. 隐藏层（Hidden Layers）: 输入层和输出层之间都是隐藏层，神经网络的“深度”通常由隐藏层的数量决定。隐藏层的神经元通过加权和激活函数处理输入，并将结果传递到下一层。
特点是：
同一层的神经元之间没有连接
第 N 层的每个神经元和第 N-1层 的所有神经元相连（这就是full connected的含义），这就是全连接神经网络
全连接神经网络接收的样本数据是二维的，数据在每一层之间需要以二维的形式传递（高维的也支持分batch）
第N-1层神经元的输出就是第N层神经元的输入
每个连接都有一个权重值（w系数和b系数）

{% asset_img Snipaste_2026-10-05_21-44-27.png "Hexo 博客封面示例" %}

每一个神经元工作时，前向传播会产生两个值，内部状态值（加权求和值）和激活值；反向传播时会产生激活值梯度和内部状态值梯度。
内部状态值
神经元或隐藏单元的内部存储值，它反映了当前神经元接收到的输入、历史信息以及网络内部的权重计算结果。
z=W⋅x+b
W：权重矩阵
x：输入值
b：偏置
激活值
通过激活函数（如 ReLU、Sigmoid、Tanh）对内部状态值进行非线性变换后得到的结果。激活值决定了当前神经元的输出。
a=f(z)
f：激活函数
z：内部状态值

## 一、引言：为什么需要神经网络

传统机器学习方法在处理图像、语音、自然语言等非结构化数据时，往往需要人工设计特征。例如，要让计算机识别一张猫的图片，传统方法需要人工提取"边缘""颜色直方图""纹理"等特征，这一过程既耗时又依赖领域知识。深度学习的核心突破在于：它能够自动从原始数据中学习到有用的特征表示，而不需要人工特征工程。

神经网络作为深度学习的基石，通过多层非线性变换，使得模型可以逐层抽象数据中的模式——较低的层级捕捉简单特征（如边缘、颜色），更高的层级则识别更复杂的模式（如物体、面部）。这种"逐层抽象"的能力，正是深度网络区别于浅层模型的关键。


## 二、神经网络基础

### 2.1 从生物神经元到人工神经元

生物神经元通过树突接收信号，在细胞核中聚集电荷，达到一定电位后通过轴突输出电信号。人工神经元对这一过程进行了数学抽象：对输入信号进行加权求和，再通过激活函数产生输出。

**单个神经元的数学表达**：

\[
z = \sum_{i=1}^{n} w_i x_i + b = \boldsymbol{w}^\top \boldsymbol{x} + b
\]

**公式解释**：
- \(x_i\)：第 \(i\) 个输入特征的值。例如在房价预测中，\(x_1\) 可能是面积，\(x_2\) 可能是房间数。
- \(w_i\)：第 \(i\) 个输入对应的权重（weight），表示该特征对输出的重要程度。权重越大，说明该特征对最终结果影响越大；权重为负，说明该特征与输出呈负相关。
- \(b\)：偏置项（bias），可以理解为神经元的"阈值"或"基准值"。即使所有输入为0，偏置仍能让神经元产生非零输出。
- \(\sum_{i=1}^{n}\)：对所有 \(n\) 个输入特征求和，即加权求和。
- \(\boldsymbol{w}^\top \boldsymbol{x}\)：向量形式的简写，\(\boldsymbol{w}\) 和 \(\boldsymbol{x}\) 都是 \(n\) 维列向量，\(\boldsymbol{w}^\top\) 表示 \(\boldsymbol{w}\) 的转置（变成行向量），两者相乘得到标量。
- \(z\)：称为**内部状态值**（或加权求和值、logits），是神经元未经非线性变换的原始输出。

接着，将 \(z\) 通过激活函数 \(f\) 得到最终输出：

\[
a = f(z)
\]

**公式解释**：
- \(f(\cdot)\)：激活函数（activation function），是一个非线性函数。它的作用是给神经网络引入非线性能力，使网络能够拟合复杂的函数关系。
- \(a\)：称为**激活值**（activation），即神经元的最终输出，会传递给下一层神经元作为输入。

### 2.2 激活函数

激活函数是神经网络具备非线性建模能力的关键。**如果没有激活函数，无论多少层神经网络叠加，最终都等价于一个线性变换**。例如，两层线性变换 \(y = W_2(W_1 x) = (W_2 W_1)x = Wx\)，仍然只是一个线性变换，无法拟合非线性数据。

#### 2.2.1 Sigmoid函数

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**公式解释**：
- \(e\)：自然对数的底数，约等于2.718。
- \(e^{-z}\)：指数函数，当 \(z\) 很大时 \(e^{-z}\) 趋近于0，当 \(z\) 很小时 \(e^{-z}\) 趋近于正无穷。
- 当 \(z \to +\infty\) 时，\(\sigma(z) \to 1\)；当 \(z \to -\infty\) 时，\(\sigma(z) \to 0\)；当 \(z = 0\) 时，\(\sigma(0) = 0.5\)。
- 因此Sigmoid将任意实数压缩到 \((0, 1)\) 区间，可以解释为概率。

**缺点**：当输入绝对值较大时（如 \(z > 6\) 或 \(z < -6\)），函数曲线变得非常平坦，导数趋近于零，导致**梯度消失**问题。在深层网络中，反向传播时梯度逐层相乘，很容易变成0，使浅层参数无法更新。

#### 2.2.2 Tanh函数

\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}
\]

**公式解释**：
- 分子 \(e^z - e^{-z}\)：当 \(z\) 为正且很大时，\(e^z\) 主导，分子为正且很大；当 \(z\) 为负且很小时，\(e^{-z}\) 主导，分子为负。
- 分母 \(e^z + e^{-z}\)：始终为正，且不小于2。
- 因此 \(\tanh(z)\) 的输出范围是 \((-1, 1)\)。
- \(\tanh(0) = 0\)，输出以零为中心，这比Sigmoid更有利于后续层的训练。

**优点**：输出以零为中心，缓解了Sigmoid的"非零均值"问题。**缺点**：仍然存在梯度消失问题。

#### 2.2.3 ReLU函数

\[
\text{ReLU}(z) = \max(0, z)
\]

**公式解释**：
- 当 \(z > 0\) 时，输出为 \(z\) 本身，导数恒为1。
- 当 \(z \le 0\) 时，输出为0，导数为0。
- 计算非常简单，只需一次比较操作。

**优点**：
1. 正区间梯度恒为1，有效缓解梯度消失问题。
2. 计算简单，训练速度快。
3. 稀疏激活：负半轴输出为0，使得网络具有一定稀疏性。

**缺点**：存在"Dead ReLU"问题。当某个神经元的输入持续为负时，其梯度始终为零，权重永远不更新，该神经元相当于"死亡"。

#### 2.2.4 Leaky ReLU

\[
\text{LeakyReLU}(z) = \max(\alpha z, z)
\]

**公式解释**：
- \(\alpha\)：一个小常数，通常取0.01或0.1。
- 当 \(z > 0\) 时，输出 \(z\)，与ReLU相同。
- 当 \(z \le 0\) 时，输出 \(\alpha z\)，是一个很小的负数，而不是0。
- 这样即使输入为负，梯度仍然为 \(\alpha\)（非零），避免了Dead ReLU问题。

**激活函数选择建议**：优先选择ReLU；如果ReLU效果不佳，可尝试Leaky ReLU等变体；少使用Sigmoid，可以尝试Tanh；回归问题的输出层可使用恒等映射（identity，即不做任何变换）。

```python
# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块，简写为nn
import torch.nn as nn
# 导入绘图库
import matplotlib.pyplot as plt

# 生成从-5到5的200个等间距点，作为横坐标
x = torch.linspace(-5, 5, 200)
# 计算Sigmoid激活值
sigmoid = torch.sigmoid(x)
# 计算Tanh激活值
tanh = torch.tanh(x)
# 计算ReLU激活值
relu = torch.relu(x)
# 计算Leaky ReLU激活值，负半轴斜率0.1
leaky_relu = nn.LeakyReLU(0.1)(x)

# 创建画布，宽10英寸，高4英寸
plt.figure(figsize=(10, 4))
# 绘制Sigmoid曲线，附带标签
plt.plot(x.numpy(), sigmoid.numpy(), label='Sigmoid')
# 绘制Tanh曲线
plt.plot(x.numpy(), tanh.numpy(), label='Tanh')
# 绘制ReLU曲线
plt.plot(x.numpy(), relu.numpy(), label='ReLU')
# 绘制Leaky ReLU曲线，需要detach()从计算图中分离
plt.plot(x.numpy(), leaky_relu.detach().numpy(), label='Leaky ReLU (0.1)')
# 显示图例
plt.legend()
# 显示网格
plt.grid(True)
# 设置标题
plt.title('Common Activation Functions')
# 设置横轴标签
plt.xlabel('z')
# 设置纵轴标签
plt.ylabel('f(z)')
# 显示图像
plt.show()
```

### 2.3 前向传播与网络结构

前向传播是指数据从输入层经过隐藏层，最终到达输出层产生预测值的过程。一个典型的前馈神经网络包含：

- **输入层**：接收原始数据，每个输入特征对应一个神经元。例如MNIST图像是28×28=784个像素，输入层就有784个神经元。
- **隐藏层**：输入层和输出层之间的所有层，网络的"深度"由隐藏层数量决定。
- **输出层**：根据任务类型（回归或分类）生成最终预测。回归问题输出1个值，10分类问题输出10个值。

同一层的神经元之间没有连接，第 \(N\) 层的每个神经元与第 \(N-1\) 层的所有神经元相连，这种结构称为**全连接神经网络**。

**两层网络的前向传播数学表达**：

第一层（隐藏层）：
\[
\boldsymbol{h} = f_1(\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1)
\]

**公式解释**：
- \(\boldsymbol{x}\)：输入向量，形状为 \((d_{in}, 1)\)。
- \(\boldsymbol{W}_1\)：第一层权重矩阵，形状为 \((d_{hidden}, d_{in})\)。
- \(\boldsymbol{b}_1\)：第一层偏置向量，形状为 \((d_{hidden}, 1)\)。
- \(\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1\)：线性变换，将 \(d_{in}\) 维输入映射到 \(d_{hidden}\) 维。
- \(f_1\)：第一层的激活函数（通常为ReLU）。
- \(\boldsymbol{h}\)：隐藏层输出，形状为 \((d_{hidden}, 1)\)。

第二层（输出层）：
\[
\boldsymbol{\hat{y}} = f_2(\boldsymbol{W}_2 \boldsymbol{h} + \boldsymbol{b}_2)
\]

**公式解释**：
- \(\boldsymbol{W}_2\)：第二层权重矩阵，形状为 \((d_{out}, d_{hidden})\)。
- \(\boldsymbol{b}_2\)：第二层偏置向量，形状为 \((d_{out}, 1)\)。
- \(f_2\)：输出层激活函数（分类用Softmax，回归用恒等映射）。
- \(\boldsymbol{\hat{y}}\)：最终预测输出，形状为 \((d_{out}, 1)\)。

### 2.4 从零实现多层感知机

以下代码参考李沐《动手学深度学习》的多层感知机从零实现：

```python
# 导入PyTorch
import torch
# 导入神经网络模块
from torch import nn
# 导入d2l工具库（李沐《动手学深度学习》配套库）
from d2l import torch as d2l

# 设置批量大小为256，即每次训练使用256个样本
batch_size = 256
# 加载Fashion-MNIST数据集，返回训练迭代器和测试迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 输入维度：28x28=784（每张图片展平为784维向量）
# 输出维度：10（10个类别）
# 隐藏单元数：256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化第一层权重W1：形状(784, 256)，服从标准正态分布乘以0.01进行缩放
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 初始化第一层偏置b1：形状(256,)，全零
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 初始化第二层权重W2：形状(256, 10)，服从标准正态分布乘以0.01
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# 初始化第二层偏置b2：形状(10,)，全零
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# 将所有参数放入列表，便于优化器统一管理
params = [W1, b1, W2, b2]

# 定义ReLU激活函数
def relu(X):
    # 创建一个与X形状相同的全零张量
    a = torch.zeros_like(X)
    # 逐元素取X与0的最大值，即ReLU
    return torch.max(X, a)

# 定义前向传播函数
def net(X):
    # 将输入X从(batch_size, 1, 28, 28)重塑为(batch_size, 784)
    X = X.reshape((-1, num_inputs))
    # 隐藏层：X @ W1 + b1，再经过ReLU激活
    H = relu(X @ W1 + b1)
    # 输出层：H @ W2 + b2，返回logits
    return (H @ W2 + b2)

# 定义交叉熵损失函数（内部包含Softmax）
loss = nn.CrossEntropyLoss()
# 训练轮数：10
num_epochs = 10
# 学习率：0.1
lr = 0.1
# 定义随机梯度下降优化器，管理所有参数
updater = torch.optim.SGD(params, lr=lr)
# 调用d2l封装好的训练函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

这段代码清晰地展示了神经网络的本质：前向传播就是一系列矩阵乘法与非线性激活的交替组合。


## 三、损失函数

损失函数（Loss Function）是深度学习中衡量模型预测值与真实值之间差异的度量。优化算法的目标就是最小化这个差异。

### 3.1 均方误差

均方误差（Mean Squared Error, MSE）是最常见的回归问题损失函数：

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**公式解释**：
- \(n\)：样本总数。
- \(y_i\)：第 \(i\) 个样本的真实值（标签）。
- \(\hat{y}_i\)：第 \(i\) 个样本的预测值。
- \(y_i - \hat{y}_i\)：预测误差（残差）。
- \((y_i - \hat{y}_i)^2\)：误差的平方。平方有两个作用：一是消除正负号，使误差同向累加；二是放大较大误差的影响。
- \(\frac{1}{n} \sum_{i=1}^{n}\)：对所有样本的平方误差求平均。
- MSE的值越小，说明预测越准确。

**MSE的性质**：MSE对异常值敏感，因为较大误差会被平方放大。当数据中存在大量异常值时，可考虑Huber Loss等更鲁棒的选择。

```python
# 导入数值计算库
import numpy as np

def mse(y_true, y_pred):
    """均方误差 - 从零实现"""
    # y_true为真实值数组，y_pred为预测值数组
    # 计算逐元素差值，平方后求均值
    return np.mean((y_true - y_pred) ** 2)

# 使用PyTorch内置的MSE损失
criterion = nn.MSELoss()
# 计算预测张量与真实张量之间的MSE
loss = criterion(y_pred_tensor, y_true_tensor)
```

### 3.2 交叉熵损失

交叉熵损失广泛用于分类问题，衡量的是实际输出概率分布与预测概率分布之间的差异。

**信息论背景**：交叉熵源自信息论，衡量用预测分布 \(q\) 去编码真实分布 \(p\) 所需的平均信息量。当 \(q\) 越接近 \(p\)，交叉熵越小。

对于二分类问题，二元交叉熵损失为：

\[
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
\]

**公式解释**：
- \(y_i \in \{0, 1\}\)：第 \(i\) 个样本的真实标签。
- \(\hat{y}_i \in (0, 1)\)：模型预测为正类的概率（通常经过Sigmoid）。
- 当 \(y_i = 1\) 时，第一项 \(y_i \log(\hat{y}_i) = \log(\hat{y}_i)\) 生效；若 \(\hat{y}_i\) 接近1，\(\log(\hat{y}_i)\) 接近0，损失小；若 \(\hat{y}_i\) 接近0，损失很大。
- 当 \(y_i = 0\) 时，第二项 \((1-y_i)\log(1-\hat{y}_i) = \log(1-\hat{y}_i)\) 生效；若 \(\hat{y}_i\) 接近0，损失小；若接近1，损失大。
- 负号：因为 \(\log\) 在 \((0, 1)\) 区间为负，取负号后损失为正。
- \(\frac{1}{n} \sum\)：对所有样本求平均。

对于多分类问题，交叉熵损失结合Softmax函数使用：

\[
\text{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
\]

**公式解释**：
- \(C\)：类别总数。
- \(y_i\)：真实标签的one-hot编码。例如3分类中，真实类别为第2类，则 \(\boldsymbol{y} = [0, 1, 0]\)。
- \(\hat{y}_i\)：模型预测为第 \(i\) 类的概率，由Softmax计算得到，满足 \(\sum_i \hat{y}_i = 1\)。
- 由于 \(y_i\) 只有一个位置为1，其余为0，所以求和实际只取真实类别对应的那一项：\(\text{CE} = -\log(\hat{y}_{true})\)。
- 若预测为真实类别的概率 \(\hat{y}_{true}\) 接近1，则损失接近0；若接近0，损失趋于正无穷。

**Softmax函数**：

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
\]

**公式解释**：
- \(z_i\)：第 \(i\) 类的logit（未归一化的分数）。
- \(e^{z_i}\)：将logit映射为正数。
- \(\sum_{j=1}^{C} e^{z_j}\)：所有类别指数的和，作为归一化因子。
- 输出 \(\hat{y}_i\) 满足 \(\hat{y}_i \in (0, 1)\) 且 \(\sum_i \hat{y}_i = 1\)，可以解释为概率分布。

```python
# 二元交叉熵 - 从零实现
def binary_cross_entropy(y_true, y_pred):
    # 设置极小值epsilon，防止log(0)导致数值溢出
    epsilon = 1e-15
    # 将预测值裁剪到[epsilon, 1-epsilon]区间
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # 计算二元交叉熵
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 多分类交叉熵 - PyTorch实现
criterion = nn.CrossEntropyLoss()
# 注意：PyTorch的CrossEntropyLoss已包含Softmax，输入应为logits而非概率
loss = criterion(logits, labels)
```

### 3.3 损失函数选择指南

| 任务类型 | 推荐损失函数 | 输出层激活函数 |
|---------|------------|--------------|
| 回归 | MSE / MAE / Huber | 恒等映射 |
| 二分类 | 二元交叉熵 | Sigmoid |
| 多分类 | 交叉熵 | Softmax |

选择损失函数时还需考虑数据特性：不平衡数据集可使用加权交叉熵，给予少数类更高的权重。


## 四、反向传播与梯度下降

### 4.1 核心思想

神经网络训练的基本思想可以概括为一个反复迭代的过程：先给出一个预测结果，计算它与真实值之间的差距（损失），然后根据这个差距调整参数，使下一次预测更接近真实值。

反向传播（Backpropagation）的核心是链式法则。假设损失 \(L\) 依赖于某层权重 \(w\)，而 \(w\) 通过中间变量 \(z\) 影响 \(L\)，则：

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}
\]

**公式解释**：
- \(\frac{\partial L}{\partial w}\)：损失对权重 \(w\) 的梯度，表示 \(w\) 微小变化时损失如何变化。
- \(\frac{\partial L}{\partial z}\)：损失对中间变量 \(z\) 的梯度，由后一层反向传播得到。
- \(\frac{\partial z}{\partial w}\)：中间变量 \(z\) 对 \(w\) 的梯度，由前向传播的局部导数决定。
- 链式法则将复杂导数分解为简单导数的乘积，使得逐层计算成为可能。

**参数更新规则**：

\[
w \leftarrow w - \eta \frac{\partial L}{\partial w}
\]

**公式解释**：
- \(w\)：待更新的权重。
- \(\eta\)：学习率（learning rate），控制每步更新的幅度。\(\eta\) 过大导致震荡，过小导致收敛慢。
- \(\frac{\partial L}{\partial w}\)：损失对 \(w\) 的梯度。
- 负号：沿梯度的反方向更新，因为梯度指向损失增大的方向。
- \(w - \eta \frac{\partial L}{\partial w}\)：沿负梯度方向移动一小步。

### 4.2 梯度下降的三种变体

**批量梯度下降**：每次使用全部训练数据计算梯度。收敛稳定但计算代价大，大数据集不可行。

**随机梯度下降**：每次仅使用单个样本。更新频繁、速度快，但梯度方差大，损失函数震荡剧烈。

**小批量梯度下降**：折中方案，每次使用一个小批量（通常50～256个样本）。充分利用GPU的矩阵运算并行性，兼具稳定性和速度，是当前实践中的事实标准。

### 4.3 从零实现梯度下降

```python
# 导入数值计算库
import numpy as np

def numerical_gradient(f, x):
    """数值微分法计算梯度"""
    # 步长h，取1e-4，太小会因浮点精度产生误差，太大会导致近似不准
    h = 1e-4
    # 初始化梯度数组，形状与x相同
    grad = np.zeros_like(x)
    # 遍历x的每个元素
    for idx in range(x.size):
        # 保存当前元素的原值
        tmp_val = x[idx]
        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 中心差分法计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 恢复原值
        x[idx] = tmp_val
    # 返回梯度数组
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """梯度下降法"""
    # 初始化参数
    x = init_x
    # 迭代step_num次
    for i in range(step_num):
        # 计算当前点的梯度
        grad = numerical_gradient(f, x)
        # 沿负梯度方向更新参数
        x -= lr * grad
    # 返回优化后的参数
    return x
```

### 4.4 训练中的核心挑战

在实际训练中，梯度下降面临几个关键挑战：

1. **学习率选择困难**：过小导致收敛极慢，过大则震荡甚至发散。
2. **梯度尺度差异**：不同参数的梯度尺度差异悬殊，统一学习率难以兼顾。
3. **鞍点问题**：在高维非凸优化中，真正的障碍往往不是局部最小值，而是**鞍点**——某些维度梯度为正、另一些为负，梯度趋近于零，SGD极难逃脱。
4. **梯度消失与爆炸**：在深层网络中，梯度逐层相乘，可能指数级衰减或增长。

这些问题催生了后续一系列优化算法的改进。


## 五、网络优化方法

优化算法的演进围绕两个核心思路展开：引入**动量**以加速收敛并逃离鞍点，以及**自适应学习率**以适应不同参数的梯度尺度。

### 5.1 Momentum（动量法）

动量法的动机来自物理直觉：想象一个球从山坡滚下，它在平坦区域会积累动量从而继续前进，不易被小的坑洼困住。

**数学表达**：

\[
v_t = \beta v_{t-1} + (1-\beta) g_t
\]

\[
w_{t+1} = w_t - \eta v_t
\]

**公式解释**：
- \(v_t\)：第 \(t\) 步的速度（动量项），累积了历史梯度信息。
- \(\beta\)：动量系数，典型值0.9。\(\beta\) 越大，历史梯度影响越大，更新越平滑。
- \(v_{t-1}\)：上一步的速度。
- \(g_t\)：第 \(t\) 步的梯度 \(\frac{\partial L}{\partial w}\)。
- \((1-\beta) g_t\)：当前梯度的加权项。
- \(v_t = \beta v_{t-1} + (1-\beta) g_t\)：速度是历史速度与当前梯度的加权平均。
- \(w_{t+1} = w_t - \eta v_t\)：用速度而非梯度更新权重。

**物理意义**：当梯度方向持续一致时，动量累积，加速前进；当梯度方向反复变化时，正负抵消，减少震荡。

### 5.2 RMSprop

RMSprop通过除以梯度平方的指数移动平均来调整学习率，使各参数获得自适应的学习率：

\[
s_t = \gamma s_{t-1} + (1-\gamma) g_t^2
\]

\[
w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
\]

**公式解释**：
- \(s_t\)：第 \(t\) 步的梯度平方的指数移动平均，反映梯度的"尺度"。
- \(\gamma\)：衰减率，典型值0.9或0.99。
- \(g_t^2\)：当前梯度的平方（逐元素）。
- \(\epsilon\)：极小常数（如 \(10^{-8}\)），防止分母为零。
- \(\frac{\eta}{\sqrt{s_t + \epsilon}}\)：自适应学习率。梯度大的参数学习率被缩小，梯度小的参数学习率被放大。
- 这有效解决了AdaGrad学习率单调递减导致训练后期更新过慢的问题。

### 5.3 Adam

Adam（Adaptive Moment Estimation）是目前最主流的优化器，它结合了Momentum和RMSprop的优点，同时使用一阶动量（梯度的指数移动平均）和二阶动量（梯度平方的指数移动平均）。

**一阶动量（类似Momentum）**：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

**二阶动量（类似RMSprop）**：

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

**偏差修正**：

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

**参数更新**：

\[
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\]

**公式解释**：
- \(m_t\)：梯度的一阶动量，类似Momentum中的速度。
- \(v_t\)：梯度的二阶动量，类似RMSprop中的梯度平方平均。
- \(\beta_1, \beta_2\)：一阶和二阶动量的衰减率，典型值0.9和0.999。
- \(\hat{m}_t, \hat{v}_t\)：偏差修正后的动量。因为 \(m_0 = 0, v_0 = 0\)，初期估计偏向0，除以 \(1-\beta^t\) 可修正。
- \(t\)：当前步数（从1开始）。
- \(\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t\)：结合了一阶动量的方向和二阶动量的自适应尺度。

```python
# 导入优化器模块
import torch.optim as optim

# SGD优化器，学习率0.01，动量0.9
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# RMSprop优化器，学习率0.001，衰减率0.99
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Adam优化器，学习率0.001，一阶/二阶动量衰减率0.9/0.999
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**优化器选择建议**：

| 优化器 | 适用场景 | 特点 |
|--------|---------|------|
| SGD + Momentum | 需要精细调参、追求最优泛化性能 | 简单稳健，对学习率敏感 |
| RMSprop | 非平稳目标、稀疏梯度 | 自适应学习率 |
| Adam | 大多数任务的默认选择 | 收敛快、稳定性好，泛化性能有时略逊于SGD |

值得注意的是，Adam虽然收敛快，但在某些应用中泛化性能可能不如SGD。实践中推荐先用Adam快速获得baseline，再用SGD+momentum精细调优。


## 六、正则化方法

过拟合是深度学习中常见的核心问题——模型在训练集上表现极好，但在测试集上表现很差。正则化技术通过控制模型复杂度来缓解过拟合。

### 6.1 权重衰减（L2正则化）

权重衰减通过在损失函数中添加L2范数惩罚项，使权重不会过大，从而控制模型复杂度：

\[
\tilde{L}(\boldsymbol{w}, b) = L(\boldsymbol{w}, b) + \frac{\lambda}{2} \|\boldsymbol{w}\|^2
\]

**公式解释**：
- \(\tilde{L}\)：加入正则项后的总损失。
- \(L(\boldsymbol{w}, b)\)：原始损失（如交叉熵、MSE）。
- \(\lambda\)：正则化系数，控制惩罚强度。\(\lambda\) 越大，权重被压缩得越厉害。
- \(\|\boldsymbol{w}\|^2 = \sum_i w_i^2\)：权重向量的L2范数平方。
- \(\frac{1}{2}\)：系数，求导后与2抵消，使梯度形式简洁。
- 惩罚项 \(\frac{\lambda}{2} \|\boldsymbol{w}\|^2\) 使权重趋向于小值，从而降低模型复杂度。

加入正则项后，参数更新规则变为：

\[
w_{t+1} = (1 - \eta\lambda) w_t - \eta \frac{\partial L}{\partial w_t}
\]

**公式解释**：
- \((1 - \eta\lambda) w_t\)：权重先乘以一个小于1的因子，即"衰减"。
- \(\eta \frac{\partial L}{\partial w_t}\)：原始梯度更新项。
- 每步更新时权重都会缩小一点，这正是"权重衰减"名称的由来。

```python
# 方式一：在优化器中指定weight_decay（推荐）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 方式二：手动在损失中添加L2惩罚项
def l2_penalty(w):
    # 计算权重的L2范数平方除以2
    return torch.sum(w.pow(2)) / 2

# 总损失 = 原始损失 + lambda * L2惩罚
loss = criterion(y_hat, y) + lamda * l2_penalty(net[0].weight)
```

正则化权重 \(\lambda\) 是控制模型复杂度的超参数：\(\lambda = 0\) 时无正则化效果；\(\lambda \to \infty\) 时权重趋向于零。

### 6.2 Dropout

Dropout（暂退法）是一种简单而有效的正则化技术。在训练过程中，Dropout随机将一部分神经元的输出置零，从而减少神经元之间的共适应性，迫使网络学习更鲁棒的特征表示。

具体来说，对每个元素以概率 \(p\) 将其置零，并以概率 \(1-p\) 将其放大 \(\frac{1}{1-p}\) 倍：

\[
x_i' = \begin{cases} 0 & \text{以概率 } p \\ \frac{x_i}{1-p} & \text{其他情况} \end{cases}
\]

**公式解释**：
- \(x_i\)：原始激活值。
- \(p\)：丢弃概率，通常取0.2到0.5。
- 以概率 \(p\) 将神经元置零，即"丢弃"该神经元。
- 以概率 \(1-p\) 保留神经元，但将值放大 \(\frac{1}{1-p}\) 倍。
- 这种缩放保证了输出期望不变：\(\mathbb{E}[x'] = (1-p) \cdot \frac{x_i}{1-p} + p \cdot 0 = x_i\)。

**重要**：Dropout只在训练阶段使用，推理阶段直接返回输入，以保证确定性的输出。

```python
# 定义带Dropout的多层感知机
class DropoutMLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens1, num_hiddens2, num_outputs):
        # 调用父类构造函数
        super().__init__()
        # 第一层全连接：输入 -> 隐藏层1
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        # 第二层全连接：隐藏层1 -> 隐藏层2
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        # 第三层全连接：隐藏层2 -> 输出
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Dropout层，丢弃概率0.2
        self.dropout1 = nn.Dropout(0.2)
        # Dropout层，丢弃概率0.5
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, X):
        # 第一层：线性 -> ReLU -> Dropout
        H1 = self.dropout1(self.relu(self.lin1(X)))
        # 第二层：线性 -> ReLU -> Dropout
        H2 = self.dropout2(self.relu(self.lin2(H1)))
        # 输出层：只做线性变换
        return self.lin3(H2)
```

通常，靠近输入层的Dropout概率设置较小（如0.2），靠近输出层设置较大（如0.5）。

### 6.3 批量归一化

批量归一化（Batch Normalization, BN）通过对每一层的输入进行标准化，使其均值接近0、方差接近1，从而加速训练并允许使用更大的学习率。

**数学表达**：

对一个小批量的输入 \(\mathcal{B} = \{x_1, ..., x_m\}\)：

\[
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
\]

\[
\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2
\]

\[
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
\]

\[
y_i = \gamma \hat{x}_i + \beta
\]

**公式解释**：
- \(m\)：小批量中的样本数。
- \(\mu_\mathcal{B}\)：小批量均值。
- \(\sigma_\mathcal{B}^2\)：小批量方差。
- \(\hat{x}_i\)：标准化后的值，均值为0，方差为1。
- \(\epsilon\)：极小常数，防止除零。
- \(\gamma\)：可学习的缩放参数，初始化为1。
- \(\beta\)：可学习的平移参数，初始化为0。
- \(y_i\)：BN层的最终输出。通过 \(\gamma, \beta\)，网络可以学习恢复原始分布，保留表达能力。

```python
# 定义带批量归一化的网络
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层全连接：784 -> 256
        self.fc1 = nn.Linear(784, 256)
        # 第一层BN，输入维度256
        self.bn1 = nn.BatchNorm1d(256)
        # 第二层全连接：256 -> 128
        self.fc2 = nn.Linear(256, 128)
        # 第二层BN
        self.bn2 = nn.BatchNorm1d(128)
        # 输出层：128 -> 10
        self.fc3 = nn.Linear(128, 10)
        # ReLU激活
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层：线性 -> BN -> ReLU
        x = self.relu(self.bn1(self.fc1(x)))
        # 第二层：线性 -> BN -> ReLU
        x = self.relu(self.bn2(self.fc2(x)))
        # 输出层
        return self.fc3(x)
```

### 6.4 其他正则化策略

- **早停法**：在验证集损失开始上升时停止训练，防止过拟合。
- **数据增强**：通过对训练样本进行随机变换（旋转、翻转、裁剪等）增加数据多样性。
- **L1正则化**：使用权重绝对值之和作为惩罚项，倾向于产生稀疏解。


## 七、实战案例

### 7.1 MNIST手写数字识别

MNIST是深度学习入门的经典数据集，包含60000张训练图像和10000张测试图像，每张为28×28的灰度手写数字。

```python
# 导入PyTorch
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim
# 导入torchvision中的数据集和变换
from torchvision import datasets, transforms

# 1. 数据加载与预处理
# 定义图像变换流水线
transform = transforms.Compose([
    # 将PIL图像转为张量，像素值归一化到[0,1]
    transforms.ToTensor(),
    # 标准化：减去均值0.1307，除以标准差0.3081（MNIST全局统计值）
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载训练集，download=True表示若本地无数据则自动下载
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
# 加载测试集
test_set = datasets.MNIST('./data', train=False, transform=transform)
# 训练集DataLoader，批量大小64，每轮打乱顺序
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
# 测试集DataLoader，批量大小1000，不打乱
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# 2. 定义CNN模型
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层：输入1通道，输出32通道，卷积核3x3，步长1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二个卷积层：输入32通道，输出64通道，卷积核3x3，步长1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout2d：对卷积特征图整体丢弃，概率0.25
        self.dropout1 = nn.Dropout2d(0.25)
        # Dropout2d：概率0.5
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层：64*12*12=9216 -> 128
        self.fc1 = nn.Linear(9216, 128)
        # 输出层：128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 + ReLU
        x = torch.relu(self.conv1(x))
        # 第二层卷积 + ReLU
        x = torch.relu(self.conv2(x))
        # 2x2最大池化，特征图尺寸从24x24变为12x12
        x = torch.max_pool2d(x, 2)
        # Dropout
        x = self.dropout1(x)
        # 展平为(batch, 9216)
        x = torch.flatten(x, 1)
        # 全连接 + ReLU
        x = torch.relu(self.fc1(x))
        # Dropout
        x = self.dropout2(x)
        # 输出层
        return self.fc2(x)

# 3. 训练配置
# 优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 实例化模型并移动到设备
model = MNISTNet().to(device)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 定义单轮训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    # 切换到训练模式（启用Dropout、BN更新）
    model.train()
    # 累计损失、正确数、样本数
    total_loss, correct, total = 0, 0, 0
    # 遍历数据加载器
    for data, target in loader:
        # 数据和标签移到设备
        data, target = data.to(device), target.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失（乘batch大小用于后续平均）
        total_loss += loss.item() * data.size(0)
        # 取最大logit对应的类别为预测
        pred = output.argmax(dim=1)
        # 统计正确预测数
        correct += pred.eq(target).sum().item()
        # 统计总样本数
        total += data.size(0)
    # 返回平均损失和准确率
    return total_loss / total, correct / total

# 5. 训练与评估
for epoch in range(10):
    # 执行一轮训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    # 打印本轮结果
    print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}')
```

该模型在MNIST上通常能达到99%以上的测试准确率。

### 7.2 CIFAR-10图像分类

CIFAR-10包含10个类别的32×32彩色图像，比MNIST更具挑战性。

```python
# 导入数据集和变换
from torchvision import datasets, transforms

# 训练集数据增强 + 预处理
transform_train = transforms.Compose([
    # 随机裁剪：先padding=4，再裁剪为32x32
    transforms.RandomCrop(32, padding=4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 转张量
    transforms.ToTensor(),
    # 按CIFAR-10统计值标准化
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 加载训练集
train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
# 加载测试集（仅标准化，不做数据增强）
test_set = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
]))

# 训练集DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
# 测试集DataLoader
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

# 带BN和Dropout的CNN
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # Block 1
            # 卷积：3 -> 64，3x3，padding=1
            nn.Conv2d(3, 64, 3, padding=1),
            # BN
            nn.BatchNorm2d(64),
            # ReLU
            nn.ReLU(),
            # 卷积：64 -> 64
            nn.Conv2d(64, 64, 3, padding=1),
            # BN
            nn.BatchNorm2d(64),
            # ReLU
            nn.ReLU(),
            # 2x2最大池化，尺寸减半
            nn.MaxPool2d(2),
            # Dropout 0.2
            nn.Dropout2d(0.2),
            # Block 2
            # 卷积：64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            # BN
            nn.BatchNorm2d(128),
            # ReLU
            nn.ReLU(),
            # 卷积：128 -> 128
            nn.Conv2d(128, 128, 3, padding=1),
            # BN
            nn.BatchNorm2d(128),
            # ReLU
            nn.ReLU(),
            # 池化
            nn.MaxPool2d(2),
            # Dropout 0.3
            nn.Dropout2d(0.3),
            # Block 3
            # 卷积：128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            # BN
            nn.BatchNorm2d(256),
            # ReLU
            nn.ReLU(),
            # 全局平均池化，输出1x1x256
            nn.AdaptiveAvgPool2d(1),
        )
        # 分类器部分
        self.classifier = nn.Sequential(
            # 展平
            nn.Flatten(),
            # 全连接：256 -> 128
            nn.Linear(256, 128),
            # ReLU
            nn.ReLU(),
            # Dropout 0.5
            nn.Dropout(0.5),
            # 输出层：128 -> 10
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 特征提取 + 分类
        return self.classifier(self.features(x))

# 实例化模型并移动到设备
model = CIFAR10Net().to(device)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# Adam优化器，带权重衰减
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# 余弦退火学习率调度器，T_max=50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 训练50轮
for epoch in range(50):
    # 执行一轮训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    # 更新学习率
    scheduler.step()
    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}')
```

该模型融合了多种技术：批量归一化加速训练、Dropout抑制过拟合、权重衰减控制模型复杂度、数据增强增加样本多样性、余弦退火学习率调度优化后期收敛。在CIFAR-10上通常可达到90%以上的准确率。


## 八、总结

从单个人工神经元到深层神经网络，从损失函数的选择到优化算法的迭代改进，从正则化技术到完整的实战项目，本文梳理了神经网络入门的完整知识体系。核心要点回顾如下：

1. **神经网络本质**：多层非线性变换的堆叠，通过前向传播产生预测，通过反向传播和梯度下降更新参数。
2. **损失函数**：回归用MSE，分类用交叉熵，选择需匹配任务类型和数据特性。
3. **优化方法**：从SGD到Adam，自适应学习率和动量机制是两大核心改进方向。
4. **正则化**：权重衰减、Dropout、批量归一化、早停法和数据增强是防止过拟合的有效手段。
5. **实践路径**：从MNIST入门到CIFAR-10进阶，逐步掌握网络设计、训练调优和性能评估的完整流程。

深度学习的精髓在于"动手"二字。建议读者在理解原理的基础上，亲手实现每一个代码示例，在真实数据集上反复实验和调参，逐步培养对模型训练过程的直觉。
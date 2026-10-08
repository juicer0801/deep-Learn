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

{% asset_img Snipaste_2026-10-06_10-50-04.png "Hexo 博客封面示例" %}

演示网址：
playground.tensorflow.org

#### 2.2.1 Sigmoid函数

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

图解：
{% asset_img Snipaste_2026-10-06_10-59-23.png "Hexo 博客封面示例" %}

**公式解释**：
- \(e\)：自然对数的底数，约等于2.718。
- \(e^{-z}\)：指数函数，当 \(z\) 很大时 \(e^{-z}\) 趋近于0，当 \(z\) 很小时 \(e^{-z}\) 趋近于正无穷。
- 当 \(z \to +\infty\) 时，\(\sigma(z) \to 1\)；当 \(z \to -\infty\) 时，\(\sigma(z) \to 0\)；当 \(z = 0\) 时，\(\sigma(0) = 0.5\)。
- 因此Sigmoid将任意实数压缩到 \((0, 1)\) 区间，可以解释为概率。

**导数公式**：

\[
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
\]

**推导过程**：

令 \(\sigma(z) = (1 + e^{-z})^{-1}\)，对 \(z\) 求导：

\[
\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})
\]

因为 \(\frac{d}{dz}(1 + e^{-z}) = -e^{-z}\)，代入得：

\[
\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}
\]

将上式拆分为两个因子的乘积：

\[
\frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}}
\]

而 \(\frac{e^{-z}}{1 + e^{-z}} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)\)。

因此：

\[
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
\]

**导数解释**：
- 当 \(\sigma(z)\) 接近 0 或 1 时，\(\sigma'(z)\) 接近 0，梯度趋近于零，即**梯度消失**。
- 导数最大值为 0.25（在 \(z=0\) 处），意味着即使处于最优位置，梯度也只有 0.25，多层叠加后迅速衰减。
- 导数可直接由函数值计算，无需重新计算指数，计算效率高。

**缺点**：当输入绝对值较大时（如 \(z > 6\) 或 \(z < -6\)），函数曲线变得非常平坦，导数趋近于零，导致**梯度消失**问题。在深层网络中，反向传播时梯度逐层相乘，很容易变成0，使浅层参数无法更新。

**作用**
sigmoid 函数可以将任意的输入映射到 (0, 1) 之间，当输入的值大致在 <-6 或者 >6 时，意味着输入任何值得到的激活值都是差不多的，这样会丢失部分的信息。比如：输入 100 和输出 10000 经过 sigmoid 的激活值几乎都是等于 1 的，但是输入的数据之间相差 100 倍的信息就丢失了。

对于 sigmoid 函数而言，输入值在 [-6, 6] 之间输出值才会有明显差异，输入值在 [-3, 3] 之间才会有比较好的效果。

通过上述导数图像，我们发现导数数值范围是 (0, 0.25)，当输入 <-6 或者 >6 时，sigmoid 激活函数图像的导数接近为 0，此时网络参数将更新极其缓慢，或者无法更新（W新 = W旧 - 学习率 * 梯度，梯度即损失函数的导数，接近于零则该公式W新几乎不变）。

一般来说， sigmoid 网络在 5 层之内（0.25**5几乎为0.0009）就会产生梯度消失现象。而且，该激活函数并不是以 0 为中心的，所以在实践中这种激活函数使用的很少。sigmoid函数一般只用于二分类的输出层。


#### 2.2.2 Tanh函数

\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}
\]

**图解**：
{% asset_img Snipaste_2026-10-06_11-12-54.png "Hexo 博客封面示例" %}

**公式解释**：
- 分子 \(e^z - e^{-z}\)：当 \(z\) 为正且很大时，\(e^z\) 主导，分子为正且很大；当 \(z\) 为负且很小时，\(e^{-z}\) 主导，分子为负。
- 分母 \(e^z + e^{-z}\)：始终为正，且不小于2。
- 因此 \(\tanh(z)\) 的输出范围是 \((-1, 1)\)。
- \(\tanh(0) = 0\)，输出以零为中心，这比Sigmoid更有利于后续层的训练。

**导数公式**：

\[
\tanh'(z) = 1 - \tanh^2(z)
\]

**推导过程**：

令 \(u = e^z - e^{-z}\)，\(v = e^z + e^{-z}\)，则 \(\tanh(z) = \frac{u}{v}\)。

根据商的求导法则 \(\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}\)：

\[
u' = e^z + e^{-z} = v, \quad v' = e^z - e^{-z} = u
\]

代入得：

\[
\tanh'(z) = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \frac{u^2}{v^2} = 1 - \tanh^2(z)
\]

**导数解释**：
- 当 \(\tanh(z)\) 接近 \(\pm 1\) 时，导数趋近于0，同样存在梯度消失问题。
- 导数最大值为 1（在 \(z=0\) 处），是 Sigmoid 最大导数的 4 倍，因此 Tanh 在浅层网络中收敛更快。
- 输出以零为中心，避免了 Sigmoid 的"非零均值"问题，使后续层输入分布更稳定。

**优点**：输出以零为中心，缓解了Sigmoid的"非零均值"问题。**缺点**：仍然存在梯度消失问题。


**作用**：
Tanh 函数将输入映射到 (-1, 1) 之间，图像以 0 为中心，在 0 点对称，当输入 大概<-3 或者 >3 时将被映射为 -1 或者 1。其导数值范围 (0, 1)，当输入的值大概 <-3 或者 > 3 时，其导数近似 0。

与 Sigmoid 相比，它是以 0 为中心的，且梯度相对于sigmoid大，使得其收敛速度要比 Sigmoid 快，减少迭代次数。然而，从图中可以看出，Tanh 两侧的导数也为 0，同样会造成梯度消失。

若使用时可在隐藏层使用tanh函数，在输出层使用sigmoid函数。


#### 2.2.3 ReLU函数

\[
\text{ReLU}(z) = \max(0, z)
\]

**图解**：
{% asset_img Snipaste_2026-10-06_11-20-37.png "Hexo 博客封面示例" %}

**公式解释**：
- 当 \(z > 0\) 时，输出为 \(z\) 本身，导数恒为1。
- 当 \(z \le 0\) 时，输出为0，导数为0。
- 计算非常简单，只需一次比较操作。

**导数公式**：

\[
\text{ReLU}'(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \\ \text{未定义（通常取 0）}, & z = 0 \end{cases}
\]

**导数解释**：
- 正区间导数恒为 1，这是 ReLU 缓解梯度消失的关键：梯度不会因激活函数而衰减。
- 负区间导数为 0，使神经元输出为 0，网络具有稀疏性。
- \(z=0\) 处不可导，但实践中随机初始化后恰好为 0 的概率极低，通常约定该点导数为 0。


**优点**：
1. 正区间梯度恒为1，有效缓解梯度消失问题。
2. 计算简单，训练速度快。
3. 稀疏激活：负半轴输出为0，使得网络具有一定稀疏性。

**缺点**：存在"Dead ReLU"问题。当某个神经元的输入持续为负时，其梯度始终为零，权重永远不更新，该神经元相当于"死亡"。

**作用**：
ReLU 激活函数将小于 0 的值映射为 0，而大于 0 的值则保持不变，它更加重视正信号，而忽略负信号，这种激活函数运算更为简单，能够提高模型的训练效率。
当x<0时，ReLU导数为0，而当x>0时，则不存在饱和问题。所以，ReLU 能够在x>0时保持梯度不衰减，从而缓解梯度消失问题。然而，随着训练的推进，部分输入会落入小于0区域，导致对应权重无法更新。这种现象被称为“神经元死亡”。
ReLU是目前最常用的激活函数。与sigmoid相比，RELU的优势是：
采用sigmoid函数，计算量大（指数运算），反向传播求误差梯度时，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。 sigmoid函数反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练。 Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，模型就变简单了，缓解了过拟合问题的发生。

#### 2.2.4 Leaky ReLU

\[
\text{LeakyReLU}(z) = \max(\alpha z, z)
\]

**公式解释**：
- \(\alpha\)：一个小常数，通常取0.01或0.1。
- 当 \(z > 0\) 时，输出 \(z\)，与ReLU相同。
- 当 \(z \le 0\) 时，输出 \(\alpha z\)，是一个很小的负数，而不是0。
- 这样即使输入为负，梯度仍然为 \(\alpha\)（非零），避免了Dead ReLU问题。

**导数公式**：

\[
\text{LeakyReLU}'(z) = \begin{cases} 1, & z > 0 \\ \alpha, & z < 0 \\ \text{未定义（通常取 } \alpha\text{）}, & z = 0 \end{cases}
\]

**导数解释**：
- 正区间导数恒为 1，与 ReLU 相同。
- 负区间导数恒为 \(\alpha\)（非零），这是 Leaky ReLU 相比 ReLU 的核心改进：即使输入为负，梯度仍能传播，神经元不会"死亡"。
- \(\alpha\) 越小，负区间越接近 ReLU；\(\alpha\) 越大，负区间信息保留越多，但可能损失稀疏性。

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

#### 2.2.5 Softmax 函数（多分类专用）

Softmax 是**多分类任务输出层**的标准激活函数，它将一个 \(C\) 维的实数向量（logits）转换为概率分布。

**函数定义**：

对 \(C\) 维输入向量 \(\boldsymbol{z} = (z_1, z_2, ..., z_C)\)，Softmax 的第 \(i\) 个输出为：

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
\]

**公式解释**：
- \(z_i\)：第 \(i\) 类的 logit（未归一化的分数），可以是任意实数。
- \(e^{z_i}\)：将 logit 映射为正数（指数函数恒为正）。
- \(\sum_{j=1}^{C} e^{z_j}\)：所有类别指数的和，作为归一化因子。
- 输出 \(s_i = \text{Softmax}(z_i)\) 满足两个性质：
  - **非负性**：\(s_i \in (0, 1)\)，因为分子分母均为正。
  - **归一性**：\(\sum_{i=1}^{C} s_i = 1\)，因为分子之和等于分母。
- 因此 \(s_i\) 可以解释为"样本属于第 \(i\) 类的概率"。

**图解**：


**数值稳定性技巧**：

直接计算 \(e^{z_i}\) 时，若 \(z_i\) 很大（如 1000），会溢出为 `inf`。实践中通常减去最大值：

\[
\text{Softmax}(z_i) = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{C} e^{z_j - \max_j z_j}}
\]

**解释**：分子分母同乘 \(e^{-\max_j z_j}\)，数学上等价，但最大指数变为 \(e^0 = 1\)，其余项均 \(\le 1\)，避免溢出。

**导数公式（雅可比矩阵）**：

Softmax 是多输入多输出函数，其导数是一个 \(C \times C\) 的雅可比矩阵：

\[
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j)
\]

其中 \(\delta_{ij}\) 是 **Kronecker delta**，定义为：

\[
\delta_{ij} = \begin{cases} 1, & i = j \\ 0, & i \ne j \end{cases}
\]

**分情况展开**：

\[
\frac{\partial s_i}{\partial z_j} = \begin{cases} s_i (1 - s_i), & i = j \\ -s_i s_j, & i \ne j \end{cases}
\]

**推导过程**：

令 \(S = \sum_{k=1}^{C} e^{z_k}\) 为分母，则 \(s_i = \frac{e^{z_i}}{S}\)。

**情形一：\(i = j\)**（对角元素）

\[
\frac{\partial s_i}{\partial z_i} = \frac{\partial}{\partial z_i}\left(\frac{e^{z_i}}{S}\right)
\]

根据商的求导法则：

\[
= \frac{e^{z_i} \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_i}}{S^2}
\]

而 \(\frac{\partial S}{\partial z_i} = \frac{\partial}{\partial z_i}\sum_k e^{z_k} = e^{z_i}\)（只有第 \(i\) 项对 \(z_i\) 敏感），代入：

\[
= \frac{e^{z_i} S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \cdot \frac{S - e^{z_i}}{S} = s_i (1 - s_i)
\]

**情形二：\(i \ne j\)**（非对角元素）

\[
\frac{\partial s_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{S}\right)
\]

分子 \(e^{z_i}\) 与 \(z_j\) 无关，故视为常数；分母 \(S\) 对 \(z_j\) 的导数为 \(e^{z_j}\)：

\[
= \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -s_i s_j
\]

**统一形式**：

将两种情形合并，即可得到 \(\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)\)。

**导数解释**：
- **对角元素** \(s_i(1-s_i)\)：当第 \(i\) 类概率 \(s_i\) 接近 0 或 1 时，该元素趋近于 0，梯度小；当 \(s_i = 0.5\) 时最大，为 0.25。
- **非对角元素** \(-s_i s_j\)：始终为负，表示提高第 \(j\) 类的 logit 会降低第 \(i\) 类的概率。
- 雅可比矩阵是**对称矩阵**，因为 \(\frac{\partial s_i}{\partial z_j} = -s_i s_j = \frac{\partial s_j}{\partial z_i}\)（当 \(i \ne j\)）。
- 雅可比矩阵每行之和为 0：\(\sum_j s_i(\delta_{ij} - s_j) = s_i(1 - \sum_j s_j) = s_i(1-1) = 0\)，这反映了 Softmax 输出之和恒为 1 的约束。

**与交叉熵的配合**：

实践中 Softmax 几乎总是与交叉熵损失配合使用。若真实标签为 \(y\)（one-hot），预测概率为 \(s\)，则损失 \(L = -\sum_i y_i \log s_i\)。此时梯度有一个极为简洁的形式：

\[
\frac{\partial L}{\partial z_i} = s_i - y_i
\]

**推导过程**：

\[
L = -\sum_k y_k \log s_k
\]

对 \(z_i\) 求导，只有 \(s_k\) 依赖 \(z_i\)，利用链式法则：

\[
\frac{\partial L}{\partial z_i} = -\sum_k y_k \cdot \frac{1}{s_k} \cdot \frac{\partial s_k}{\partial z_i}
\]

代入 Softmax 导数 \(\frac{\partial s_k}{\partial z_i} = s_k(\delta_{ki} - s_i)\)：

\[
= -\sum_k y_k \cdot \frac{1}{s_k} \cdot s_k(\delta_{ki} - s_i) = -\sum_k y_k (\delta_{ki} - s_i)
\]

展开求和：

\[
= -\left[\sum_k y_k \delta_{ki} - \sum_k y_k s_i\right] = -\left[y_i - s_i \sum_k y_k\right]
\]

由于 \(y\) 是 one-hot 编码，\(\sum_k y_k = 1\)，故：

\[
\frac{\partial L}{\partial z_i} = s_i - y_i
\]

**解释**：这个结果非常优雅——预测概率与真实标签的差值就是梯度。若预测正确（\(s_i\) 接近 1，\(y_i = 1\)），梯度接近 0；若预测错误（\(s_i\) 接近 0，\(y_i = 1\)），梯度为负，参数会向提高该类别概率的方向更新。这也是 PyTorch 的 `nn.CrossEntropyLoss` 内部直接接收 logits 的原因：它将 Softmax 与交叉熵合并为一个数值稳定的运算，避免单独计算 Softmax 再取对数带来的精度损失。

**作用**：
Softmax 就是将网络输出的 logits 通过 softmax 函数，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们将它理解成概率，选取概率最大（也就是值对应最大的）节点，作为我们的预测目标类别。


#### 2.2.6 激活函数及其导数汇总表

| 激活函数 | 函数表达式 | 导数表达式 | 输出范围 | 主要问题 |
|---------|-----------|-----------|---------|---------|
| Sigmoid | \(\sigma(z) = \frac{1}{1+e^{-z}}\) | \(\sigma'(z) = \sigma(z)(1-\sigma(z))\) | \((0,1)\) | 梯度消失、非零均值 |
| Tanh | \(\tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}\) | \(\tanh'(z) = 1-\tanh^2(z)\) | \((-1,1)\) | 梯度消失 |
| ReLU | \(\max(0, z)\) | \(\begin{cases}1, & z>0 \\ 0, & z<0\end{cases}\) | \([0,+\infty)\) | Dead ReLU |
| Leaky ReLU | \(\max(\alpha z, z)\) | \(\begin{cases}1, & z>0 \\ \alpha, & z<0\end{cases}\) | \((-\infty,+\infty)\) | 需调 \(\alpha\) |
| Softmax | \(\frac{e^{z_i}}{\sum_j e^{z_j}}\) | \(s_i(\delta_{ij}-s_j)\) | \((0,1)\)，和为1 | 仅用于输出层 |

{% asset_img Snipaste_2026-10-06_11-44-18.png "Hexo 博客封面示例" %}

#### 2.2.7 代码实现：激活函数及其导数

```python
# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块，简写为nn
import torch.nn as nn
# 导入绘图库
import matplotlib.pyplot as plt

# 生成从-5到5的200个等间距点，作为横坐标
x = torch.linspace(-5, 5, 200)

# ==================== 激活函数值计算 ====================
# 计算Sigmoid激活值
sigmoid = torch.sigmoid(x)
# 计算Tanh激活值
tanh = torch.tanh(x)
# 计算ReLU激活值
relu = torch.relu(x)
# 计算Leaky ReLU激活值，负半轴斜率0.1
leaky_relu = nn.LeakyReLU(0.1)(x)

# ==================== 激活函数导数计算 ====================
# Sigmoid导数：sigmoid(z) * (1 - sigmoid(z))
sigmoid_grad = sigmoid * (1 - sigmoid)
# Tanh导数：1 - tanh(z)^2
tanh_grad = 1 - tanh ** 2
# ReLU导数：z>0为1，否则为0；用float转换布尔张量
relu_grad = (x > 0).float()
# Leaky ReLU导数：z>0为1，否则为0.1
leaky_relu_grad = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.1))

# ==================== 绘制激活函数曲线 ====================
# 创建画布，宽12英寸，高5英寸
plt.figure(figsize=(12, 5))

# 创建左子图：激活函数值
plt.subplot(1, 2, 1)
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
plt.title('Activation Functions')
# 设置横轴标签
plt.xlabel('z')
# 设置纵轴标签
plt.ylabel('f(z)')

# 创建右子图：激活函数导数
plt.subplot(1, 2, 2)
# 绘制Sigmoid导数曲线
plt.plot(x.numpy(), sigmoid_grad.numpy(), label="Sigmoid'")
# 绘制Tanh导数曲线
plt.plot(x.numpy(), tanh_grad.numpy(), label="Tanh'")
# 绘制ReLU导数曲线
plt.plot(x.numpy(), relu_grad.numpy(), label="ReLU'")
# 绘制Leaky ReLU导数曲线
plt.plot(x.numpy(), leaky_relu_grad.numpy(), label="Leaky ReLU'")
# 显示图例
plt.legend()
# 显示网格
plt.grid(True)
# 设置标题
plt.title('Derivatives of Activation Functions')
# 设置横轴标签
plt.xlabel('z')
# 设置纵轴标签
plt.ylabel("f'(z)")

# 自动调整子图间距
plt.tight_layout()
# 显示图像
plt.show()

# ==================== Softmax 及其雅可比矩阵演示 ====================
# 构造一个3维logit向量
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# 计算Softmax输出
s = torch.softmax(z, dim=0)
# 打印Softmax输出及其和
print(f"Softmax输出: {s.detach().numpy()}, 和为: {s.sum().item():.4f}")

# 逐元素计算雅可比矩阵 J[i][j] = ∂s_i / ∂z_j
# 初始化3x3雅可比矩阵
J = torch.zeros(3, 3)
# 遍历每个输出维度i
for i in range(3):
    # 对s_i关于z求梯度，retain_graph=True保留计算图供下次使用
    grad_i = torch.autograd.grad(s[i], z, retain_graph=True)[0]
    # 将梯度填入雅可比矩阵第i行
    J[i] = grad_i
# 打印雅可比矩阵
print(f"Softmax雅可比矩阵:\n{J.numpy()}")

# 用公式 s_i(δ_ij - s_j) 手动验证
# 构造单位矩阵
I = torch.eye(3)
# 外积 s_i * s_j
outer = s.detach().unsqueeze(1) * s.detach().unsqueeze(0)
# 对角项为 s_i(1-s_i)，非对角项为 -s_i s_j
J_manual = torch.diag(s.detach()) - outer
# 打印手动计算结果
print(f"手动公式计算:\n{J_manual.numpy()}")
# 验证两者是否一致
print(f"最大误差: {(J - J_manual).abs().max().item():.2e}")
```

**代码输出解读**：
- 左图展示四个激活函数的形状：Sigmoid 和 Tanh 呈 S 形，ReLU 和 Leaky ReLU 在正半轴为直线。
- 右图展示导数：Sigmoid 导数最大仅 0.25，Tanh 导数最大为 1，ReLU 正半轴导数为 1、负半轴为 0，Leaky ReLU 负半轴为 0.1。
- Softmax 雅可比矩阵的对角元素为正（\(s_i(1-s_i)\)），非对角元素为负（\(-s_i s_j\)），每行之和为 0，与理论推导完全一致。


#### 2.2.8 激活函数选择建议

对于隐藏层:
优先选择ReLU激活函数
如果ReLu效果不好，那么尝试其他激活，如Leaky ReLu等。
如果你使用了ReLU， 需要注意一下Dead ReLU问题， 避免出现0梯度从而导致过多的神经元死亡。
少用使用sigmoid激活函数，可以尝试使用tanh激活函数

对于输出层:
二分类问题选择sigmoid激活函数
多分类问题选择softmax激活函数
回归问题选择identity激活函数


| 场景 | 推荐激活函数 | 理由 |
|------|-------------|------|
| 隐藏层（默认） | ReLU | 计算简单、缓解梯度消失 |
| 隐藏层（ReLU失效） | Leaky ReLU / ELU | 避免 Dead ReLU |
| 二分类输出层 | Sigmoid | 输出可解释为概率 |
| 多分类输出层 | Softmax | 输出为概率分布，与交叉熵配合 |
| 回归输出层 | 恒等映射（无激活） | 输出范围不受限 |
| RNN 隐藏层 | Tanh | 输出以零为中心，适合序列建模 |

**核心结论**：
1. 激活函数引入非线性，是深层网络表达能力的来源。
2. 导数的性质决定了梯度传播的质量：导数接近 0 会导致梯度消失，导数为 0 会导致神经元死亡。
3. Softmax 是多分类输出层的标准选择，其与交叉熵的联合梯度 \(s_i - y_i\) 形式简洁，是深度学习中最优雅的数学结果之一。

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

前向传播就是一系列矩阵乘法与非线性激活的交替组合。


### 2.5 参数初始化

#### 2.5.1 为什么参数初始化如此重要

神经网络的训练本质上是**非凸优化**过程：损失函数关于参数的曲面存在大量鞍点和局部极小值。参数初始化决定了优化的**起点**，起点不同，最终收敛到的解可能完全不同。

参数初始化不当会导致以下几类问题：

1. **对称性问题**：如果所有参数初始化为相同值，同一层的所有神经元在前向传播中产生完全相同的输出，反向传播中收到完全相同的梯度，更新后参数仍然完全相同。这意味着无论网络多宽，其表达能力等价于只有一个神经元。这种现象称为**对称性破坏失败**。

**解释**：
在神经网络中，**对称性**指的是同一层内不同神经元在参数和功能上完全等价、无法区分的状态。这种状态通常由不恰当的参数初始化引起，最典型的是全零初始化。

具体来说，假设第 \(l\) 层有两个神经元 \(j\) 和 \(k\)，如果它们的权重和偏置完全相同：

\[
W_j = W_k, \quad b_j = b_k
\]

那么对于任意输入 \(\boldsymbol{x}\)，它们计算出的加权求和值必然相同：

\[
z_j = W_j^\top \boldsymbol{x} + b_j = W_k^\top \boldsymbol{x} + b_k = z_k
\]

经过同一个激活函数后，输出也完全相同：

\[
a_j = f(z_j) = f(z_k) = a_k
\]

进入反向传播后，由于它们对损失的贡献完全一致，计算出的梯度也必然相同：

\[
\frac{\partial L}{\partial W_j} = \frac{\partial L}{\partial W_k}
\]

因此，经过一次参数更新后，二者仍然保持相等。如此反复，无论训练多少轮，这两个神经元始终完全一样。这意味着，虽然网络在结构上拥有多个神经元，但在功能上它们只相当于一个神经元。网络的容量被浪费，表达能力被严重削弱。

打个比方：一个团队如果所有成员在知识、经验、判断上完全一致，面对同一任务时每个人都会给出相同的方案，无法形成分工与互补。团队的人数虽多，整体能力却等同于一个人。神经网络中的神经元也是如此，它们需要各自不同，才能从数据中学习到多样化的特征。

在下文参数初始化的方案中全0，全1，固定值无法打破对称性
kaiming初始化，xavier初始化，随机初始化可以打破对称性

**打破对称性**，就是让同一层的不同神经元在初始化时拥有不同的参数。随机初始化（如 Xavier、He 初始化）使每个神经元从略微不同的起点出发，前向传播产生不同输出，反向传播得到不同梯度，从而在训练中逐渐分化，各自承担不同的角色。这正是参数初始化必须引入随机性的根本原因。

2. **梯度消失**：如果权重初始值过小，每层的激活值和梯度会逐层衰减。以Sigmoid为例，若每层输出方差缩小为原来的 \(k\) 倍（\(k<1\)），经过 \(L\) 层后方差变为 \(k^L\)，当 \(L\) 较大时梯度趋于零，浅层参数几乎无法更新。

3. **梯度爆炸**：如果权重初始值过大，每层的激活值和梯度会逐层放大。同样经过 \(L\) 层，方差变为 \(k^L\)（\(k>1\)），当 \(L\) 较大时梯度趋于无穷，参数更新步长巨大，损失震荡甚至出现 `NaN`。

4. 收敛缓慢

因此，良好的初始化策略应满足两个目标：
- **破坏对称性**：使不同神经元的初始参数不同。
- **保持信号方差稳定**：使前向传播的激活值和反向传播的梯度在层间保持大致相同的尺度。

---

#### 2.5.2 全零初始化及其失败原因

最简单的想法是将所有权重和偏置初始化为零：

\[
W = 0, \quad b = 0
\]

**前向传播分析**：

对于第 \(l\) 层的第 \(j\) 个神经元：

\[
z_j^{(l)} = \sum_i W_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)} = 0
\]

所有神经元的加权求和值均为 0，经过激活函数后输出也相同：

\[
a_j^{(l)} = f(0) = \text{常数}
\]

**反向传播分析**：

对于同一层的任意两个神经元 \(j\) 和 \(k\)，其对损失的梯度为：

\[
\frac{\partial L}{\partial W_{ji}^{(l)}} = \delta_j^{(l)} \cdot a_i^{(l-1)}
\]

由于 \(a_i^{(l-1)}\) 相同，且 \(\delta_j^{(l)}\) 也因对称性相同，故：

\[
\frac{\partial L}{\partial W_{ji}^{(l)}} = \frac{\partial L}{\partial W_{ki}^{(l)}}
\]

所有神经元的梯度完全相同，更新后仍然相同。网络永远无法打破对称性，等价于只有一个神经元在工作。

**结论**：全零初始化**不可用**。

---

#### 2.5.3 小随机初始化

为破坏对称性，最直接的方法是用小的随机数初始化权重：

\[
W_{ji} \sim \mathcal{N}(0, \sigma^2)
\]

其中 \(\sigma\) 是一个较小的标准差（如 0.01）。

**公式解释**：
- \(\mathcal{N}(0, \sigma^2)\)：均值为 0、方差为 \(\sigma^2\) 的正态分布。
- 每个权重独立采样，不同神经元得到不同初始值，破坏对称性。
- 均值为 0 保证正负权重均衡，避免初始输出系统性偏移。
- \(\sigma\) 较小（如 0.01）是为了避免初始输出过大导致饱和。

**代码实现（李沐《动手学深度学习》风格）**：

```python
# 定义一个函数，按正态分布初始化权重
def normal_init(m, mean, std):
    # 判断m是否为线性层
    if isinstance(m, nn.Linear):
        # 权重按N(mean, std^2)初始化
        m.weight.data.normal_(mean, std)
        # 偏置置零
        m.bias.data.zero_()

# 定义网络
net = nn.Sequential(
    nn.Linear(784, 256),  # 输入层到隐藏层
    nn.ReLU(),            # 激活函数
    nn.Linear(256, 10)    # 隐藏层到输出层
)

# 对每个子模块应用初始化
net.apply(lambda m: normal_init(m, 0, 0.01))
```

**小随机初始化的局限**：

对于浅层网络（2～3层），小随机初始化通常可以工作。但对于深层网络，问题显现：

- **前向传播**：假设每层输入 \(x\) 的方差为 \(\sigma_x^2\)，权重方差为 \(\sigma_w^2\)，则输出 \(z = \sum_i w_i x_i\) 的方差为：

\[
\text{Var}(z) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2
\]

**公式解释**：
- \(n_{in}\)：该层的输入维度（即上一层神经元数）。
- \(\sigma_w^2\)：权重的方差。
- \(\sigma_x^2\)：输入的方差。
- 由于各 \(w_i x_i\) 独立，方差可直接相加，共 \(n_{in}\) 项。

若 \(n_{in} \cdot \sigma_w^2 \ne 1\)，则每层输出方差都会被放大或缩小 \(n_{in} \cdot \sigma_w^2\) 倍。当 \(\sigma_w = 0.01\)、\(n_{in} = 256\) 时，\(n_{in} \cdot \sigma_w^2 = 0.0256 \ll 1\)，每层方差缩小约 40 倍，几层之后激活值趋于零，梯度也随之消失。

---

#### 2.5.4 参数初始化方案
均匀分布初始化
权重参数初始化从区间均匀随机取值，默认区间为（0，1）。可以设置为在(-1/√d,1/√d)均匀分布中生成当前神经元的权重，其中d为神经元的输入数量

正态分布初始化
随机初始化从均值为0，标准差是1的高斯分布中取样，使用一些很小的值对参数W进行初始化

全0初始化
将神经网络中的所有权重参数初始化为 0

全1初始化
将神经网络中的所有权重参数初始化为 1

固定值初始化
将神经网络中的所有权重参数初始化为某个固定值

{% asset_img Snipaste_2026-10-06_15-20-23.png "Hexo 博客封面示例" %}

kaiming 初始化，也叫做 HE 初始化
HE 初始化分为正态分布的 HE 初始化、均匀分布的 HE 初始化.
正态分布的he初始化
它是从 [0, std] 中抽取样本的，std = sqrt(2 / fan_in)
均匀分布的he初始化
它从 [-limit，limit] 中的均匀分布中抽取样本, limit是 sqrt(6 / fan_in)
fan_in 输入层神经元的个数（特征的个数）

xavier 初始化，也叫做 Glorot初始化
该方法也有两种，一种是正态分布的 xavier 初始化、一种是均匀分布的 xavier 初始化.
正态化的Xavier初始化
它是从 [0, std] 中抽取样本的，std = sqrt(2 / (fan_in + fan_out))
均匀分布的Xavier初始化
[-limit，limit] 中的均匀分布中抽取样本, limit 是 sqrt(6 / (fan_in + fan_out))
fan_in 是输入层神经元的个数， fan_out 是输出层神经元个数

{% asset_img Snipaste_2026-10-06_15-26-45.png "Hexo 博客封面示例" %}

- **随机初始化**

  - 均匀分布初始化：权重参数初始化从区间均匀随机取值，默认区间为（0，1）。可以设置为在(-$$1\over\sqrt{d}$$,$$1\over\sqrt{d}$$)均匀分布中生成当前神经元的权重，其中d为神经元的输入数量。

  - 正态分布初始化：随机初始化从均值为0，标准差是1的高斯分布中取样，使用一些很小的值对参数W进行初始化

  - **优点**：能有效打破对称性

  - **缺点**：随机选择范围不当可能导致梯度问题

  - **适用场景**：浅层网络或低复杂度模型。隐藏层1-3层，总层数不超过5层。

- **全0初始化**：将神经网络中的所有权重参数初始化为0
  - **优点**：实现简单
  - **缺点**：无法打破对称性，所有神经元更新方向相同，无法有效训练
  - **适用场景**：几乎不使用，仅用于偏置项的初始化

- **全1初始化**：将神经网络中的所有权重参数初始化为1
  - **优点**：实现简单
  - **缺点**
    - 无法打破对称性，所有神经元更新方向相同，无法有效训练
    - 会导致激活值在网络中呈指数增长，容易出现梯度爆炸
  - **适用场景**
    - 测试或调试：比如验证神经网络是否能正常前向传播和反向传播
    - 特殊模型结构：某些稀疏网络或特定的自定义网络中可能需要手动设置部分参数为1
    - 偏置初始化：偶尔可以将偏置初始化为小的正值（如 0.1），但很少用1作为偏置的初始值

- **固定值初始化**：将神经网络中的所有权重参数初始化为某个固定值
  - **优点**：实现简单
  - **缺点**
    - 无法打破对称性，所有神经元更新方向相同，无法有效训练
    - 初始权重过大或过小可能导致梯度爆炸或梯度消失
  - **适用场景**
    - 测试或调试

- **kaiming初始化**，也叫做**HE初始化**：专为ReLU和其变体设计，考虑到ReLU激活函数的特性，对输入维度进行缩放
  - HE初始化分为正态分布的HE初始化、均匀分布的HE初始化
    - 正态分布的he初始化
      - w权重值从均值为0, 标准差为std中随机采样，std = `sqrt(2 / fan_in)`
      - std值越大，w权重值离均值0分布相对较广，计算得到的内部状态值有较大的正值或负值
    - 均匀分布的he初始化
      - 它从[-limit，limit] 中的均匀分布中抽取样本, `limit` 是 `sqrt(6 / fan_in)`
    - `fan_in` 输入神经元的个数，**当前层**接受的**来自上一层**的神经元的数量。简单来说，就是当前层接收多少个输入
  - **优点**：适合 ReLU，能保持梯度稳定
  - **缺点**：对非 ReLU 激活函数效果一般
  - **适用场景**：深度网络(10层及以上)，使用 ReLU、Leaky ReLU 激活函数

- **xavier初始化**，也叫做**Glorot初始化**：根据网络输入和输出的维度自动选择权重范围，使输入和输出的方差相同
  - xavier初始化分为正态分布的xavier初始化、均匀分布的xavier初始化
    - 正态化的Xavier初始化
      - w权重值从均值为0, 标准差为std中随机采样，std = `sqrt(2 / (fan_in + fan_out))`
      - std值越小，w权重值离均值0分布相对集中，计算得到的内部状态值有较小的正值或负值
    - 均匀分布的Xavier初始化
      - [-limit，limit] 中的均匀分布中抽取样本, limit 是 `sqrt(6 / (fan_in + fan_out))`
    - fan_in 是输入神经元个数，**当前层**接受的**来自上一层**的神经元的数量。简单来说，就是当前层接收多少个输入

    - fan_out 是输出神经元个数，**当前层**输出的神经元的数量，也就是当前层会传递给**下一层**的神经元的数量。简单来说，就是当前层会产生多少个输出。

  - **优点**：适用于Sigmoid、Tanh 等激活函数，解决梯度消失问题
  - **缺点**：对 ReLU 等激活函数表现欠佳
  - **适用场景**：深度网络(10层及以上)，使用 Sigmoid 或 Tanh 激活函数

```python
import torch.nn as nn


# 1. 均匀分布随机初始化
def test01():

    linear = nn.Linear(5, 3) #创建一个线性层，输入维度5，输出维度3
    # 从0-1均匀分布产生参数

    nn.init.uniform_(linear.weight) #随机初始化权重w，从0-1均匀分布产生参数

    nn.init.uniform_(linear.bias) #随机初始化权重b，从0-1均匀分布产生参数

    print(linear.weight.data)
    print(linear.bias.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-41-44.png "Hexo 博客封面示例" %}


```python
# 2. 固定初始化
def test02():

    linear = nn.Linear(5, 3)  #创建一个线性层，输入维度5，输出维度3

    nn.init.constant_(linear.weight, 5) #对权重w进行初始化，设置固定权重为3

    nn.init.constant_(linear.bias, 5) #对权重b进行初始化，设置固定权重为3

    print(linear.weight.data)
    print(linear.bias.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-47-11.png "Hexo 博客封面示例" %}

```python
# 3. 全0初始化
def test03():

    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight) #对权重w进行初始化,全0
    print(linear.weight.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-52-00.png "Hexo 博客封面示例" %}


```python
# 4. 全1初始化
def test04():

    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-53-32.png "Hexo 博客封面示例" %}


```python
# 5. 正态分布随机初始化
def test05():

    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(linear.weight.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-54-30.png "Hexo 博客封面示例" %}


```python
# 6. kaiming 初始化
def test06():

    # kaiming 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)

    # kaiming 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)
```
图解流程结果图：
{% asset_img Snipaste_2026-10-06_16-57-01.png "Hexo 博客封面示例" %}

```python
# 7. xavier 初始化
def test07():

    # xavier 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data)

    # xavier 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)
```

#### 2.5.4.1 Xavier 初始化（Glorot 初始化）

Xavier 初始化由 Glorot 和 Bengio 于 2010 年提出，目标是使**每一层输出的方差等于输入的方差**，从而在前向传播中保持信号尺度稳定。

**核心思路**：

对于线性层 \(z = Wx + b\)，假设：
- 输入 \(x\) 各分量独立，均值为 0，方差为 \(\sigma_x^2\)。
- 权重 \(W\) 各分量独立，均值为 0，方差为 \(\sigma_w^2\)。
- \(W\) 与 \(x\) 相互独立。

则输出 \(z_j = \sum_{i=1}^{n_{in}} W_{ji} x_i\) 的方差为：

\[
\text{Var}(z_j) = \sum_{i=1}^{n_{in}} \text{Var}(W_{ji} x_i) = \sum_{i=1}^{n_{in}} \text{Var}(W_{ji}) \text{Var}(x_i) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2
\]

**公式解释**：
- \(\text{Var}(W_{ji} x_i) = \text{Var}(W_{ji})\text{Var}(x_i)\)：因为 \(W_{ji}\) 与 \(x_i\) 独立且均值为 0，乘积的方差等于方差的乘积。
- 求和共 \(n_{in}\) 项，每项相同，故得到 \(n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2\)。

要使 \(\text{Var}(z_j) = \sigma_x^2\)，需满足：

\[
n_{in} \cdot \sigma_w^2 = 1 \quad \Rightarrow \quad \sigma_w^2 = \frac{1}{n_{in}}
\]

同理，考虑反向传播时，梯度从输出层传回输入层，需要满足：

\[
n_{out} \cdot \sigma_w^2 = 1 \quad \Rightarrow \quad \sigma_w^2 = \frac{1}{n_{out}}
\]

**公式解释**：
- \(n_{out}\)：该层的输出维度（即下一层神经元数）。
- 反向传播时梯度经过 \(W^\top\) 传播，方差分析类似，但涉及的是 \(n_{out}\)。

两个条件很难同时满足（除非 \(n_{in} = n_{out}\)）。Xavier 采用**调和平均**折中：

\[
\sigma_w^2 = \frac{2}{n_{in} + n_{out}}
\]

**Xavier 正态分布形式**：

\[
W_{ji} \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
\]

**Xavier 均匀分布形式**：

\[
W_{ji} \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
\]

**公式解释**：
- \(U(-a, a)\)：区间 \([-a, a]\) 上的均匀分布，其方差为 \(\frac{a^2}{3}\)。
- 令 \(\frac{a^2}{3} = \frac{2}{n_{in}+n_{out}}\)，解得 \(a = \sqrt{\frac{6}{n_{in}+n_{out}}}\)。
- 均匀分布和正态分布形式在效果上差别不大，实践中均常用。

**适用性**：Xavier 初始化假设激活函数在零点附近近似线性（如 Tanh、Sigmoid），因此**最适合 Tanh 和 Sigmoid**。对于 ReLU，其负半轴输出为 0，破坏了零均值假设，Xavier 会导致方差偏小。

```python
# Xavier 正态分布初始化
def xavier_normal_init(m):
    # 只处理线性层
    if isinstance(m, nn.Linear):
        # 获取输入维度和输出维度
        fan_in, fan_out = m.in_features, m.out_features
        # 计算标准差 sqrt(2 / (fan_in + fan_out))
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        # 按正态分布初始化权重
        m.weight.data.normal_(0, std)
        # 偏置置零
        m.bias.data.zero_()

# PyTorch内置实现
net = nn.Sequential(nn.Linear(784, 256), nn.Tanh(), nn.Linear(256, 10))
# 对每个线性层应用Xavier正态初始化
net.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)
```

---

#### 2.5.5 He 初始化（Kaiming 初始化）

He 初始化由何恺明于 2015 年提出，专门针对 **ReLU 及其变体**设计，解决了 Xavier 在 ReLU 网络上方差偏小的问题。

**核心思路**：

ReLU 将负半轴输出置零，因此其输出 \(a = \text{ReLU}(z)\) 的方差不再是 \(\text{Var}(z)\)，而是：

\[
\text{Var}(a) = \frac{1}{2} \text{Var}(z)
\]

**公式解释**：
- 假设 \(z\) 关于 0 对称分布，则 ReLU 将一半的值置为 0，保留一半的正值。
- 保留部分的条件期望为 \(\mathbb{E}[z | z>0]\)，计算可得 \(\text{Var}(a) = \frac{1}{2}\text{Var}(z)\)。
- 这意味着每经过一层 ReLU，方差减半。

若要使 \(\text{Var}(a) = \sigma_x^2\)，需：

\[
\text{Var}(z) = 2\sigma_x^2
\]

代入 \(\text{Var}(z) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2\)：

\[
n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2 = 2 \sigma_x^2 \quad \Rightarrow \quad \sigma_w^2 = \frac{2}{n_{in}}
\]

**He 正态分布形式**：

\[
W_{ji} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
\]

**He 均匀分布形式**：

\[
W_{ji} \sim U\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)
\]

**公式解释**：
- 分子为 2（而非 Xavier 中的 2），是因为 ReLU 使方差减半，需要额外补偿。
- 分母只有 \(n_{in}\)（而非 Xavier 中的 \(n_{in} + n_{out}\)），因为反向传播中 ReLU 的梯度要么为 1 要么为 0，其期望同样为 \(\frac{1}{2}\)，也需补偿 2 倍，最终形式中只保留 \(n_{in}\)。

**Leaky ReLU 的推广**：

对于 Leaky ReLU（负半轴斜率 \(\alpha\)），其输出方差为：

\[
\text{Var}(a) = \frac{1 + \alpha^2}{2} \text{Var}(z)
\]

**公式解释**：
- 正半轴（概率 \(\frac{1}{2}\)）保持原方差。
- 负半轴（概率 \(\frac{1}{2}\)）乘以 \(\alpha\)，方差贡献为 \(\alpha^2 \text{Var}(z)\)。
- 两者平均得到 \(\frac{1+\alpha^2}{2}\text{Var}(z)\)。

对应的 He 初始化方差为：

\[
\sigma_w^2 = \frac{2}{(1 + \alpha^2) \cdot n_{in}}
\]

```python
# He 正态分布初始化
def kaiming_normal_init(m):
    # 只处理线性层
    if isinstance(m, nn.Linear):
        # 获取输入维度
        fan_in = m.in_features
        # 计算标准差 sqrt(2 / fan_in)
        std = (2.0 / fan_in) ** 0.5
        # 按正态分布初始化权重
        m.weight.data.normal_(0, std)
        # 偏置置零
        m.bias.data.zero_()

# PyTorch内置实现（推荐）
net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
# 对每个线性层应用Kaiming正态初始化，mode='fan_in'表示使用输入维度
net.apply(lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
          if isinstance(m, nn.Linear) else None)

# Leaky ReLU 对应的初始化
# a=0.01 为负半轴斜率
nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
```

---

#### 2.5.6 偏置初始化

偏置 \(b\) 的初始化通常比权重简单，常见做法是**置零**：

\[
b = 0
\]

**原因**：
- 偏置不参与方差传播的分析（它是常数，不放大输入信号）。
- 置零是最简单且通常有效的方法。
- 权重随机初始化已经破坏了对称性，偏置无需再随机。

**例外情况**：
- **ReLU 前的偏置**：有时初始化为小的正数（如 0.01），以降低初始阶段 ReLU 死亡的概率。
- **输出层偏置**：对于分类任务，如果类别极度不平衡，可将输出层偏置初始化为 \(\log(\frac{p}{1-p})\)，其中 \(p\) 是正类先验概率。这可以加速初期收敛。
- **LSTM 遗忘门偏置**：通常初始化为 1，使遗忘门初始状态接近"完全保留"，有利于长序列建模。

```python
# 偏置置零（默认行为）
def zero_bias_init(m):
    if isinstance(m, nn.Linear):
        # 偏置全部置零
        m.bias.data.zero_()

# 输出层偏置按先验概率初始化
def prior_bias_init(m, prior):
    if isinstance(m, nn.Linear):
        # prior为各分类别的先验概率
        m.bias.data = torch.log(prior / (1 - prior))
```

---

#### 2.5.7 批量归一化对初始化的影响

批量归一化（Batch Normalization, BN）的引入大幅降低了对初始化策略的敏感性。BN 对每一层的输入进行标准化：

\[
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
\]

**公式解释**：
- \(\mu_\mathcal{B}\)：当前小批量的均值。
- \(\sigma_\mathcal{B}^2\)：当前小批量的方差。
- \(\epsilon\)：极小常数，防止除零。
- 无论输入 \(x_i\) 的尺度如何，BN 都会将其标准化到均值为 0、方差为 1 的分布。

因此，即使权重初始化使得某层输出方差偏离理想值，BN 也会将其"拉回"标准尺度。这使得：
- 可以使用更大的学习率。
- 对初始化策略的选择不再那么敏感。
- 但 BN 本身引入了可学习的缩放参数 \(\gamma\) 和平移参数 \(\beta\)，通常初始化为 \(\gamma = 1, \beta = 0\)。

```python
# 带BN的网络，BN参数默认gamma=1, beta=0
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # gamma初始为1, beta初始为0
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 线性 -> BN -> ReLU
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
```

**即使有 BN，仍建议使用 He 或 Xavier 初始化**，因为 BN 只在小批量维度上标准化，且训练初期的小批量统计量可能不稳定。

---

#### 2.5.8 初始化方法对比实验

以下代码构造一个 10 层全连接网络，分别用不同初始化方法训练，观察激活值方差和梯度范数的变化：

```python
# 导入PyTorch
import torch
import torch.nn as nn
# 导入绘图库
import matplotlib.pyplot as plt

# 设置随机种子，保证可复现
torch.manual_seed(42)

# 构造10层全连接网络
def build_net(init_method):
    layers = []
    dims = [256] * 11  # 11个维度，构造10层
    for i in range(10):
        # 添加线性层
        layers.append(nn.Linear(dims[i], dims[i+1]))
        # 添加Tanh激活（便于观察梯度）
        layers.append(nn.Tanh())
    net = nn.Sequential(*layers)
    # 应用指定的初始化方法
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if init_method == 'small':
                # 小随机初始化：std=0.01
                nn.init.normal_(m.weight, 0, 0.01)
            elif init_method == 'xavier':
                # Xavier正态初始化
                nn.init.xavier_normal_(m.weight)
            elif init_method == 'kaiming':
                # Kaiming正态初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # 偏置置零
            nn.init.zeros_(m.bias)
    return net

# 构造一个输入样本
x = torch.randn(64, 256)

# 记录三种初始化下的激活值方差
methods = ['small', 'xavier', 'kaiming']
colors = ['red', 'green', 'blue']

plt.figure(figsize=(12, 5))

# ========== 左图：激活值方差随层数变化 ==========
plt.subplot(1, 2, 1)
for method, color in zip(methods, colors):
    net = build_net(method)
    # 逐层前向传播，记录每层输出方差
    variances = []
    with torch.no_grad():
        h = x
        for layer in net:
            h = layer(h)
            # 记录当前层输出的方差
            variances.append(h.var().item())
    # 绘制方差曲线
    plt.plot(range(1, len(variances)+1), variances, label=method, color=color, marker='o')

plt.xlabel('Layer')
plt.ylabel('Activation Variance')
plt.title('Forward Pass Variance')
plt.legend()
plt.grid(True)
# 纵轴取对数，便于观察指数级变化
plt.yscale('log')

# ========== 右图：梯度范数随层数变化 ==========
plt.subplot(1, 2, 2)
for method, color in zip(methods, colors):
    net = build_net(method)
    # 前向传播
    out = net(x)
    # 构造一个简单的损失：输出平方和
    loss = (out ** 2).sum()
    # 反向传播
    loss.backward()
    # 记录每层权重的梯度范数（从输出层到输入层逆序）
    grad_norms = []
    for m in net.modules():
        if isinstance(m, nn.Linear) and m.weight.grad is not None:
            # 计算梯度L2范数
            grad_norms.append(m.weight.grad.norm().item())
    # 反转顺序，使横轴从输入层到输出层
    grad_norms = grad_norms[::-1]
    plt.plot(range(1, len(grad_norms)+1), grad_norms, label=method, color=color, marker='o')

plt.xlabel('Layer (from input to output)')
plt.ylabel('Gradient Norm')
plt.title('Backward Pass Gradient Norm')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.show()
```

**实验结果解读**：

- **左图（前向方差）**：
  - `small` 初始化：方差随层数指数衰减（因 \(n_{in} \cdot \sigma_w^2 = 256 \times 0.0001 = 0.0256 \ll 1\)），深层激活值趋近于零。
  - `xavier` 初始化：方差在层间大致保持稳定，因为 \(\sigma_w^2 = \frac{2}{256+256} = \frac{1}{256}\)，恰好使 \(n_{in}\sigma_w^2 = 1\)（对线性层而言）。
  - `kaiming` 初始化：方差同样保持稳定，因 \(\sigma_w^2 = \frac{2}{256}\)，补偿了 ReLU 的方差减半效应。注意本实验用 Tanh，Kaiming 会略微放大方差，实际中应与 ReLU 搭配。

- **右图（反向梯度）**：
  - `small` 初始化：梯度范数从输出层到输入层指数衰减，浅层几乎无法更新。
  - `xavier` 和 `kaiming`：梯度范数在层间保持稳定，确保所有层都能有效学习。

---

#### 2.5.9 初始化方法选择指南

| 激活函数 | 推荐初始化 | 方差公式 | 说明 |
|---------|-----------|---------|------|
| Sigmoid | Xavier | \(\sigma_w^2 = \frac{2}{n_{in}+n_{out}}\) | 输出近似零均值 |
| Tanh | Xavier | \(\sigma_w^2 = \frac{2}{n_{in}+n_{out}}\) | 同上 |
| ReLU | He (Kaiming) | \(\sigma_w^2 = \frac{2}{n_{in}}\) | 补偿方差减半 |
| Leaky ReLU | He (Kaiming) | \(\sigma_w^2 = \frac{2}{(1+\alpha^2)n_{in}}\) | 考虑负半轴斜率 |
| SELU | LeCun | \(\sigma_w^2 = \frac{1}{n_{in}}\) | 自归一化网络 |
| 带BN的网络 | He / Xavier | — | BN 降低敏感性 |

**实践建议**：

1. **默认选择 He 初始化 + ReLU**：这是当前最主流的组合，适用于大多数前馈网络和卷积网络。
2. **使用 PyTorch 内置函数**：`nn.init.kaiming_normal_`、`nn.init.xavier_normal_` 等已实现数值稳定版本，优于手写。
3. **配合 BN 使用**：如果网络包含 BN，初始化策略的选择空间更大，He 或 Xavier 均可。
4. **输出层特殊处理**：输出层偏置可按任务先验设置，如分类任务的不平衡类别。
5. **RNN/LSTM**：LSTM 遗忘门偏置初始化为 1，其余参数通常用 Xavier 或正交初始化。

**核心结论**：参数初始化的本质是**控制信号方差在层间的传播**。Xavier 使方差在线性层间保持不变，He 进一步补偿了 ReLU 的方差减半效应。理解方差传播的分析方法。

激活函数ReLU及其系列，优先用kaiming
激活函数非ReLU，优先用xavier
浅层网络可以用 随机初始化

### 2.6神经网络搭建和参数计算

####  构建神经网络

在pytorch中定义深度神经网络其实就是层堆叠的过程，继承自nn.Module，实现两个方法：

- `__init__`方法中定义网络中的层结构，主要是全连接层，并进行初始化
- forward方法，在调用神经网络模型对象的时候，底层会自动调用该函数。该函数中为初始化定义的layer传入数据，进行前向传播等。

接下来我们来构建如下图所示的神经网络模型：
{% asset_img Snipaste_2026-10-06_17-34-54.png "Hexo 博客封面示例" %}

**编码设计如下：**

- 第1个隐藏层：权重初始化采用标准化的xavier初始化 激活函数使用sigmoid
- 第2个隐藏层：权重初始化采用标准化的He初始化 激活函数采用relu
- out输出层线性层 假若多分类，采用softmax做数据归一化


**构造神经网络模型代码:**

```python
import torch
import torch.nn as nn
from torchsummary import summary  # 计算模型参数,查看模型结构, pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/


#创建神经网络模型类，自定义类继承nn.model
class Model(nn.Module):

    # 初始化属性值
    def __init__(self):

        # 调用父类的初始化属性值，确保nn.Module的初始化代码能够正确执行
        super(Model, self).__init__()

        # 创建第一个隐藏层模型, 3个输入特征,3个输出特征
        self.linear1 = nn.Linear(3, 3)

        # 初始化权重(xavier初始化)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias) #参数b初始化为0

        # 创建第二个隐藏层模型, 3个输入特征(上一层的输出特征),2个输出特征
        self.linear2 = nn.Linear(3, 2)

        # 初始化权重
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias) #参数b初始化为0

        # 创建输出层模型,输入特征数2，输出特征数2
        self.out = nn.Linear(2, 2)

	# 创建前向传播方法, 调用神经网络模型对象时自动执行forward()方法，固定的名称
    def forward(self, x):
        # 数据经过第一个线性层
        x = self.linear1(x)  #加权求和

        # 使用sigmoid激活函数
        x = torch.sigmoid(x)  #激活函数

        #第一层合并版写法
        #x = torch.sigmoid(self.linear1(x))

        # 数据经过第二个线性层
        x = self.linear2(x)

        # 使用relu激活函数
        x = torch.relu(x)

        #第二层合并版写法
        #x = torch.relu(self.linear2(x))

        # 数据经过输出层
        x = self.out(x)
        # 使用softmax激活函数
        # dim=-1:每一维度行数据相加为1，即按行计算，一条一条样本地处理
        x = torch.softmax(x, dim=-1)

        return x
```

**训练神经网络模型代码:**

```python
# 创建构造模型函数(与上面class同级)
def train():
    # 实例化model对象
    my_model = Model()

    # 随机产生数据
    my_data = torch.randn(5, 3)
    print("my_data-->", my_data)
    print("my_data shape", my_data.shape)

    # 数据经过神经网络模型训练，（把随机产生的数据仍进上面创建好的模型）
    output = my_model(my_data)  #实例化模型底层自动调用了forward()方法，进行前向传播
    print("output-->", output)  #（5，3）
    print("output shape-->", output.shape)
    print(f'output.requires_grad:{output.requires_grad}')  #Ture,自动调用了自动微分

    # 计算模型参数
    # 计算每层每个神经元的w和b个数总和
    print("======计算模型参数======")

    #参数1：神经网络模型对象，参数2：输入数据的维度特征，参数3：
    summary(my_model, input_size=(3,), batch_size=5)


    # 查看模型参数
    print("======查看模型参数w和b======")
    for name, parameter in my_model.named_parameters():
        print(f'name:{name}')
        print(f'param:{param}\n')


if __name__ == '__main__':
    train()
```

{% asset_img Snipaste_2026-10-06_20-46-36.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-06_21-08-52.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-06_21-02-55.png "Hexo 博客封面示例" %}

输出结果是第一层12个，第二层8个，输出层6个，和我们预测的一样，随后在进行反向传播更新这些参数，如此不断迭代



####  观察数据形状变化

- 观察程序输入和输出的数据形状变化

  - 输入5行数据，输出也是5行数据
  - 输入5行数据3个特征，经过第一个隐藏层是3个特征，经过第二个隐藏层是2个特征，经过输出层是2个特征
  - 模型最终预测结果是：5行2列数据

  ```python
  mydata.shape---> torch.Size([5, 3])
  output.shape---> torch.Size([5, 2])
  mydata--->
    tensor([[-0.3714, -0.8578, -1.6988],
          [ 0.3149,  0.0142, -1.0432],
          [ 0.5374, -0.1479, -2.0006],
          [ 0.4327, -0.3214,  1.0928],
          [ 2.2156, -1.1640,  1.0289]])
  output--->
   tensor([[0.5095, 0.4905],
          [0.5218, 0.4782],
          [0.5419, 0.4581],
          [0.5163, 0.4837],
          [0.6030, 0.3970]], grad_fn=<SoftmaxBackward>)
  ```
(5,3)-> (5,3)-> (5,2)-> (5,2)

{% asset_img Snipaste_2026-10-06_21-19-04.png "Hexo 博客封面示例" %}

输出结果:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                     [5, 3]              12
            Linear-2                     [5, 2]               8
            Linear-3                     [5, 2]               6
================================================================
Total params: 26
Trainable params: 26
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------

**总结**：
{% asset_img 03-1.png "Hexo 博客封面示例" %}


## 三、损失函数

### 分类任务损失函数

损失函数（Loss Function）是深度学习中衡量模型预测值与真实值之间差异的度量。优化算法的目标就是最小化这个差异。

#### 多分类任务损失函数

在多分类任务通常使用softmax将logits转换为概率的形式，所以多分类的交叉熵损失也叫做softmax损失，它的计算方法是：

$$
\mathcal{L} = - \sum_{i=1}^{n} y_i \log(S(f_\theta(\mathbf{x}_i)))
$$
*(注：公式上方标注 "labels (one-hot)" 指向 $y_i$，下方标注 "Softmax" 指向 $S$)*

其中：
1. y是样本x属于某一个类别的真实概率 （0-1）
2. 而f(x)是样本属于某一类别的预测分数
3. S是softmax激活函数,将属于某一类别的预测分数转换成概率
4. L用来衡量真实值y和预测值f(x)之间差异性的损失结果

{% asset_img Snipaste_2026-10-07_18-04-49.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-07_18-22-02.png "Hexo 博客封面示例" %}

**注意**：
看公式，如果你使用多分类任务损失函数，在公式中已经自动进行了softmax，故而在模型定义时可以省略（模型训练前）
{% asset_img Snipaste_2026-10-07_18-27-42.png "Hexo 博客封面示例" %}

从概率角度理解，我们的目的是最小化正确类别所对应的预测概率的对数的负值(损失值最小)

分类:
    分类问题:
        多分类交叉熵损失: CrossEntropyLoss
        二分类交叉熵损失: BCELoss
    回归问题:
        MAE: Mean Absolute Error, 平均绝对误差.
        MSE: Mean Squared Error, 均方误差.
        Smooth L1: 结合上述两个的特点做的升级, 优化.
    
    多分类交叉熵损失: CrossEntropyLoss
    设计思路:
        Loss = - Σylog(S(f(x)))
    简单记忆:
        x:          样本
        f(x):       加权求和
        S(f(x)):    处理后的概率
        y:          样本x属于某一个类别的 真实概率.
    损失函数结果 = 最小化 正确类别所对应的 预测概率的对数的 负值(损失值最小)
细节:
    CrossEntropyLoss = Softmax() + 损失计算, 后续如果用这个损失函数, 则: 输出层就不用额外调用 softmax()激活函数了.

在PyTorch中使用`nn.CrossEntropyLoss()`实现，如下所示：

```python
import torch
from torch import nn


# 分类损失函数：交叉熵损失使用nn.CrossEntropyLoss()实现。nn.CrossEntropyLoss()=softmax+损失计算
def test01():
	# 设置真实值: 可以是热编码后的结果也可以不进行热编码
	# y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float32)
	# 注意：类型必须是64位整型数据

    # 1. 手动创建样本的真实值 -> 就是上述公式中的 y
	y_true = torch.tensor([1, 2], dtype=torch.int64)
	y_pred = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], requires_grad=True, dtype=torch.float32)


	# 实例化交叉熵损失，默认求平均损失(创建多分类交叉熵损失函数.)
    # 手动创建样本的预测值 -> 就是上述公式中的 f(x)
	# reduction='sum'：总损失
	loss = nn.CrossEntropyLoss() # 平均损失, 来源于参数: reduction: str = "mean",

	# 计算损失结果
	my_loss = loss(y_pred, y_true).detach().numpy()
	print(f'loss:{my_loss}')
```


### 二分类任务损失函数（BCE）

在处理二分类任务时，我们不再使用softmax激活函数，而是使用sigmoid激活函数，那损失函数也相应的进行调整，使用二分类的交叉熵损失函数：

$$
L = -y \log \hat{y} - (1 - y) \log(1 - \hat{y})
$$

其中：
- y是样本x属于某一个类别的真实概率（0或1）
- 而$$\hat{y}$$是样本属于某一类别的预测概率
- L用来衡量真实值y与预测值$$\hat{y}$$之间差异性的损失结果。

这边建议理解一下公式，上面的公式明显分为两类，y为0或1，会造成 - 左右两式其中一个必为0，正好呼应二分类 

在pytorch中实现时使用`nn.BCELoss()`，如下所示：

*(此处为损失曲线图表：展示了 y=0 和 y=1 时的损失变化曲线，横轴为 y^，纵轴为 Loss)*

{% asset_img Snipaste_2026-10-07_18-27-42.png "Hexo 博客封面示例" %}

课堂代码：
```python
"""
案例:
    演示二分类任务的损失函数.

二分类任务的损失函数(BCELoss):
    公式:
        Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
    细节:
        因为公式中没有包含Sigmoid激活函数, 所以使用BCELoss的时候, 还需要手动指定 Sigmoid.(区别于sotfmax)
"""

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 演示: 二分类任务的损失函数.
def dm01():
    # 1. 设置真实值.
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)

    # 2. 设置预测值(概率) （自定义，和不唯1也可以）
    y_pred = torch.tensor([0.6901, 0.5423, 0.2639])

    # 3. 创建二分类交叉熵损失函数.
    criterion = nn.BCELoss()    # reduction: str = "mean" -> 均值

    # 4. 计算损失值.
    loss = criterion(y_pred, y_true)
    print(f'损失值: {loss}')

# 2. 测试
if __name__ == '__main__':
    dm01()
```


笔记代码：
```python
import torch
from torch import nn


def test02():
    # 1 设置真实值和预测值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)

    # 预测值是sigmoid输出的结果
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469], requires_grad=True)

    # 2 实例化二分类交叉熵损失
    loss = nn.BCELoss()

    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss：', my_loss)
```
### 回归任务损失函数

####  MAE损失函数

**mean absolute loss(MAE)**也被称为L1 Loss，是以绝对误差作为距离
损失函数公式：

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} |y_i - f_\theta(x_i)|
$$

{% asset_img Snipaste_2026-10-07_22-16-32.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-07_22-17-02.png "Hexo 博客封面示例" %}

特点是：

- 由于L1 loss具有稀疏性，为了惩罚较大的值，因此常常将其作为正则项添加到其他loss中作为约束。(0点不可导, 产生稀疏矩阵)
- L1 loss的最大问题是梯度在零点不平滑，导致会跳过极小值
- 适用于回归问题中存在异常值或噪声数据时，可以减少对离群点的敏感性

{% asset_img 03-9.png "Hexo 博客封面示例" %}

在PyTorch中使用`nn.L1Loss()`实现，如下所示：
笔记代码：

```python
import torch
from torch import nn


# 计算inputs与target之差的绝对值
def test03():
    # 1 设置真实值和预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    # 2 实例MAE损失对象
    loss = nn.L1Loss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss:', my_loss)
```

课堂代码：

```python
"""
案例:
    演示 回归任务的损失函数介绍.


回归任务常用损失函数如下:
    MAE:   Mean Absolute Error, 平均绝对误差.
        公式:
            误差绝对值之和 / 样本总数
        类似于L1正则化, 权重可以降维0, 数据会变得稀疏.

        弊端:
            在0点不平滑, 可能错过最小值.

    MSE:   Mean Squared Error, 均方误差.
        公式:
            误差平方之和 / 样本总数
        弊端:
            如果差值过大, 可能存在梯度爆炸的情况.

    Smooth L1:
        就是基于MAE 和 MSE做的综合, 在 [-1, 1]是 L2(MSE), 其它段时L1.
        这样即解决了L1不平滑的问题(0点不可导, 可能错过最小值)
        又解决了L2(MSE)的 梯度爆炸的问题.
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数, 演示: MAE 损失函数.
def dm01():
    # 1. 定义变量, 记录: 三个样本的真实值.（为了演示损失函数，人为假设出来的标准答案。真实项目中，它们会从数据集中读取，而不是凭空写出来。）
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量, 记录: 预测值.（真实训练里，y_pred 是模型根据输入 x_batch 算出来的，不是手写的。）
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建MAE损失函数对象.
    criterion = nn.L1Loss()

    # 4. 计算损失.
    loss = criterion(y_pred, y_true)

    # 5. 输出损失.
    print(f'MAE: {loss}')

# 4. 测试
if __name__ == '__main__':
    # dm01()    # 0.699999988079071
    # dm02()    # 0.6700000166893005
    dm03()      # 0.33500000834465027
```

{% asset_img Snipaste_2026-10-07_22-52-51.png "Hexo 博客封面示例" %}

2. 真实训练里，y_true在自定义的 Dataset 里，通常会返回 (x, y_true)：
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.0], [2.0], [3.0]])
        self.y = torch.tensor([2.0, 2.0, 2.0])  # 这就是真实值

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)
```
然后 DataLoader 会把它变成一批一批的数据：

```python
dataloader = DataLoader(MyDataset(), batch_size=3)

for x_batch, y_true_batch in dataloader:
    print(x_batch)
    print(y_true_batch)
这里的 y_true_batch 就相当于你 dm01() 里的：

python
y_true = torch.tensor([2.0, 2.0, 2.0])
```

#### MSE损失函数
**Mean Squared Loss/ Quadratic Loss(MSE loss)**也被称为L2 loss，或欧氏距离，它以误差的平方和的均值作为距离

**函数定义**：

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]



特点是：

- L2 loss也常常作为正则项，对于离群点（outliers）敏感，因为平方项会放大大误差
- 当预测值与目标值相差很大时, 梯度容易爆炸
  - 梯度爆炸:网络层之间的梯度（值大于1.0）重复相乘导致的指数级增长会产生梯度爆炸

- 适用于大多数标准回归问题，如房价预测、温度预测等

在PyTorch中通过`nn.MSELoss()`实现：
课堂代码：
```python
import torch
from torch import nn


def test04():
    # 1 设置真实值和预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    # 2 实例MSE损失对象
    loss = nn.MSELoss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('myloss:', my_loss)
```

笔记代码：
```python
# 2. 定义函数, 演示: MSE 损失函数.
def dm02():
    # 1. 定义变量, 记录: 真实值.
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量, 记录: 预测值.
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建MSE损失函数对象.
    criterion = nn.MSELoss()

    # 4. 计算损失.
    loss = criterion(y_pred, y_true)

    # 5. 输出损失.
    print(f'MSE: {loss}')
```

### Smooth L1损失函数
> smooth L1说的是光滑之后的L1，是一种结合了均方误差（MSE）和平均绝对误差（MAE）优点的损失函数。它在误差较小时表现得像 MSE，在误差较大时则更像 MAE。

**函数定义**：

\[
\text{Quantile}(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}), & y \ge \hat{y} \\ (1 - \tau)(\hat{y} - y), & y < \hat{y} \end{cases}
\]

{% asset_img 03-13.png "Hexo 博客封面示例" %}


该函数实际上就是一个**分段函数**

- 在[-1,1]之间实际上就是L2损失，这样解决了L1的不光滑问题
- 在[-1,1]区间外，实际上就是L1损失，这样就解决了离群点梯度爆炸的问题

特点是：

- **对离群点更加鲁棒**：当误差较大时，损失函数会线性增加（而不是像MSE那样平方增加），因此它对离群点的惩罚更小，避免了MSE对离群点过度敏感的问题

- **计算梯度时更加平滑**：与MAE相比，Smooth L1在小误差时表现得像MSE，避免了在训练过程中因使用绝对误差而导致的梯度不连续问题

笔记代码：
在PyTorch中使用`nn.SmoothL1Loss()`计算该损失，如下所示：

```python
import torch
from torch import nn


def test05():
    # 1 设置真实值和预测值
    y_true = torch.tensor([0, 3])
    y_pred = torch.tensor([0.6, 0.4], requires_grad=True)
    # 2 实例smmothL1损失对象
    loss = nn.SmoothL1Loss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss:', my_loss)
```

课堂代码：

```python
"""
    Smooth L1:
        就是基于MAE 和 MSE做的综合, 在 [-1, 1]是 L2(MSE), 其它段时L1.
        这样即解决了L1不平滑的问题(0点不可导, 可能错过最小值)
        又解决了L2(MSE)的 梯度爆炸的问题.
"""
# 3. 定义函数, 演示: Smooth L1 损失函数.

def dm03():
    # 1. 定义变量, 记录: 真实值.
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量, 记录: 预测值.
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建Smooth L1损失函数对象.
    criterion = nn.SmoothL1Loss()

    # 4. 计算损失.
    loss = criterion(y_pred, y_true)

    # 5. 输出损失.
    print(f'Smooth L1: {loss}')
```

### 详解

损失函数（Loss Function）是深度学习中衡量模型预测值与真实值之间差异的度量。优化算法的目标就是最小化这个差异。不同的任务类型需要不同的损失函数：**回归任务**关心数值的接近程度，**二分类任务**关心二值判断的准确性，**多分类任务**关心类别概率分布的匹配程度。

本文按任务类型分别讲述各类损失函数，每个函数均包含数学定义、公式解释、几何/概率意义、代码实现和实际案例。


#### 三·一 回归任务的损失函数

回归任务的目标是预测连续数值，如房价预测、温度预测、股票价格预测等。回归损失衡量的是预测值与真实值之间的数值距离。

##### 3.1.1 均方误差（MSE / L2 Loss）

**函数定义**：

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**公式解释**：
- \(n\)：样本总数。
- \(y_i\)：第 \(i\) 个样本的真实值（标签），是一个实数。
- \(\hat{y}_i\)：第 \(i\) 个样本的预测值，由模型输出。
- \(y_i - \hat{y}_i\)：预测误差（残差）。正值表示预测偏小，负值表示预测偏大。
- \((y_i - \hat{y}_i)^2\)：误差的平方。平方有两个作用：一是消除正负号，使误差同向累加，避免正负抵消；二是放大较大误差的影响，使模型更关注难以拟合的样本。
- \(\frac{1}{n} \sum_{i=1}^{n}\)：对所有样本的平方误差求平均，使损失值与样本数量无关，便于跨数据集比较。
- MSE 的值越小，说明预测越准确；当所有预测完全正确时，MSE = 0。

**概率意义**：

假设真实值 \(y_i = f(x_i) + \epsilon_i\)，其中噪声 \(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\) 服从均值为 0、方差为 \(\sigma^2\) 的正态分布，则预测值 \(\hat{y}_i\) 的似然函数为：

\[
p(y_i | x_i) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)
\]

对所有样本取负对数似然：

\[
-\log \prod_i p(y_i | x_i) = \sum_i \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} + \text{常数}
\]

忽略常数项和系数，最大化似然等价于最小化 MSE。因此，**MSE 等价于在高斯噪声假设下的最大似然估计**。

**导数公式**：

\[
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
\]

**公式解释**：
- \(\hat{y}_i - y_i\)：预测值与真实值之差。若预测偏大（\(\hat{y}_i > y_i\)），梯度为正，参数更新使预测减小；若预测偏小，梯度为负，参数更新使预测增大。
- \(\frac{2}{n}\)：缩放因子。实际训练中该系数会被学习率吸收，不影响优化方向。

**优缺点**：
- 优点：处处可导，梯度平滑，优化稳定；对高斯噪声数据是最优的。
- 缺点：对异常值敏感。若某个样本的误差为 100，则其平方贡献为 10000，会主导整个损失，使模型偏向拟合异常值。

**代码实现**：

```python
# 导入数值计算库
import numpy as np
# 导入PyTorch
import torch
import torch.nn as nn

# ==================== 从零实现 MSE ====================
def mse_loss(y_true, y_pred):
    """均方误差 - 从零实现"""
    # y_true: 真实值，形状 (n,)
    # y_pred: 预测值，形状 (n,)
    # 计算逐元素差值
    diff = y_true - y_pred
    # 平方后求均值
    return np.mean(diff ** 2)

# 构造示例数据
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# 计算MSE
loss = mse_loss(y_true, y_pred)
print(f"MSE: {loss:.4f}")  # 输出约 0.3750

# ==================== PyTorch实现 ====================
# 内置MSELoss
criterion = nn.MSELoss()
# 转换为张量
y_true_t = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred_t = torch.tensor([2.5, 0.0, 2.0, 8.0], requires_grad=True)
# 计算MSE
loss_t = criterion(y_pred_t, y_true_t)
# 反向传播
loss_t.backward()
# 查看梯度：dMSE/dy_pred = 2/n * (y_pred - y_true)
print(f"梯度: {y_pred_t.grad.numpy()}")
```

**案例：房价预测**。假设用线性回归预测房价，输入特征为面积、房间数等，输出为价格。MSE 是此类任务的标准损失函数。若数据中存在个别极端高价的豪宅，MSE 会被这些样本主导，此时可考虑使用 Huber Loss 或对数据进行对数变换。


##### 3.1.2 平均绝对误差（MAE / L1 Loss）

**函数定义**：

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

**公式解释**：
- \(|y_i - \hat{y}_i|\)：误差的绝对值，衡量预测值与真实值的距离，不区分正负。
- 与 MSE 不同，MAE 不对误差平方，因此**对异常值的敏感度更低**。
- 若误差为 100，MAE 贡献为 100，而 MSE 贡献为 10000。

**概率意义**：

MAE 等价于**拉普拉斯噪声假设下的最大似然估计**。拉普拉斯分布的尾部比高斯分布更厚，对异常值更鲁棒。

**导数公式**：

\[
\frac{\partial \text{MAE}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(\hat{y}_i - y_i)
\]

**公式解释**：
- \(\text{sign}(\cdot)\)：符号函数。当 \(\hat{y}_i > y_i\) 时返回 +1，当 \(\hat{y}_i < y_i\) 时返回 -1，当相等时返回 0。
- 梯度大小恒定为 \(\frac{1}{n}\)，与误差大小无关。这意味着 MAE 对每个样本的"关注度"相同，不会因某个样本误差大而特别偏向它。
- 缺点：在 \(\hat{y}_i = y_i\) 处不可导（符号函数不连续），梯度不平滑。

**优缺点**：
- 优点：对异常值鲁棒；损失具有明确的物理意义（平均绝对偏差）。
- 缺点：在零点不可导，优化时可能震荡；梯度恒定，收敛速度可能较慢。

**代码实现**：

```python
# ==================== 从零实现 MAE ====================
def mae_loss(y_true, y_pred):
    """平均绝对误差 - 从零实现"""
    # 计算逐元素差值的绝对值
    diff = np.abs(y_true - y_pred)
    # 求均值
    return np.mean(diff)

# 使用示例
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
print(f"MAE: {mae_loss(y_true, y_pred):.4f}")  # 输出 0.5

# ==================== PyTorch实现 ====================
# 内置L1Loss
criterion = nn.L1Loss()
loss = criterion(y_pred_t, y_true_t)
```

**案例：股票价格预测**。股票数据常含突发波动（如财报公布、政策变动）导致的异常值。使用 MAE 可避免模型被这些异常值带偏，更稳健地捕捉长期趋势。


##### 3.1.3 Huber Loss（平滑 L1 Loss）

**函数定义**：

\[
\text{Huber}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2, & |y - \hat{y}| > \delta \end{cases}
\]

**公式解释**：
- \(\delta\)：超参数，控制平方区和线性区的分界点，通常取 1.0。
- 当误差绝对值 \(|y - \hat{y}| \le \delta\) 时，使用平方损失（类似 MSE），保证小误差时梯度平滑、收敛快。
- 当误差绝对值 \(|y - \hat{y}| > \delta\) 时，使用线性损失（类似 MAE），避免大误差被平方放大，从而对异常值鲁棒。
- 第二段的 \(-\frac{1}{2}\delta^2\) 是常数项，用于保证函数在 \(|y-\hat{y}| = \delta\) 处连续。验证：\(\frac{1}{2}\delta^2 = \delta \cdot \delta - \frac{1}{2}\delta^2\)，两边相等。

**导数公式**：

\[
\frac{\partial \text{Huber}}{\partial \hat{y}} = \begin{cases} \hat{y} - y, & |y - \hat{y}| \le \delta \\ \delta \cdot \text{sign}(\hat{y} - y), & |y - \hat{y}| > \delta \end{cases}
\]

**公式解释**：
- 小误差区：梯度与误差成正比，误差越大梯度越大，收敛快。
- 大误差区：梯度恒定 ±δ，不会因异常值而爆炸。

**优缺点**：
- 优点：兼具 MSE 的平滑性和 MAE 的鲁棒性，是两者的折中。
- 缺点：需调节超参数 δ，δ 的选择依赖数据尺度。

```python
# ==================== 从零实现 Huber Loss ====================
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss - 从零实现"""
    # 计算误差
    diff = y_true - y_pred
    # 计算绝对误差
    abs_diff = np.abs(diff)
    # 小误差区：使用平方损失
    small = 0.5 * diff ** 2
    # 大误差区：使用线性损失
    large = delta * abs_diff - 0.5 * delta ** 2
    # 根据阈值选择
    return np.mean(np.where(abs_diff <= delta, small, large))

print(f"Huber: {huber_loss(y_true, y_pred):.4f}")

# ==================== PyTorch实现 ====================
criterion = nn.HuberLoss(delta=1.0)
loss = criterion(y_pred_t, y_true_t)
```

**案例：目标检测中的边界框回归**。Faster R-CNN、YOLO 等目标检测模型在回归边界框坐标时广泛使用 Huber Loss，因为标注框的位置可能存在噪声，Huber 能兼顾精度和鲁棒性。

##### 3.1.3.1 Smooth L1 Loss

Smooth L1 损失函数，又称 **Huber 损失的一种特例**，最早由 Ross Girshick 在 Fast R-CNN 论文中提出，用于目标检测中的边界框回归。它旨在结合 MSE（L2 Loss）和 MAE（L1 Loss）的优点：在误差较小时保持平方损失的平滑可导特性，在误差较大时保持线性损失的鲁棒性。

Smooth L1 与 Huber Loss 在数学形式上有细微差别：Huber Loss 使用参数 \(\delta\) 控制分界点，而 Smooth L1 通常将分界点固定为 1，并且第二段的常数项有所调整。


###### 3.1.3.1 函数定义

**分段形式**：

\[
\text{SmoothL1}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| < 1 \\ |y - \hat{y}| - \frac{1}{2}, & |y - \hat{y}| \ge 1 \end{cases}
\]

**公式解释**：
- \(y\)：真实值（如目标检测中标注框的坐标）。
- \(\hat{y}\)：预测值（模型输出的边界框坐标）。
- \(|y - \hat{y}|\)：预测误差的绝对值。
- **第一段**：当误差绝对值小于 1 时，损失为 \(\frac{1}{2}(y-\hat{y})^2\)。此时损失函数是二次的，曲线光滑，梯度随误差线性变化，便于优化收敛。
- **第二段**：当误差绝对值大于等于 1 时，损失为 \(|y - \hat{y}| - \frac{1}{2}\)。此时损失函数是线性的，梯度恒定为 ±1，不会因大误差而爆炸。
- **常数项 \(-\frac{1}{2}\)**：用于保证函数在 \(|y - \hat{y}| = 1\) 处连续。验证：第一段当 \(|y-\hat{y}|=1\) 时，损失为 \(\frac{1}{2} \cdot 1^2 = \frac{1}{2}\)；第二段当 \(|y-\hat{y}|=1\) 时，损失为 \(1 - \frac{1}{2} = \frac{1}{2}\)。两段相等，函数连续。

**统一形式**：

Smooth L1 也可以用如下紧凑形式表示：

\[
\text{SmoothL1}(y, \hat{y}) = \begin{cases} 0.5 (y - \hat{y})^2, & |y - \hat{y}| < 1 \\ |y - \hat{y}| - 0.5, & \text{其他} \end{cases}
\]

这一形式也是 PyTorch 中 `nn.SmoothL1Loss` 的实现方式（默认 `beta=1.0`）。

**与 Huber Loss 的对比**：

| 特性 | Huber Loss | Smooth L1 |
|------|-----------|-----------|
| 分界点 | 参数 \(\delta\)，可调 | 固定为 1 |
| 第二段形式 | \(\delta |y-\hat{y}| - \frac{1}{2}\delta^2\) | \(|y-\hat{y}| - \frac{1}{2}\) |
| 当 \(\delta = 1\) | 与 Smooth L1 完全相同 | — |
| 常见场景 | 通用回归 | 目标检测 |

若将 Huber Loss 的 \(\delta\) 设为 1，两者在数学上完全等价：

\[
\text{Huber}_{\delta=1}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2, & |y-\hat{y}| \le 1 \\ |y-\hat{y}| - \frac{1}{2}, & |y-\hat{y}| > 1 \end{cases} = \text{SmoothL1}(y, \hat{y})
\]

因此，Smooth L1 可以理解为 Huber Loss 的一个特例。



###### 3.1.3.2 导数公式

**分段导数**：

\[
\frac{\partial \text{SmoothL1}}{\partial \hat{y}} = \begin{cases} \hat{y} - y, & |y - \hat{y}| < 1 \\ \text{sign}(\hat{y} - y), & |y - \hat{y}| \ge 1 \end{cases}
\]

**公式解释**：
- **第一段**（误差小于 1）：梯度为 \(\hat{y} - y\)，与误差成正比。误差越大，梯度越大，参数更新步长越大，收敛越快。
- **第二段**（误差大于等于 1）：梯度为 \(\text{sign}(\hat{y} - y)\)，即 ±1。无论误差多大，梯度大小始终为 1，避免了 L2 损失中梯度随误差线性增长导致的梯度爆炸。
- \(\text{sign}(\cdot)\)：符号函数，当 \(\hat{y} > y\) 时返回 +1，当 \(\hat{y} < y\) 时返回 -1。

**梯度行为分析**：

| 误差大小 | 损失值 | 梯度大小 | 特点 |
|---------|-------|---------|------|
| 0 | 0 | 0 | 完全正确，不更新 |
| 0.5 | 0.125 | 0.5 | 平滑区，梯度小 |
| 1 | 0.5 | 1 | 分界点 |
| 10 | 9.5 | 1 | 线性区，梯度饱和 |
| 100 | 99.5 | 1 | 线性区，梯度仍为 1 |

**关键性质**：
- 梯度有界：\(|\frac{\partial \text{SmoothL1}}{\partial \hat{y}}| \le 1\)，这保证了即使遇到异常值，梯度也不会爆炸。
- 平滑可导：在 \(|y-\hat{y}| = 1\) 处，左导数 \(= 1\)，右导数 \(= 1\)，函数一阶连续可导。
- 在 \(y = \hat{y}\) 处，导数为 0，损失最小。

**二阶导数**：

\[
\frac{\partial^2 \text{SmoothL1}}{\partial \hat{y}^2} = \begin{cases} 1, & |y - \hat{y}| < 1 \\ 0, & |y - \hat{y}| > 1 \end{cases}
\]

在分界点 \(|y-\hat{y}| = 1\) 处二阶导数不连续，但一阶导数连续，因此对梯度下降法而言足够平滑。


###### 3.1.3.3 概率意义

Smooth L1 没有严格的单一概率分布对应的最大似然估计，但可以理解为**高斯分布与拉普拉斯分布的混合**：

- 小误差区（\(|y-\hat{y}| < 1\)）：损失为二次，对应高斯噪声假设。
- 大误差区（\(|y-\hat{y}| \ge 1\)）：损失为线性，对应拉普拉斯噪声假设。

因此，Smooth L1 假设数据中同时存在**小幅度高斯噪声和大幅度异常值**，模型在拟合正常数据的同时对异常值保持鲁棒。

---

###### 3.1.3.4 与其他回归损失的对比

**函数曲线对比**：

| 误差 \(|y-\hat{y}|\) | MSE 损失 | MAE 损失 | Smooth L1 损失 |
|--------------------|---------|---------|---------------|
| 0.1 | 0.01 | 0.1 | 0.005 |
| 0.5 | 0.25 | 0.5 | 0.125 |
| 1.0 | 1.0 | 1.0 | 0.5 |
| 2.0 | 4.0 | 2.0 | 1.5 |
| 5.0 | 25.0 | 5.0 | 4.5 |
| 10.0 | 100.0 | 10.0 | 9.5 |
| 100.0 | 10000.0 | 100.0 | 99.5 |

**梯度对比**：

| 误差 \(|y-\hat{y}|\) | MSE 梯度 | MAE 梯度 | Smooth L1 梯度 |
|--------------------|---------|---------|---------------|
| 0.5 | 1.0 | ±1 | 0.5 |
| 1.0 | 2.0 | ±1 | 1.0 |
| 10.0 | 20.0 | ±1 | 1.0 |
| 100.0 | 200.0 | ±1 | 1.0 |

**结论**：
- MSE 梯度随误差线性增长，对大误差极其敏感，易导致梯度爆炸。
- MAE 梯度恒定为 ±1，但零点不可导，训练后期容易震荡。
- Smooth L1 兼顾两者：小误差时梯度平滑衰减（利于收敛），大误差时梯度有界（防止爆炸）。

---

###### 3.1.3.5 代码实现

从零实现

```python
# 导入数值计算库
import numpy as np
# 导入PyTorch
import torch
import torch.nn as nn

# ==================== 从零实现 Smooth L1 Loss ====================
def smooth_l1_loss(y_true, y_pred, beta=1.0):
    """Smooth L1损失 - 从零实现
    
    参数:
        y_true: 真实值，形状(n,)或(n, d)
        y_pred: 预测值，形状(n,)或(n, d)
        beta: 分界点，默认1.0
    返回:
        平均Smooth L1损失（标量）
    """
    # 计算逐元素误差
    diff = y_true - y_pred
    # 计算误差的绝对值
    abs_diff = np.abs(diff)
    # 第一段：误差小于beta，使用0.5 * diff^2 / beta
    # 除以beta是为了让两段在beta处导数连续
    small_loss = 0.5 * diff ** 2 / beta
    # 第二段：误差大于等于beta，使用abs_diff - 0.5 * beta
    large_loss = abs_diff - 0.5 * beta
    # 根据阈值逐元素选择
    loss = np.where(abs_diff < beta, small_loss, large_loss)
    # 对所有元素求平均
    return np.mean(loss)

# 构造示例数据
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.5, 2.2, 1.0, 4.1, 10.0])

# 计算Smooth L1损失
loss = smooth_l1_loss(y_true, y_pred)
print(f"Smooth L1: {loss:.4f}")
# 逐样本损失：0.125, 0.02, 1.5, 0.005, 4.5
# 平均：(0.125+0.02+1.5+0.005+4.5)/5 = 1.23
```

**逐行解释**：

- `diff = y_true - y_pred`：计算每个样本的误差。例如第一个样本 \(1.0 - 1.5 = -0.5\)。
- `abs_diff = np.abs(diff)`：取绝对值，得到 \(0.5, 0.2, 2.0, 0.1, 5.0\)。
- `small_loss = 0.5 * diff ** 2 / beta`：计算第一段损失。注意这里除以了 `beta`，当 `beta=1` 时与前面公式一致。当 `beta` 可变时，这个形式保证两段在分界点处导数连续。例如误差 0.5 时，损失为 \(0.5 \times 0.25 = 0.125\)。
- `large_loss = abs_diff - 0.5 * beta`：计算第二段损失。例如误差 5.0 时，损失为 \(5.0 - 0.5 = 4.5\)。
- `loss = np.where(abs_diff < beta, small_loss, large_loss)`：逐元素选择。误差小于 beta 的用第一段，否则用第二段。
- `np.mean(loss)`：对所有样本求平均，得到最终损失。

PyTorch 实现

```python
# ==================== PyTorch实现 ====================
# 方式一：使用nn.SmoothL1Loss，默认beta=1.0
criterion = nn.SmoothL1Loss()
# 转换为张量
y_true_t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_t = torch.tensor([1.5, 2.2, 1.0, 4.1, 10.0], requires_grad=True)
# 计算损失
loss = criterion(y_pred_t, y_true_t)
print(f"PyTorch Smooth L1: {loss.item():.4f}")
# 反向传播
loss.backward()
# 查看梯度（应为分段形式：小误差区为diff，大误差区为sign）
print(f"梯度: {y_pred_t.grad.numpy()}")

# 方式二：使用nn.HuberLoss，设置delta=1.0，等价于Smooth L1
criterion_huber = nn.HuberLoss(delta=1.0)
loss_huber = criterion_huber(y_pred_t, y_true_t)
print(f"Huber (delta=1.0): {loss_huber.item():.4f}")
# 两者数值应相同

# 方式三：使用F.smooth_l1_loss函数式接口
import torch.nn.functional as F
loss_f = F.smooth_l1_loss(y_pred_t, y_true_t, beta=1.0)
print(f"F.smooth_l1_loss: {loss_f.item():.4f}")

# 方式四：自定义beta的Smooth L1
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # 计算误差
        diff = y_pred - y_true
        # 计算绝对值
        abs_diff = torch.abs(diff)
        # 分段计算
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )
        # 根据reduction返回
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# 使用自定义损失
criterion_custom = SmoothL1Loss(beta=1.0)
loss_custom = criterion_custom(y_pred_t, y_true_t)
print(f"自定义 Smooth L1: {loss_custom.item():.4f}")
```

**梯度输出解读**：

对 y_pred = [1.5, 2.2, 1.0, 4.1, 10.0] 和 y_true = [1.0, 2.0, 3.0, 4.0, 5.0]：

| 样本 | diff = y_pred - y_true | 误差大小 | 所在区间 | 梯度 |
|------|----------------------|---------|---------|------|
| 1 | 0.5 | 0.5 < 1 | 平滑区 | 0.5 |
| 2 | 0.2 | 0.2 < 1 | 平滑区 | 0.2 |
| 3 | -2.0 | 2.0 ≥ 1 | 线性区 | -1.0 |
| 4 | 0.1 | 0.1 < 1 | 平滑区 | 0.1 |
| 5 | 5.0 | 5.0 ≥ 1 | 线性区 | 1.0 |

注意第 3 个样本预测为 1.0，真实为 3.0，误差 -2.0，梯度为 -1.0（符号函数），而非 -2.0（线性增长）。这就是 Smooth L1 对大误差梯度饱和的效果。


###### 3.1.3.6 实际案例：目标检测中的边界框回归

Smooth L1 最经典的应用是 **Fast R-CNN / Faster R-CNN** 中的边界框回归。目标检测模型需要预测边界框的四个坐标 \((x, y, w, h)\)，即中心坐标和宽高。直接使用 MSE 会因大误差导致梯度爆炸，使用 MAE 又会在零点附近不稳定。Smooth L1 完美解决了这一问题。

```python
# ==================== 目标检测边界框回归示例 ====================
import torch
import torch.nn as nn

# 假设一个批次有4个候选框
# 真实边界框坐标 [x_center, y_center, width, height]
gt_boxes = torch.tensor([
    [100.0, 150.0, 50.0, 80.0],
    [200.0, 300.0, 60.0, 90.0],
    [50.0,  80.0,  30.0, 40.0],
    [400.0, 500.0, 100.0, 120.0]
])

# 模型预测的边界框坐标（含较大误差）
pred_boxes = torch.tensor([
    [105.0, 148.0, 52.0, 78.0],   # 误差较小
    [210.0, 320.0, 65.0, 95.0],   # 中等误差
    [55.0,  90.0,  35.0, 45.0],   # 小误差
    [500.0, 600.0, 150.0, 200.0]  # 大误差（异常值）
], requires_grad=True)

# 使用Smooth L1损失
criterion = nn.SmoothL1Loss(reduction='mean')
loss = criterion(pred_boxes, gt_boxes)
print(f"边界框回归Smooth L1损失: {loss.item():.4f}")

# 反向传播
loss.backward()
print(f"梯度:\n{pred_boxes.grad.numpy()}")

# 对比：使用MSE
criterion_mse = nn.MSELoss()
pred_boxes2 = pred_boxes.detach().clone().requires_grad_(True)
loss_mse = criterion_mse(pred_boxes2, gt_boxes)
loss_mse.backward()
print(f"MSE损失: {loss_mse.item():.4f}")
print(f"MSE梯度:\n{pred_boxes2.grad.numpy()}")
```

**结果分析**：

第 4 个样本（大误差）：
- Smooth L1 损失：每个坐标误差约 100～100，损失值约 99.5 × 4 = 398（线性区）。
- MSE 损失：每个坐标误差平方约 10000，损失值约 10000 × 4 = 40000。
- Smooth L1 梯度：±1（有界）。
- MSE 梯度：约 ±100（梯度爆炸风险）。

可以看到，Smooth L1 对异常值样本的损失和梯度都远小于 MSE，训练更稳定。

###### 3.1.3.7 变体：带 beta 参数的 Smooth L1

PyTorch 的 `nn.SmoothL1Loss` 支持 `beta` 参数（旧版本称为 `reduction` 无关的 `beta`，新版本默认 1.0）：

\[
\text{SmoothL1}_\beta(y, \hat{y}) = \begin{cases} \frac{1}{2\beta}(y-\hat{y})^2, & |y-\hat{y}| < \beta \\ |y-\hat{y}| - \frac{\beta}{2}, & |y-\hat{y}| \ge \beta \end{cases}
\]

**公式解释**：
- \(\beta\)：分界点参数，控制平滑区和线性区的切换位置。
- 当 \(\beta\) 较小时，更多误差落在平滑区，损失接近 MSE。
- 当 \(\beta\) 较大时，更多误差落在平方区（因为分界点大），但 \(\beta\) 过大等价于纯 MSE。
- \(\beta\) 过小则等价于纯 MAE。
- 第一段的 \(\frac{1}{2\beta}\) 保证在 \(|y-\hat{y}| = \beta\) 处两段导数值连续（左导数 \(= 1\)，右导数 \(= 1\)）。

**导数公式**：

\[
\frac{\partial \text{SmoothL1}_\beta}{\partial \hat{y}} = \begin{cases} \frac{\hat{y} - y}{\beta}, & |y-\hat{y}| < \beta \\ \text{sign}(\hat{y} - y), & |y-\hat{y}| \ge \beta \end{cases}
\]

**代码实现**：

```python
# 使用PyTorch内置的beta参数
criterion = nn.SmoothL1Loss(beta=0.5)
loss = criterion(pred_boxes, gt_boxes)
print(f"beta=0.5 的损失: {loss.item():.4f}")

criterion = nn.SmoothL1Loss(beta=2.0)
loss = criterion(pred_boxes, gt_boxes)
print(f"beta=2.0 的损失: {loss.item():.4f}")
```

**beta 的选择建议**：
- 目标检测中，坐标的尺度差异较大。Fast R-CNN 原文将边界框坐标归一化后，使用默认 \(\beta = 1.0\)。
- 若预测目标尺度较大（如像素级坐标），可适当增大 \(\beta\)。
- 若目标尺度较小，可减小 \(\beta\)。


###### 3.1.3.8 应用场景汇总

| 应用场景 | 说明 |
|---------|------|
| Faster R-CNN | 边界框回归分支的标准损失 |
| Fast R-CNN | 同上，最早使用 Smooth L1 的论文 |
| SSD | 目标检测中的定位损失 |
| YOLO | 某些版本中使用 Smooth L1 回归边界框 |
| 关键点检测 | 人脸、人体姿态的关键点坐标回归 |
| 单目深度估计 | 像素级深度值回归，对异常深度值鲁棒 |
| 机器人控制 | 连续动作值回归 |

---

###### 3.1.3.9 核心结论

Smooth L1 是回归损失中"鱼与熊掌兼得"的设计：**小误差时用平方损失保证平滑收敛，大误差时用线性损失防止梯度爆炸**。它通过固定分界点（默认 1.0）简化了 Huber Loss 的超参数调节，成为目标检测领域的事实标准。

理解 Smooth L1 的关键在于理解其**梯度行为**：梯度有界（不超过 1）是它抗异常值的根本原因。当反向传播经过一个误差为 100 的样本时，MSE 会传递 100 倍放大的梯度，而 Smooth L1 只传递 1 倍梯度。这一性质使得训练过程不会被少数异常样本主导，模型能够稳定地学习整体数据分布。

在实践中，若遇到以下情形，应优先考虑 Smooth L1：
1. 回归目标中存在异常值或标注噪声。
2. 训练过程中损失出现剧烈震荡或梯度爆炸。
3. 需要兼顾收敛速度和鲁棒性，且不希望手动调节 Huber 的 δ。

##### 3.1.4 分位数损失（Quantile Loss）

**函数定义**：

\[
\text{Quantile}(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}), & y \ge \hat{y} \\ (1 - \tau)(\hat{y} - y), & y < \hat{y} \end{cases}
\]

**公式解释**：
- \(\tau \in (0, 1)\)：分位数参数。例如 \(\tau = 0.5\) 对应中位数，\(\tau = 0.9\) 对应 90% 分位数。
- 当真实值 \(y\) 大于预测值 \(\hat{y}\) 时（预测偏低），损失为 \(\tau (y - \hat{y})\)。
- 当真实值 \(y\) 小于预测值 \(\hat{y}\) 时（预测偏高），损失为 \((1-\tau)(\hat{y} - y)\)。
- 若 \(\tau = 0.9\)，预测偏低时的惩罚是预测偏高时的 9 倍，因此模型会倾向输出较大的预测值，使约 90% 的真实值落在预测值之下。

**应用**：分位数损失用于**区间预测**。若同时用 \(\tau = 0.1\) 和 \(\tau = 0.9\) 训练两个模型，可得到 80% 置信区间。

```python
# ==================== 从零实现 Quantile Loss ====================
def quantile_loss(y_true, y_pred, tau=0.5):
    """分位数损失"""
    # 计算误差
    diff = y_true - y_pred
    # 当diff>=0时，损失为tau*diff；否则为(1-tau)*(-diff)
    return np.mean(np.maximum(tau * diff, (1 - tau) * (-diff)))

print(f"Quantile (tau=0.5): {quantile_loss(y_true, y_pred, 0.5):.4f}")
print(f"Quantile (tau=0.9): {quantile_loss(y_true, y_pred, 0.9):.4f}")

# ==================== PyTorch实现 ====================
def quantile_loss_torch(y_pred, y_true, tau=0.5):
    diff = y_true - y_pred
    return torch.mean(torch.max(tau * diff, (1 - tau) * (-diff)))
```

**案例：电力负荷预测**。电力公司不仅需要预测平均负荷，还需要预测高峰负荷的上界，以便预留发电容量。使用 \(\tau = 0.95\) 的分位数损失可预测 95% 分位数负荷，为调度提供安全边界。



##### 3.1.5 回归损失函数选择指南

| 损失函数 | 对异常值 | 可导性 | 适用场景 |
|---------|---------|--------|---------|
| MSE | 敏感 | 处处可导 | 高斯噪声、无异常值 |
| MAE | 鲁棒 | 零点不可导 | 拉普拉斯噪声、有异常值 |
| Huber | 较鲁棒 | 处处可导 | 综合场景、目标检测 |
| Quantile | — | 处处可导 | 区间预测 |


#### 三·二 二分类任务的损失函数

二分类任务的目标是将样本分为两个类别（如正/负、是/否）。模型通常输出一个概率值 \(\hat{y} \in (0, 1)\)，表示样本属于正类的概率。

##### 3.2.1 二元交叉熵（Binary Cross-Entropy, BCE）

**函数定义**：

\[
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
\]

**公式解释**：
- \(n\)：样本总数。
- \(y_i \in \{0, 1\}\)：第 \(i\) 个样本的真实标签。1 表示正类，0 表示负类。
- \(\hat{y}_i \in (0, 1)\)：模型预测为正类的概率，通常由 Sigmoid 函数产生。
- \(y_i \log(\hat{y}_i)\)：当真实标签为正类（\(y_i = 1\)）时生效。若预测概率 \(\hat{y}_i\) 接近 1，则 \(\log(\hat{y}_i)\) 接近 0，损失小；若 \(\hat{y}_i\) 接近 0，则 \(\log(\hat{y}_i)\) 趋于负无穷，损失趋于正无穷。
- \((1-y_i)\log(1-\hat{y}_i)\)：当真实标签为负类（\(y_i = 0\)）时生效。若预测概率 \(\hat{y}_i\) 接近 0，则 \(\log(1-\hat{y}_i)\) 接近 0，损失小；若接近 1，损失很大。
- 负号：因为 \(\log\) 在 \((0,1)\) 区间为负值，取负号后损失为正。
- \(\frac{1}{n} \sum\)：对所有样本求平均。

**简化形式**：由于 \(y_i\) 只能取 0 或 1，两项中每次只有一项生效，BCE 可简写为：

\[
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \log \hat{p}_i
\]

其中 \(\hat{p}_i = y_i \hat{y}_i + (1-y_i)(1-\hat{y}_i)\)，即模型预测为**真实类别**的概率。

**概率意义**：

BCE 等价于**伯努利分布下的最大似然估计**。对每个样本，真实标签 \(y_i\) 服从参数为 \(\hat{y}_i\) 的伯努利分布：

\[
p(y_i | \hat{y}_i) = \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}
\]

取负对数似然，即得 BCE。

**导数公式**：

\[
\frac{\partial \text{BCE}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)}
\]

**公式解释**：
- \(\hat{y}_i - y_i\)：预测概率与真实标签的差。
- \(\hat{y}_i(1-\hat{y}_i)\)：分母，正是 Sigmoid 函数的导数形式。
- 若将 BCE 与 Sigmoid 结合（即 Sigmoid + BCE），链式法则后可得简洁的梯度：

\[
\frac{\partial \text{BCE}}{\partial z_i} = \frac{1}{n}(\hat{y}_i - y_i)
\]

其中 \(z_i\) 是 Sigmoid 的输入 logit。这个结果与 Softmax + 交叉熵的梯度形式完全一致。

**优缺点**：
- 优点：梯度平滑，与 Sigmoid 配合时梯度形式简洁；概率解释清晰。
- 缺点：对类别不平衡敏感。若正类占 1%，模型倾向于全部预测为负类，此时 BCE 仍较小，但召回率为 0。

**代码实现**：

```python
# ==================== 从零实现 BCE ====================
def binary_cross_entropy(y_true, y_pred):
    """二元交叉熵 - 从零实现"""
    # 设置极小值epsilon，防止log(0)导致数值溢出
    epsilon = 1e-15
    # 将预测值裁剪到[epsilon, 1-epsilon]区间
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # 计算二元交叉熵
    # y_true为1时取log(y_pred)，为0时取log(1-y_pred)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 示例：5个样本的真实标签和预测概率
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.4, 0.2])

print(f"BCE: {binary_cross_entropy(y_true, y_pred):.4f}")

# ==================== PyTorch实现 ====================
# 方式一：BCELoss（输入需为概率，即经过Sigmoid后的值）
criterion = nn.BCELoss()
# 方式二：BCEWithLogitsLoss（输入为logits，内部自动做Sigmoid，数值更稳定）
criterion_logits = nn.BCEWithLogitsLoss()

y_true_t = torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32)
y_pred_t = torch.tensor([0.9, 0.1, 0.8, 0.4, 0.2], requires_grad=True)
loss = criterion(y_pred_t, y_true_t)
loss.backward()
print(f"梯度: {y_pred_t.grad.numpy()}")
```

**案例：垃圾邮件分类**。输入邮件文本，输出是否为垃圾邮件。若数据集正负样本比例约为 1:1，BCE 是标准选择。若垃圾邮件仅占 5%，则需考虑加权 BCE 或 Focal Loss。

##### 3.2.2 加权二元交叉熵（Weighted BCE）

**函数定义**：

\[
\text{WBCE} = -\frac{1}{n} \sum_{i=1}^{n} [w_1 \cdot y_i \log(\hat{y}_i) + w_0 \cdot (1-y_i)\log(1-\hat{y}_i)]
\]

**公式解释**：
- \(w_1\)：正类样本的权重。
- \(w_0\)：负类样本的权重。
- 通常设置 \(w_1 = \frac{n}{2 n_1}\)，\(w_0 = \frac{n}{2 n_0}\)，其中 \(n_1, n_0\) 分别是正负样本数。这样两类对总损失的贡献相等。
- 当正类稀少时，\(w_1\) 较大，使模型更关注正类样本，提升召回率。

**代码实现**：

```python
# ==================== PyTorch实现 ====================
# pos_weight为正类权重，通常设为负样本数/正样本数
# 若正类100个，负类900个，则pos_weight=9
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))

# 或手动加权
def weighted_bce(y_true, y_pred, w1=1.0, w0=1.0):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(w1 * y_true * np.log(y_pred) + w0 * (1 - y_true) * np.log(1 - y_pred))
```

**案例：医学影像诊断**。在癌症筛查中，阳性样本（患病）远少于阴性样本。若不加权，模型可能全部预测为"健康"就能获得 99% 的准确率，但漏诊率极高。加权 BCE 使模型更关注阳性样本，降低漏诊风险。

##### 3.2.3 Focal Loss

**函数定义**：

\[
\text{FL}(y, \hat{y}) = \begin{cases} -\alpha (1-\hat{y})^\gamma \log(\hat{y}), & y = 1 \\ -(1-\alpha) \hat{y}^\gamma \log(1-\hat{y}), & y = 0 \end{cases}
\]

**公式解释**：
- \(\alpha \in (0, 1)\)：类别权重，用于平衡正负样本，通常取 0.25。
- \(\gamma \ge 0\)：聚焦参数，通常取 2.0。
- \((1-\hat{y})^\gamma\)：当真实标签为正类且预测概率 \(\hat{y}\) 接近 1（易分类样本）时，\((1-\hat{y})^\gamma\) 接近 0，损失被大幅降低；当 \(\hat{y}\) 接近 0（难分类样本）时，\((1-\hat{y})^\gamma\) 接近 1，损失几乎不变。
- 同理，\(\hat{y}^\gamma\) 对负类的易分类样本降权。
- 效果：模型自动**聚焦于难分类样本**，减少易分类样本的主导作用。

**导数公式**（以正类为例）：

\[
\frac{\partial \text{FL}}{\partial \hat{y}} = -\alpha \left[ \gamma (1-\hat{y})^{\gamma-1} \log(\hat{y}) \cdot (-1) \cdot \hat{y} + (1-\hat{y})^\gamma \cdot \frac{1}{\hat{y}} \right]
\]

**公式解释**：
- 梯度包含两部分：一部分来自 \((1-\hat{y})^\gamma\) 的导数，另一部分来自 \(\log(\hat{y})\) 的导数。
- 当 \(\hat{y} \to 1\) 时，\((1-\hat{y})^\gamma \to 0\)，梯度趋于 0，易分类样本几乎不更新。
- 当 \(\hat{y} \to 0\) 时，\((1-\hat{y})^\gamma \to 1\)，梯度接近普通 BCE，难分类样本正常更新。

**代码实现**：

```python
# ==================== 从零实现 Focal Loss ====================
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal Loss - 从零实现"""
    epsilon = 1e-15
    # 裁剪预测值，防止log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # 正类损失：-alpha * (1-y_pred)^gamma * log(y_pred)
    pos_loss = -alpha * (1 - y_pred) ** gamma * np.log(y_pred)
    # 负类损失：-(1-alpha) * y_pred^gamma * log(1-y_pred)
    neg_loss = -(1 - alpha) * y_pred ** gamma * np.log(1 - y_pred)
    # 根据真实标签选择
    loss = y_true * pos_loss + (1 - y_true) * neg_loss
    return np.mean(loss)

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
print(f"Focal Loss: {focal_loss(y_true, y_pred):.4f}")

# ==================== PyTorch实现（多分类版本） ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算交叉熵（内部含Softmax）
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        # 计算pt = exp(-ce_loss)，即预测为真实类别的概率
        pt = torch.exp(-ce_loss)
        # Focal Loss = alpha * (1-pt)^gamma * CE
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()
```

**案例：目标检测（RetinaNet）**。目标检测中，背景锚框（负样本）数量远超前景锚框（正样本），比例可达 1000:1。Focal Loss 是 RetinaNet 的核心创新，通过降低易分类背景样本的权重，使模型专注于难分类的前景目标，在保持精度的同时解决了极端类别不平衡问题。

##### 3.2.4 Hinge Loss（合页损失）

**函数定义**：

\[
\text{Hinge}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
\]

**公式解释**：
- \(y \in \{-1, +1\}\)：真实标签，使用 ±1 而非 0/1。
- \(\hat{y}\)：模型输出的原始分数（未经过 Sigmoid），可以是任意实数。
- \(y \cdot \hat{y}\)：若预测方向正确（\(y\) 与 \(\hat{y}\) 同号），乘积为正；若方向错误，乘积为负。
- \(1 - y \cdot \hat{y}\)：要求正确类别的分数至少比错误类别高 1（间隔 margin）。
- \(\max(0, \cdot)\)：若 \(y \cdot \hat{y} \ge 1\)，即分类正确且间隔足够，损失为 0；否则损失为 \(1 - y \cdot \hat{y}\)。

**导数公式**：

\[
\frac{\partial \text{Hinge}}{\partial \hat{y}} = \begin{cases} -y, & y \cdot \hat{y} < 1 \\ 0, & y \cdot \hat{y} \ge 1 \end{cases}
\]

**公式解释**：
- 当样本被正确分类且间隔大于 1 时，梯度为 0，不再更新，模型"满意"。
- 当间隔不足或分类错误时，梯度为 \(-y\)，推动 \(\hat{y}\) 向正确方向移动。

**应用**：Hinge Loss 是支持向量机（SVM）的核心损失。在深度学习中，它主要用于需要"间隔"的任务，如人脸验证中的三元组损失变体。

```python
# ==================== 从零实现 Hinge Loss ====================
def hinge_loss(y_true, y_pred):
    """Hinge Loss - 从零实现，y_true取值为-1或+1"""
    # max(0, 1 - y_true * y_pred)
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# 示例：标签用±1表示
y_true = np.array([1, -1, 1, 1, -1])
y_pred = np.array([0.8, -0.5, 0.3, -0.2, -0.9])
print(f"Hinge: {hinge_loss(y_true, y_pred):.4f}")

# ==================== PyTorch实现 ====================
criterion = nn.HingeEmbeddingLoss()
```

**案例：人脸验证**。判断两张人脸是否属于同一人。Hinge Loss 要求同类人脸的特征距离小于异类人脸距离减去一个间隔，确保模型学到的特征具有判别性。

##### 3.2.5 二分类损失函数选择指南

| 损失函数 | 适用场景 | 关键参数 |
|---------|---------|---------|
| BCE | 类别平衡的标准二分类 | 无 |
| Weighted BCE | 类别不平衡 | pos_weight |
| Focal Loss | 极端类别不平衡、难样本多 | α, γ |
| Hinge | 需要间隔的判别任务 | margin |


#### 三·三 多分类任务的损失函数

多分类任务的目标是将样本分到 \(C\) 个类别中的一个（\(C > 2\)）。模型输出 \(C\) 个分数（logits），经过 Softmax 后得到概率分布。

##### 3.3.1 多分类交叉熵（Categorical Cross-Entropy）

**函数定义**：

\[
\text{CE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
\]

**公式解释**：
- \(n\)：样本总数。
- \(C\)：类别总数。
- \(y_{i,c}\)：第 \(i\) 个样本的真实标签的 one-hot 编码。若真实类别为 \(c^*\)，则 \(y_{i,c^*} = 1\)，其余为 0。
- \(\hat{y}_{i,c}\)：模型预测第 \(i\) 个样本属于第 \(c\) 类的概率，由 Softmax 计算，满足 \(\sum_c \hat{y}_{i,c} = 1\)。
- 由于 \(y_{i,c}\) 只有一个位置为 1，内层求和实际只取真实类别那一项：

\[
\text{CE} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i,c^*})
\]

即损失等于**真实类别预测概率的负对数**。
- 若预测为真实类别的概率 \(\hat{y}_{i,c^*}\) 接近 1，损失接近 0；若接近 0，损失趋于正无穷。

**概率意义**：

CE 等价于**多项分布下的最大似然估计**。对每个样本，真实标签服从参数为 \(\hat{\boldsymbol{y}}_i\) 的分类分布：

\[
p(y_i | \hat{\boldsymbol{y}}_i) = \prod_{c=1}^{C} \hat{y}_{i,c}^{y_{i,c}}
\]

取负对数似然，即得 CE。

**Softmax 函数**：

\[
\hat{y}_{i,c} = \text{Softmax}(z_{i,c}) = \frac{e^{z_{i,c}}}{\sum_{k=1}^{C} e^{z_{i,k}}}
\]

**公式解释**：
- \(z_{i,c}\)：第 \(i\) 个样本第 \(c\) 类的 logit。
- 分子 \(e^{z_{i,c}}\)：将 logit 映射为正数。
- 分母 \(\sum_k e^{z_{i,k}}\)：所有类别指数的和，作为归一化因子。
- 输出满足 \(\hat{y}_{i,c} \in (0,1)\) 且 \(\sum_c \hat{y}_{i,c} = 1\)。

**导数公式**（Softmax + CE 联合）：

\[
\frac{\partial \text{CE}}{\partial z_{i,c}} = \frac{1}{n} (\hat{y}_{i,c} - y_{i,c})
\]

**推导要点**：
- 先对 \(\hat{y}_{i,c}\) 求导，再乘以 Softmax 的雅可比矩阵，链式法则后化简。
- 结果极为简洁：**预测概率与真实标签的差值**。
- 若预测正确（\(\hat{y}_{i,c^*} \approx 1\)，\(y_{i,c^*} = 1\)），梯度接近 0；若预测错误（\(\hat{y}_{i,c^*} \approx 0\)，\(y_{i,c^*} = 1\)），梯度为负，参数向提高该类别概率的方向更新。

**代码实现**：

```python
# ==================== 从零实现 Softmax + CE ====================
def softmax(z):
    """数值稳定的Softmax"""
    # 减去每行最大值，防止exp溢出
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    # 计算指数
    exp_z = np.exp(z_shifted)
    # 归一化
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(logits, labels):
    """多分类交叉熵 - 从零实现"""
    # 计算Softmax概率
    probs = softmax(logits)
    # 取真实类别对应的概率
    n = len(labels)
    # 对每个样本取log(prob[真实类别])
    log_likelihood = -np.log(probs[np.arange(n), labels] + 1e-15)
    # 求平均
    return np.mean(log_likelihood)

# 示例：3个样本，3个类别
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3],
                   [0.1, 0.2, 3.0]])
labels = np.array([0, 1, 2])  # 真实类别

print(f"Softmax概率:\n{softmax(logits)}")
print(f"交叉熵: {cross_entropy(logits, labels):.4f}")

# ==================== PyTorch实现 ====================
# nn.CrossEntropyLoss 内部包含 Softmax，输入应为logits
criterion = nn.CrossEntropyLoss()
logits_t = torch.tensor(logits, requires_grad=True)
labels_t = torch.tensor(labels)
loss = criterion(logits_t, labels_t)
loss.backward()
# 梯度 = softmax(logits) - one_hot(labels)
print(f"梯度:\n{logits_t.grad.numpy()}")
```

**注意**：PyTorch 的 `nn.CrossEntropyLoss` 接收的是**未经过 Softmax 的 logits**，而非概率。它在内部将 Softmax 与交叉熵合并为一个数值稳定的运算，避免单独计算 Softmax 再取对数带来的精度损失。这与 `nn.BCELoss`（接收概率）不同。

**案例：ImageNet 图像分类**。ResNet、VGG 等模型在 1000 类 ImageNet 上训练时，标准损失就是多分类交叉熵。模型输出 1000 维 logits，经过 Softmax 得到每类的概率，CE 衡量预测分布与真实 one-hot 标签的差异。

##### 3.3.2 负对数似然损失（NLL Loss）

**函数定义**：

\[
\text{NLL} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i,c^*})
\]

**公式解释**：
- \(\hat{y}_{i,c^*}\)：模型预测第 \(i\) 个样本属于真实类别 \(c^*\) 的概率。
- 与 CE 形式相同，但 NLL 假设输入已经是**对数概率**（即经过 `log_softmax` 的输出），而非原始概率。
- 在 PyTorch 中，`nn.NLLLoss` 接收 log 概率，`nn.CrossEntropyLoss` 接收 logits。两者本质相同，但数值稳定性处理位置不同。

**代码实现**：

```python
# ==================== PyTorch实现 ====================
# NLLLoss 需要配合 log_softmax 使用
log_probs = torch.nn.functional.log_softmax(logits_t, dim=1)
criterion_nll = nn.NLLLoss()
loss_nll = criterion_nll(log_probs, labels_t)
print(f"NLL Loss: {loss_nll.item():.4f}")

# CrossEntropyLoss 等价于 log_softmax + NLLLoss
criterion_ce = nn.CrossEntropyLoss()
loss_ce = criterion_ce(logits_t, labels_t)
print(f"CE Loss: {loss_ce.item():.4f}")
# 两者数值应相同
```

**说明**：`log_softmax` 先计算 Softmax 再取对数，但通过数学变换直接计算 \(\log(\hat{y}_{i,c}) = z_{i,c} - \log \sum_k e^{z_{i,k}}\)，避免中间概率值过小导致的精度损失。这是 `CrossEntropyLoss` 内部的做法。

##### 3.3.3 加权交叉熵（Weighted Cross-Entropy）

**函数定义**：

\[
\text{WCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} w_c \cdot y_{i,c} \log(\hat{y}_{i,c})
\]

**公式解释**：
- \(w_c\)：第 \(c\) 类的权重。通常设置 \(w_c = \frac{n}{C \cdot n_c}\)，其中 \(n_c\) 是第 \(c\) 类的样本数。
- 样本少的类别获得更大权重，使其对损失的贡献与样本多的类别相当。
- 若不设置权重，样本多的类别会主导损失，模型倾向于预测多数类。

**代码实现**：

```python
# ==================== PyTorch实现 ====================
# 假设3个类别的样本数为[1000, 100, 100]
# 权重与样本数成反比
class_weights = torch.tensor([1.0/1000, 1.0/100, 1.0/100])
# 归一化（可选）
class_weights = class_weights / class_weights.sum() * 3

criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(logits_t, labels_t)
```

**案例：医学图像多分类**。在皮肤病变分类中，常见病变样本多，罕见病变样本少。加权 CE 使模型对罕见类别也保持敏感，提升整体诊断覆盖率。

##### 3.3.4 KL 散度（Kullback-Leibler Divergence）

**函数定义**：

\[
\text{KL}(P \| Q) = \sum_{c=1}^{C} P(c) \log \frac{P(c)}{Q(c)}
\]

**公式解释**：
- \(P\)：真实分布（如 one-hot 标签或软标签）。
- \(Q\)：预测分布（模型输出的概率）。
- \(P(c)\)：真实分布中第 \(c\) 类的概率。
- \(Q(c)\)：预测分布中第 \(c\) 类的概率。
- \(\log \frac{P(c)}{Q(c)}\)：衡量两个分布在类别 \(c\) 上的差异。
- KL 散度衡量用 \(Q\) 近似 \(P\) 时损失的信息量。当 \(P = Q\) 时，KL = 0。
- **注意**：KL 散度不对称，\(\text{KL}(P\|Q) \ne \text{KL}(Q\|P)\)。

**与交叉熵的关系**：

\[
\text{KL}(P \| Q) = \sum_c P(c) \log P(c) - \sum_c P(c) \log Q(c) = -H(P) + \text{CE}(P, Q)
\]

**公式解释**：
- \(H(P) = -\sum_c P(c) \log P(c)\)：真实分布的熵，与模型参数无关，是常数。
- \(\text{CE}(P, Q) = -\sum_c P(c) \log Q(c)\)：交叉熵。
- 因此，**最小化 KL 散度等价于最小化交叉熵**（因为 \(H(P)\) 是常数）。
- 当 \(P\) 是 one-hot 分布时，\(H(P) = 0\)，KL 散度等于交叉熵。

**代码实现**：

```python
# ==================== 从零实现 KL 散度 ====================
def kl_divergence(p, q):
    """KL散度 - 从零实现"""
    epsilon = 1e-15
    # P和Q都是概率分布，形状(n, C)
    q = np.clip(q, epsilon, 1)
    # 逐元素计算 P * log(P/Q)，对类别求和
    return np.mean(np.sum(p * np.log(p / q), axis=1))

# 示例：真实分布和预测分布
P = np.array([[1.0, 0.0, 0.0],   # 样本1真实为类别0
              [0.0, 1.0, 0.0]])  # 样本2真实为类别1
Q = np.array([[0.8, 0.1, 0.1],   # 样本1预测
              [0.2, 0.7, 0.1]])  # 样本2预测

print(f"KL: {kl_divergence(P, Q):.4f}")

# ==================== PyTorch实现 ====================
# KLDivLoss 接收log概率作为输入
criterion = nn.KLDivLoss(reduction='batchmean')
# 输入需为log_softmax，目标为概率分布
loss = criterion(torch.log_softmax(logits_t, dim=1), torch.tensor(P, dtype=torch.float32))
```

**案例：知识蒸馏**。Hinton 提出的知识蒸馏中，教师模型输出"软标签"（如 [0.7, 0.2, 0.1]），学生模型通过最小化与教师输出的 KL 散度来学习。软标签比 one-hot 标签携带更多信息（如类别间的相似性），帮助学生模型获得更好的泛化能力。

##### 3.3.5 标签平滑交叉熵（Label Smoothing CE）

**函数定义**：

\[
\text{LS-CE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} \tilde{y}_{i,c} \log(\hat{y}_{i,c})
\]

其中平滑后的标签为：

\[
\tilde{y}_{i,c} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{C}, & c = c^* \\ \frac{\epsilon}{C}, & c \ne c^* \end{cases}
\]

**公式解释**：
- \(\epsilon\)：平滑参数，通常取 0.1。
- \(c^*\)：真实类别。
- 原本 one-hot 标签中，真实类别为 1，其余为 0。
- 平滑后，真实类别的目标从 1 降为 \(1 - \epsilon + \frac{\epsilon}{C}\)，其余类别从 0 升为 \(\frac{\epsilon}{C}\)。
- 所有类别的目标之和仍为 1：\((1-\epsilon+\frac{\epsilon}{C}) + (C-1)\frac{\epsilon}{C} = 1\)。
- 效果：模型不再被要求将真实类别的概率推向 1、其余推向 0，而是允许一定的不确定性，从而**缓解过拟合、提升泛化**。

**代码实现**：

```python
# ==================== PyTorch实现 ====================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits_t, labels_t)

# ==================== 手动实现 ====================
def label_smoothing_ce(logits, labels, epsilon=0.1):
    """标签平滑交叉熵"""
    n, C = logits.shape
    # 计算log_softmax
    log_probs = torch.log_softmax(logits, dim=1)
    # 构造平滑标签
    smooth_labels = torch.full((n, C), epsilon / C)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon + epsilon / C)
    # 计算交叉熵
    return -(smooth_labels * log_probs).sum(dim=1).mean()
```

**案例：Inception 网络**。Google 的 Inception-v2/v3 论文中首次系统使用标签平滑，在 ImageNet 上将 top-1 错误率降低了约 0.2%。标签平滑现已成为现代图像分类模型的标准技巧。

##### 3.3.6 多分类损失函数选择指南

| 损失函数 | 适用场景 | 关键参数 |
|---------|---------|---------|
| CE | 类别平衡的标准多分类 | 无 |
| NLL | 配合 log_softmax 使用 | 无 |
| Weighted CE | 类别不平衡 | class_weights |
| KL 散度 | 知识蒸馏、分布匹配 | 无 |
| Label Smoothing CE | 缓解过拟合、提升泛化 | ε |


#### 三·四 三类任务损失函数总览

| 任务类型 | 损失函数 | 输出层激活 | 公式核心 |
|---------|---------|-----------|---------|
| 回归 | MSE | 恒等映射 | \(\frac{1}{n}\sum(y-\hat{y})^2\) |
| 回归 | MAE | 恒等映射 | \(\frac{1}{n}\sum\|y-\hat{y}\|\) |
| 回归 | Huber | 恒等映射 | 分段：小误差平方，大误差线性 |
| 回归 | Quantile | 恒等映射 | 分位数加权绝对误差 |
| 二分类 | BCE | Sigmoid | \(-\frac{1}{n}\sum[y\log\hat{y}+(1-y)\log(1-\hat{y})]\) |
| 二分类 | Weighted BCE | Sigmoid | BCE + 类别权重 |
| 二分类 | Focal | Sigmoid | BCE × \((1-\hat{p})^\gamma\) |
| 二分类 | Hinge | 恒等映射 | \(\max(0, 1-y\hat{y})\) |
| 多分类 | CE | Softmax | \(-\frac{1}{n}\sum\sum y\log\hat{y}\) |
| 多分类 | NLL | LogSoftmax | \(-\frac{1}{n}\sum\log\hat{y}_{c^*}\) |
| 多分类 | Weighted CE | Softmax | CE + 类别权重 |
| 多分类 | KL 散度 | Softmax | \(\sum P\log\frac{P}{Q}\) |
| 多分类 | Label Smoothing CE | Softmax | CE + 平滑标签 |


#### 三·五 损失函数选择的实践建议

1. **回归任务**：默认使用 MSE。若数据含异常值，改用 Huber 或 MAE。若需区间预测，使用 Quantile Loss。

2. **二分类任务**：默认使用 BCE（PyTorch 中推荐 `BCEWithLogitsLoss`，数值更稳定）。若类别不平衡，使用 Weighted BCE 或 Focal Loss。若任务需要间隔（如人脸验证），使用 Hinge Loss。

3. **多分类任务**：默认使用 CE（PyTorch 中 `CrossEntropyLoss`，输入 logits）。若类别不平衡，使用 Weighted CE。若需提升泛化，加入 Label Smoothing。若做知识蒸馏，使用 KL 散度。

4. **数值稳定性**：始终优先使用 PyTorch 内置损失函数，它们内部做了数值稳定处理（如 `log_softmax`、减去最大值等）。手动实现时需注意 `log(0)` 和 `exp` 溢出问题。

5. **损失函数与激活函数的搭配**：分类任务的输出层激活函数与损失函数必须匹配。Sigmoid 配 BCE，Softmax 配 CE。若使用 PyTorch 的 `CrossEntropyLoss` 或 `BCEWithLogitsLoss`，则输出层不要加激活函数，因为损失内部已包含。

**核心结论**：损失函数是模型学习的"指南针"，它定义了什么是"好"的预测。选择合适的损失函数，需要理解任务的数据分布、噪声特性和类别平衡情况。理解每种损失函数的概率背景（最大似然估计）和梯度行为，比死记公式更重要——它让你在面对新任务时，能够自行设计或组合合适的损失函数。

## 四、反向传播与梯度下降


###  梯度下降算法回顾

**梯度下降法**简单来说就是一种**寻找使损失函数最小化的方法**。

从数学角度来看，**梯度的方向是函数增长速度最快的方向，那么梯度的反方向就是函数减少最快的方向**，所以有：

{% asset_img 12.png "Hexo 博客封面示例" %}

其中，η是学习率，如果学习率**太小**，那么每次训练之后得到的效果都太小，**增大训练的时间成本**。如果，学习率**太大**，那就有可能**直接跳过最优解**，进入无限的训练中。解决的方法就是，学习率也需要随着训练的进行而变化。

{% asset_img 13.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-08_08-16-44.png"Hexo 博客封面示例" %}

在上图中我们展示了一维和多维的损失函数，损失函数呈碗状。在训练过程中损失函数对权重的偏导数就是损失函数在该位置点的梯度。我们可以看到，沿着负梯度方向移动，就可以到达损失函数底部，从而使损失函数最小化。这种利用损失函数的梯度迭代地寻找最小值的过程就是梯度下降的过程。



**在进行模型训练时，有三个基础的概念：**

1. **Epoch**: 使用全部数据对模型进行以此完整训练，训练次数
2. **Batch**: 使用训练集中的小部分样本对模型权重进行以此反向传播的参数更新，每次训练每批次样本数量
3. **Iteration**: 使用一个 Batch 数据对模型进行一次参数更新的过程，每次训练批次数

假设数据集有 50000 个训练样本，现在选择 Batch Size = 256 对模型进行训练。
每个 Epoch 要训练的图片数量：50000 
训练集具有的 Batch 个数：50000/256+1=196
每个 Epoch 具有的 Iteration 个数：196 
10个 Epoch 具有的 Iteration 个数：1960

**公式**：(训练样本数 + batch_size - 1)/2 = 训练集具有的 Batch 个数

例子：
{% asset_img Snipaste_2026-10-08_08-00-33.png "Hexo 博客封面示例" %}


在深度学习中，**梯度下降的几种方式的根本区别就在于Batch Size不同**,如下表所示：
{% asset_img Snipaste_2026-10-08_08-00-33.png "Hexo 博客封面示例" %}


==注：上表中 Mini-Batch 的 Batch 个数为 N / B + 1 是针对未整除的情况。整除则是 N / B。==


### 反向传播（BP算法）

前向传播：指的是数据输入到神经网络中，逐层向前传输，一直运算到输出层为止。
反向传播（Back Propagation）：利用损失函数 ERROR值，从后往前，结合梯度下降算法，依次求各个参数的偏导，并进行参数更新
{% asset_img Snipaste_2026-10-08_08-25-45.png "Hexo 博客封面示例" %}

> 利用**反向传播算法**对神经网络进行训练。该方法与**梯度下降算法**相结合，对网络中所有权重**计算损失函数的梯度**，并利用梯度值来**更新权值**以最小化损失函数。

####  反向传播概念

**前向传播**：指的是数据输入到神经网络中，逐层向前传输，一直运算到输出层为止。

**反向传播（Back Propagation）**：利用损失函数ERROR值，从后往前，结合梯度下降算法，依次求各个参数的偏导，并进行参数更新。


{% asset_img 14.png "Hexo 博客封面示例" %}


在网络的训练过程中经过前向传播后得到的最终结果跟训练样本的真实值总是存在一定误差，这个误差便是损失函数 ERROR。想要减小这个误差，**就用损失函数 ERROR，从后往前，依次求各个参数的偏导，这就是反向传播（Back Propagation）**。

####  反向传播详解

**反向传播算法利用链式法则对神经网络中的各个节点的权重进行更新**。

【举个栗子🌰：】

如下图是一个简单的神经网络用来举例：**激活函数为sigmoid**

{% asset_img image-20210129103846076.png "Hexo 博客封面示例" %}

**前向传播运算**：

{% asset_img image-20210129104612230.png "Hexo 博客封面示例" %}

接下来是**反向传播**（求网络误差对各个权重参数的梯度）：

我们先来求最简单的，求误差E对w5的导数。首先明确这是一个**链式法则**的求导过程，要求误差E对w5的导数，需要先求误差E对$$out_{o1}$$的导数，再求$$out_{o1}$$对$$net_{o1}$$的导数，最后再求$$net_{o1}$$对$$w_5$$的导数，经过这个**链式法则**，我们就可以求出误差E对$$w_5$$的导数（偏导），如下图所示：

{% asset_img image-20210129104646474.png "Hexo 博客封面示例" %}

导数（梯度）已经计算出来了，下面就是**反向传播与参数更新过程**：
{% asset_img image-20210129104718140.png "Hexo 博客封面示例" %}

如果要想求**误差E对w1的导数**，误差E对w1的求导路径不止一条，这会稍微复杂一点，但换汤不换药，计算过程如下所示：

{% asset_img image-20210129104756053.png "Hexo 博客封面示例" %}

### 梯度下降优化方法

梯度下降优化算法中，可能会碰到以下情况：

- 碰到平缓区域，梯度值较小，参数优化变慢
- 碰到 “鞍点” ，梯度为0，参数无法优化
- 碰到局部最小值，参数不是最优

对于这些问题, 出现了一些对梯度下降算法的优化方法，例如：**Momentum**、**AdaGrad**、**RMSprop**、**Adam** 等

{% asset_img Snipaste_2026-10-08_08-49-57.png "Hexo 博客封面示例" %}

####  指数加权平均

我们最常见的算数平均指的是将所有数加起来除以数的个数，每个数的权重是相同的。指数加权平均指的是给每个数赋予不同的权重求得平均数。移动平均数，指的是计算最近邻的 N 个数来获得平均数。

**指数移动加权平均**则是参考各数值，并且各数值的权重都不同，距离越远的数字对平均数计算的贡献就越小（权重较小），距离越近则对平均数的计算贡献就越大（权重越大）。

比如：明天气温怎么样，和昨天气温有很大关系，而和一个月前的气温关系就小一些。

计算公式可以用下面的式子来表示：

{% asset_img Snipaste_2026-10-08_09-08-25.png "Hexo 博客封面示例" %}

- $S_t$ 表示指数加权平均值;
- $Y_t$ 表示t时刻的值;
- β 调节权重系数，该值越大平均数越平缓。因为β越大，意味着当前梯度影响越小，越看重指数加权

**第100天的指数加权平均值为:**

{% asset_img 1735177998337.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-08_09-14-08.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-08_09-17-39.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-08_09-53-02.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-08_09-55-11.png "Hexo 博客封面示例" %}

**注意**
1. 根据上面的迭代公式可知0.9*指数加权平均 + 0.1*本次的梯度，即使本次梯度为0，权重也不会为0，避免了上面所说的鞍点

笔记代码：

**下面通过代码来看结果，随机产生 30 天的气温数据：**

```python
import torch
import matplotlib.pyplot as plt


ELEMENT_NUMBER = 30


# 1. 实际平均温度
def test01():

    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def test02(beta=0.9):
    
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    print(temperature)

    exp_weight_avg = []
    # idx从1开始
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        # idx-2：2-2=0，exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()


if __name__ == '__main__':
    test01()
    test02(0.5)
    test02(0.9)
```
{% asset_img 50.png "Hexo 博客封面示例" %}

课堂代码：
```python
"""
案例:
    演示近30天, 天气分布情况.

结论:
    针对于 β(调节权重系数)来讲, 其值越大说明: 越依赖指数加权平均, 越不依赖本地的梯度值, 数据就越: 平缓.
"""


import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30


# 1. 实际平均温度
def dm01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def dm02(beta=0.9):
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    exp_weight_avg = []
    # idx从1开始
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        # idx-2：2-2=0，exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()


if __name__ == '__main__':
    dm01()      # 不考虑 权重系数, 每个值的权重都一致.
    dm02(0.5)   # 考虑 权重系数, 值越小, 数据越陡.
    dm02(0.9)   # 考虑 权重系数, 值越大, 数据越平缓 0.9
```

{% asset_img 38.png "Hexo 博客封面示例" %}
{% asset_img 39.png "Hexo 博客封面示例" %}
{% asset_img 40.png "Hexo 博客封面示例" %}

**从程序运行结果可以看到：**

- 指数加权平均绘制出的气温变化曲线更加==平缓==
- β 的值越大，则绘制出的折线==越加平缓，波动越小==(1-β越小,t时刻的$S_t$越不依赖$Y_t$的值)
- β 值一般默认都是 0.9

####  动量算法Momentum

当梯度下降碰到 “峡谷” 、”平缓”、”鞍点” 区域时, 参数更新速度变慢。 Momentum 通过**指数加权平均法**，累计历史梯度值，进行参数更新，越近的梯度值对当前参数更新的重要性越大。

**梯度计算公式**：

​				$$s_t=βs_{t−1}+(1−β)g_t$$

**参数更新公式**：

​				$$w_t=w_{t−1}−ηs_t$$

$s_t$是当前时刻指数加权平均梯度值

$s_{t-1}$是历史指数加权平均梯度值

$g_t$是当前时刻的梯度值

β 是调节权重系数，通常取 0.9 或 0.99

η是学习率

$w_t$是当前时刻模型权重参数


```python
咱们举个例子，假设：权重 β 为 0.9，例如：
第一次梯度值：s1 = g1 = w1 
第二次梯度值：s2 = 0.9*s1 + g2*0.1 
第三次梯度值：s3 = 0.9*s2 + g3*0.1 
第四次梯度值：s4 = 0.9*s3 + g4*0.1 
1. w 表示初始梯度
2. g 表示当前轮数计算出的梯度值
3. s 表示历史梯度移动加权平均值

梯度下降公式中梯度的计算，就不再是当前时刻t的梯度值，而是历史梯度值的指数移动加权平均值。
公式修改为：
Wt = Wt-1 - η*St
Wt：当前时刻模型权重参数
St：当前时刻指数加权平均梯度值
η：学习率
```

Monmentum 优化方法是如何一定程度上克服 “平缓”、”鞍点”、”峡谷” 的问题呢？

{% asset_img Snipaste_2026-10-08_10-35-42.png "Hexo 博客封面示例" %}



- 当处于鞍点位置时，由于当前的梯度为 0，参数无法更新。但是 Momentum 动量梯度下降算法已经在先前积累了一些梯度值，很有可能使得跨过鞍点。
- 由于 mini-batch 普通的梯度下降算法，每次选取少数的样本梯度确定前进方向，可能会出现震荡，使得训练时间变长。Momentum 使用移动加权平均，平滑了梯度的变化，使得前进方向更加平缓，有利于加快训练过程。一定程度上有利于降低 “峡谷” 问题的影响。

笔记代码：

在pytorch中动量梯度优化法编程实践如下：

```python
def test01():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # sum()转换为标量张量，才可以自动微分
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：SGD 指定参数beta=0.9
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

**结果显示：**

```python
第1次: 梯度w.grad: 1.000000, 更新后的权重:0.990000
第2次: 梯度w.grad: 0.990000, 更新后的权重:0.971100
```

课堂代码：
```python
"""
案例:
    演示 梯度下降优化方法.

梯度下降相关介绍:
    概述:
        梯度下降是结合 本次损失函数的导数(作为梯度) 基于学习率 来更新权重的.
    公式:
        W新 = W旧 - 学习率 * (本次的)梯度
    存在的问题:
        1. 遇到平缓区域, 梯度下降(权重更新)可能会慢.
        2. 可能会遇到 鞍点(梯度为0)
        3. 可能会遇到 局部最小值.
    解决思路:
        从上述的 学习率 或者 梯度入手, 进行优化, 于是有了: 动量法Momentum, 自适应学习率AdaGrad, RMSProp, 综合衡量: Adam

    动量法Momentum:
        动量法公式:
            St = β * St-1 + (1 - β) * Gt
        解释:
            St:     本次的指数移动加权平均结果.
            β:      调节权重系数, 越大, 数据越平缓, 历史指数移动加权平均 比重越大, 本次梯度权重越小.
            St-1:   历史的指数移动加权平均结果.
            Gt:     本次计算出的梯度(不考虑历史梯度).

        加入动量法后的 梯度更新公式:
            W新 = W旧 - 学习率 * St

    自适应学习率: AdaGrad(Adaptive Gradient Estimation)
        公式:
            累计平方梯度:
                St = St-1 + Gt * Gt
                解释:
                    St:     累计平方梯度
                    St-1:   历史累计平方梯度.
                    Gt:     本次的梯度.
            学习率:
                学习率 = 学习率 / (sqrt(St) + 小常数)
                解释:
                    小常数: 1e-10, 目的: 防止分母变为0
            梯度下降公式:
                W新 = W旧 - 调整后的学习率 * Gt
        缺点:
            可能会导致学习率过早, 过量的降低, 导致模型后期学习率太小, 较难找到最优解.


    自适应学习率: RMSProp(Root Mean Square Propagation) -> 可以看做是 对AdaGrad做的优化, 加入 调和权重系数.
        公式:
            指数加权平均 累计历史平方梯度:
                St = β * St-1 +  (1 - β) * Gt * Gt
                解释:
                    St:     累计平方梯度
                    St-1:   历史累计平方梯度.
                    Gt:     本次的梯度.
                    β:      调和权重系数.
            学习率:
                学习率 = 学习率 / (sqrt(St) + 小常数)
                解释:
                    小常数: 1e-10, 目的: 防止分母变为0
            梯度下降公式:
                W新 = W旧 - 调整后的学习率 * Gt
        优点:
           RMSProp通过引入 衰减系数β, 控制历史梯度 对 历史梯度信息获取的多少.

    自适应矩估计: Adam(Adaptive Moment Estimation)
        思路:
            即优化学习率, 又优化梯度.
        公式:
            一阶矩: 算均值.
                Mt = β1 * Mt-1 + (1 - β1) * Gt          充当: 梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt     充当: 学习率
            二阶矩: 梯度的方差.
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β2 ^ t)
            权重更新公式:
                W新 = W旧 - 学习率 / (sqrt(St^) + 小常数)  *  Mt^
        大白话翻译:
            Adam = RMSProp + Momentum

总结: 如何选择梯度下降优化方法
    简单任务和较小的模型:
        SGD, 动量法
    复杂任务或者有大量数据:
        Adam
    需要处理稀疏数据或者文本数据:
        AdaGrad, RMSProp
"""

# 导包
import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定义函数, 演示: 梯度下降优化方法 -> 动量法(Momentum)
def dm01_momentum():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器(函数对象) -> 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: 当momentum=0(默认), 只考虑: 本次梯度.

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')



# 2. 定义函数, 演示: 梯度下降优化方法 -> 自适应学习率(AdaGrad)
def dm02_adagrad():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.

    # 思路2: 基于AdaGrad(自适应学习率).
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')

# 3. 定义函数, 演示: 梯度下降优化方法 -> 自适应学习率(RMSProp)
def dm03_rmsprop():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.

    # 思路2: 基于AdaGrad(自适应学习率).
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3: 基于RMSProp(自适应学习率).
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')


# 4. 定义函数, 演示: 梯度下降优化方法 -> 自适应矩估计(Adam)
def dm04_adam():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.

    # 思路2: 基于AdaGrad(自适应学习率).
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3: 基于RMSProp(自适应学习率).
    # optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 思路4: 基于Adam(自适应矩估计).
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999)) # betas=(梯度用的 衰减系数, 学习率用的 衰减系数)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')



# 5. 测试
if __name__ == '__main__':
    dm01_momentum()
    # dm02_adagrad()
    # dm03_rmsprop()
    # dm04_adam()


```

#### AdaGrad

AdaGrad 通过对不同的参数分量使用不同的学习率，**AdaGrad 的学习率总体会逐渐减小**，这是因为 AdaGrad 认为：在起初时，我们距离最优目标仍较远，可以使用较大的学习率，加快训练速度，随着迭代次数的增加，学习率逐渐下降。

其计算步骤如下：

1. 初始化学习率 η、初始化参数w、小常数 σ = 1e-10（防止下面的公式分母为0）

2. 初始化梯度累计变量 s = 0
3. 从训练集中采样 m 个样本的小批量，计算梯度$g_t​$
4. **累积平方梯度**: $s_t​$ = $s_{t-1}​$ + $g_t​$ ⊙ $g_t​$，⊙ 表示各个分量相乘（点乘）

5. 学习率 η 的计算公式如下：

      ​			η = $$η\over\sqrt{s_t}+σ$$

6. 权重参数更新公式如下：

      ​			$w_t$ = $$w_{t-1}$$ - $$η\over\sqrt{s_t}+σ$$ * $g_t$

7. 重复 3-7 步骤




**AdaGrad 缺点是可能会使得学习率过早、过量的降低，导致模型训练后期学习率太小，较难找到最优解。**


{% asset_img Snipaste_2026-10-08_09-55-11.png "Hexo 博客封面示例" %}


在PyTorch中AdaGrad优化法编程实践如下：

```python
def test02():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：adagrad优化方法
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

**结果显示：**

```python
第1次: 梯度w.grad: 1.000000, 更新后的权重:0.990000
第2次: 梯度w.grad: 0.990000, 更新后的权重:0.982965
```

课堂代码
```python
"""
自适应学习率: AdaGrad(Adaptive Gradient Estimation)
        公式:
            累计平方梯度:
                St = St-1 + Gt * Gt
                解释:
                    St:     累计平方梯度
                    St-1:   历史累计平方梯度.
                    Gt:     本次的梯度.
            学习率:
                学习率η = 学习率η / (sqrt(St) + 小常数)
                解释:
                    小常数: 1e-10, 目的: 防止分母变为0
            梯度下降公式:
                W新 = W旧 - 调整后的学习率 * Gt
        缺点:
            可能会导致学习率过早, 过量的降低, 导致模型后期学习率太小, 较难找到最优解.
"""
# 2. 定义函数, 演示: 梯度下降优化方法 -> 自适应学习率(AdaGrad)
def dm02_adagrad():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.

    # 思路2: 基于AdaGrad(自适应学习率).
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')
```

####  RMSProp

**RMSProp 优化算法是对 AdaGrad 的优化**。最主要的不同是，其使用**指数加权平均梯度**替换历史梯度的平方和。β使用作为调和系数，调节历史梯度，（1-β）调节当前梯度，如此梯度不至于都相同

有趣的是，RMSProp在AdaGrad的基础上加上β，动量法Momentum在SGD（普通的权重参数更新公式）的基础上加β，他们的图像与β特性都差不多

其计算过程如下：

1. 初始化学习率 η、初始化权重参数w、小常数 σ = 1e-10

2. 初始化梯度累计变量 s = 0

3. 从训练集中采样 m 个样本的小批量，计算梯度 $g_t$

4. 使用指数加权平均累计历史梯度，⊙ 表示各个分量相乘，公式如下：

      ​			$s_t$ = β$s_{t-1}$ + (1-β)$g_t$⊙$g_t$

5. 学习率 η 的计算公式如下：

      ​			η = $η\over\sqrt{s_t}+σ​$

6. 权重参数更新公式如下：

      ​			$w_t$ = $w_{t-1}$ - $η\over\sqrt{s_t}+σ$ * $g_t$

7. 重复 3-7 步骤

RMSProp 与 AdaGrad 最大的区别是对梯度的累积方式不同，对于每个梯度分量仍然使用不同的学习率。

**RMSProp 通过引入衰减系数β，控制历史梯度对历史梯度信息获取的多少**. 被证明在神经网络非凸条件下的优化更好，学习率衰减更加合理一些。

需要注意的是：AdaGrad 和 RMSProp 都是对于不同的参数分量使用不同的学习率，如果某个参数分量的梯度值较大，则对应的学习率就会较小，如果某个参数分量的梯度较小，则对应的学习率就会较大一些。

课堂代码：
```python
"""
    自适应学习率: RMSProp(Root Mean Square Propagation) -> 可以看做是 对AdaGrad做的优化, 加入 调和权重系数.
        公式:
            指数加权平均 累计历史平方梯度:
                St = β * St-1 +  (1 - β) * Gt * Gt
                解释:
                    St:     累计平方梯度
                    St-1:   历史累计平方梯度.
                    Gt:     本次的梯度.
                    β:      调和权重系数.
            学习率:
                学习率 = 学习率 / (sqrt(St) + 小常数)
                解释:
                    小常数: 1e-10, 目的: 防止分母变为0
            梯度下降公式:
                W新 = W旧 - 调整后的学习率 * Gt
        优点:
           RMSProp通过引入 衰减(调节)系数β, 控制历史梯度 对 历史梯度信息获取的多少.
"""         
# 3. 定义函数, 演示: 梯度下降优化方法 -> 自适应学习率(RMSProp)
def dm03_rmsprop():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.

    # 思路2: 基于AdaGrad(自适应学习率).
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3: 基于RMSProp(自适应学习率).
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')
```

RMSPrp方法参数：
{% asset_img Snipaste_2026-10-08_11-39-34.png "Hexo 博客封面示例" %}

在PyTorch中RMSprop梯度优化法，编程实践如下：

```python
def test03():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：RMSprop算法，其中alpha对应beta
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

**结果显示：**

```python
第1次: 梯度w.grad: 1.000000, 更新后的权重:0.968377
第2次: 梯度w.grad: 0.968377, 更新后的权重:0.945788
```


####  Adam

- Momentum 使用指数加权平均计算当前的梯度值

- AdaGrad、RMSProp 使用自适应的学习率

- Adam优化算法（Adaptive Moment Estimation，自适应矩估计）将 Momentum 和 RMSProp 算法结合在一起
  - 修正梯度: 使⽤梯度的指数加权平均 
  - 修正学习率: 使⽤梯度平⽅的指数加权平均

- **原理**：Adam 是结合了 **Momentum** 和 **RMSProp** 优化算法的优点的自适应学习率算法。它计算了梯度的一阶矩（平均值）和二阶矩（梯度的方差）的自适应估计，从而动态调整学习率。

根据公式要传2个β，分别用于梯度和学习率

- **梯度计算公式**：

  ​		$$m_t=β_1m_{t−1}+(1−β_1)g_t$$

  ​		$$s_t=β_2s_{t−1}+(1−β_2)gt^2​$$

  ​		$$\hat{m_t}​$$ = $$m_t\over1−β_1^t​$$, $$\hat{s_t}​$$=$$s_t\over1−β_2^t​$$

- **权重参数更新公式**:

     ​		$$w_t​$$ = $$w_{t−1}​$$ − $$η\over\sqrt{\hat{s_t}}+ϵ​$$$\hat{m_t}​$

其中，$$m_t$$ 是梯度的一阶矩估计，$$s_t$$ 是梯度的二阶矩估计，$$ \hat{m_t}$$和 $$\hat{s_t}$$ 是偏差校正后的估计。


课堂代码：
```python
"""    自适应矩估计: Adam(Adaptive Moment Estimation)
        思路:
            即优化学习率, 又优化梯度.
        公式:
            一阶矩: 算均值.
                Mt = β1 * Mt-1 + (1 - β1) * Gt          充当: 梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt     充当: 学习率
            二阶矩: 梯度的方差.
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β2 ^ t)
            权重参数更新公式:
                W新 = W旧 - 学习率 / (sqrt(St^) + 小常数)  *  Mt^
        大白话翻译:
            Adam = RMSProp + Momentum
"""
def dm04_adam():
    # 1. 初始化权重参数.
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.,又monmentum参数则是动量方法，没有就是普通的参数更新（SGD）

    # 思路2: 基于AdaGrad(自适应学习率).
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3: 基于RMSProp(自适应学习率).
    # optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 思路4: 基于Adam(自适应矩估计).
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999)) # betas=(梯度用的 衰减系数, 学习率用的 衰减系数)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5.重复上述的步骤, 第2次 更新权重参数.
    # 5.1 定义损失函数.
    criterion = ((w ** 2) / 2.0)
    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 5.3 打印结果.
    print(f'w: {w}, w.grad: {w.grad}')
```


在PyTroch中，Adam梯度优化法编程实践如下：

```python
def test04():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True)
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：Adam算法，其中betas(β）是指数加权的系数
    optimizer = torch.optim.Adam([w], lr=0.01, betas=[0.9, 0.99])
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

**结果显示：**

```python
第1次: 梯度w.grad: 1.000000, 更新后的权重:0.990000 
第2次: 梯度w.grad: 0.990000, 更新后的权重:0.980003
```

####  小结

| **优化算法** | **优点**                                            | **缺点**                                               | **适用场景**                                         |
| ------------ | --------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| **SGD**      | 简单、容易实现。                                    | 收敛速度较慢，容易震荡，特别是在复杂问题中。           | 用于简单任务，或者当数据特征分布相对稳定时。         |
| **Momentum** | 可以加速收敛，减少震荡，特别是在高曲率区域。        | 需要手动调整动量超参数，可能会在小步长训练中过度更新。 | 用于非平稳优化问题，尤其是深度学习中的应用。         |
| **AdaGrad**  | 自适应调整学习率，适用于稀疏数据。                  | 学习率会在训练过程中逐渐衰减，可能导致早期停滞。       | 适合稀疏数据，如 NLP 或推荐系统中的特征。            |
| **RMSProp**  | 解决了 AdaGrad 学习率过早衰减的问题，适应性强。     | 需要选择合适的超参数，更新可能会过于激进。             | 适用于动态问题、非平稳目标函数，如深度学习训练。     |
| **Adam**     | 结合了 Momentum 和 RMSProp 的优点，适应性强且稳定。 | 需要调节更多的超参数，训练过程中可能会产生较大波动。   | 广泛适用于各种深度学习任务，特别是非平稳和复杂问题。 |

- **简单任务和较小的模型**：SGD 或 Momentum

- **复杂任务或有大量数据**：Adam 是最常用的选择，因其在大部分任务上都表现优秀

- **需要处理稀疏数据或文本数据**：Adagrad 或 RMSProp

## 学习率衰减优化方法

### 为什么要进行学习率优化

在训练神经网络时，一般情况下学习率都会随着训练而变化。这主要是由于，在神经网络训练的后期，如果**学习率过高，会造成loss的振荡**，但是如果**学习率减小的过慢，又会造成收敛变慢**的情况。

运行下面代码，观察学习率设置不同对网络训练的影响：

```python
import torch
import matplotlib.pyplot as plt

# x看成是权重，y看成是loss，下面通过代码来理解学习率的作用
def func(x_t):
    return torch.pow(2*x_t, 2)  # y = 4 x ^2

# 采用较小的学习率，梯度下降的速度慢
# 采用较大的学习率，梯度下降太快越过了最小值点，导致不收敛，甚至震荡
def dm01():

    x = torch.tensor([2.], requires_grad=True)
    # 记录loss迭代次数，画曲线
    iter_rec, loss_rec, x_rec = list(), list(), list()

    # 实验学习率： 0.01 0.02 0.03 0.1 0.2 0.3 0.4
    # lr = 0.1    # 正常的梯度下降
    # lr = 0.125      # 当学习率设置0.125 一下子求出一个最优解
                    # x=0 y=0 在x=0处梯度等于0 x的值x=x-lr*x.grad就不用更新了
                    # 后续再多少次迭代 都固定在最优点

    lr = 0.2      # x从2.0一下子跨过0点，到了左侧负数区域
    # lr = 0.3      # 梯度越来越大 梯度爆炸
    max_iteration = 4
    for i in range(max_iteration):
        y = func(x)   # 得出loss值
        y.backward()  # 计算x的梯度
        print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
            i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))
        x_rec.append(x.item())      # 梯度下降点 列表
        # 更新参数
        x.data.sub_(lr * x.grad)    # x = x - x.grad
        x.grad.zero_()
        iter_rec.append(i)          # 迭代次数 列表
        loss_rec.append(y.item())  # 损失值 列表，这里将y改为y.item()以获取标量值
    # 迭代次数-损失值 关系图
    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.grid()
    plt.xlabel("Iteration X")
    plt.ylabel("Loss value Y")
    # 函数曲线-下降轨迹 显示图
    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.detach().numpy(), y.detach().numpy(), label="y = 4*x^2")
    y_rec = [func(torch.tensor(i)).item() for i in x_rec]
    print('x_rec--->', x_rec)
    print('y_rec--->', y_rec)
    # 指定线的颜色和样式（-ro：红色圆圈，b-：蓝色实线等）
    plt.subplot(122).plot(x_rec, y_rec, '-ro')
    plt.grid()
    plt.legend()
    plt.show()

dm01()
```

**运行效果图如下：**

可以看出：采用较小的学习率，梯度下降的速度慢；采用较大的学习率，梯度下降太快越过了最小值点，导致震荡，甚至不收敛（梯度爆炸）。

{% asset_img 04-01.png "Hexo 博客封面示例" %}

课堂代码：
```python
"""
案例:
    通过代码演示学习率 对 梯度(权重优化)的影响.

结论:
    学习率越小, 梯度下降越慢.
    学习率越大, 梯度下降越快, 可能会越过最小值, 造成震荡, 甚至不收敛(梯度爆炸)
"""

import torch
import matplotlib.pyplot as plt

# x看成是权重，y看成是loss，下面通过代码来理解学习率的作用
def func(x_t):
    return torch.pow(2*x_t, 2)  # y = 4 x ^2

# 采用较小的学习率，梯度下降的速度慢
# 采用较大的学习率，梯度下降太快越过了最小值点，导致不收敛，甚至震荡
def dm01():

    x = torch.tensor([2.], requires_grad=True)
    # 记录loss迭代次数，画曲线
    iter_rec, loss_rec, x_rec = list(), list(), list()

    # 实验学习率： 0.01 0.02 0.03 0.1 0.2 0.3 0.4
    # lr = 0.01     # 较小的学习率,  会导致梯度下降速度慢
    # lr = 0.1    # 正常的梯度下降
    # lr = 0.125      # 当学习率设置0.125 一下子求出一个最优解
                    # x=0 y=0 在x=0处梯度等于0 x的值x=x-lr*x.grad就不用更新了
                    # 后续再多少次迭代 都固定在最优点

    # lr = 0.2      # x从2.0一下子跨过0点，到了左侧负数区域
    lr = 0.3      # 梯度越来越大 梯度爆炸
    max_iteration = 4
    for i in range(max_iteration):
        y = func(x)   # 得出loss值
        y.backward()  # 计算x的梯度
        print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
            i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))
        x_rec.append(x.item())      # 梯度下降点 列表
        # 更新参数
        x.data.sub_(lr * x.grad)    # x = x - x.grad
        x.grad.zero_()
        iter_rec.append(i)          # 迭代次数 列表
        loss_rec.append(y.item())  # 损失值 列表，这里将y改为y.item()以获取标量值
    # 迭代次数-损失值 关系图
    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.grid()
    plt.xlabel("Iteration X")
    plt.ylabel("Loss value Y")
    # 函数曲线-下降轨迹 显示图
    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.detach().numpy(), y.detach().numpy(), label="y = 4*x^2")
    y_rec = [func(torch.tensor(i)).item() for i in x_rec]
    print('x_rec--->', x_rec)
    print('y_rec--->', y_rec)
    # 指定线的颜色和样式（-ro：红色圆圈，b-：蓝色实线等）
    plt.subplot(122).plot(x_rec, y_rec, '-ro')
    plt.grid()
    plt.legend()
    plt.show()

dm01()
```


###  等间隔学习率衰减

等间隔学习率衰减方式如下所示：

{% asset_img 04-02.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-08_15-47-02.png "Hexo 博客封面示例" %}

在PyTorch中实现时使用：

```python
#   step_size：调整间隔数=50
#   gamma：调整系数=0.5
#   调整方式：lr = lr * gamma
optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
```

具体使用方式如下：

```python
import torch
from torch import optim
import matplotlib.pyplot as plt


def test_StepLR():
    # 0.参数初始化
    LR = 0.1  # 设置学习率初始化值为0.1
    iteration = 10
    max_epoch = 200
    # 1 初始化参数
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)

    # 2.优化器
    optimizer = optim.SGD([w], lr=LR, momentum=0.9)

    # 3.设置学习率下降策略，50轮换一次学习率，每次换时学习率降为原来的0.5倍
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    # 4.获取学习率的值和当前的epoch
    lr_list, epoch_list = [], []
    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_last_lr()) # 获取当前lr
        epoch_list.append(epoch) # 获取当前的epoch
        for i in range(iteration):  # 遍历每一个batch数据
            loss = (w*x-y_true)**2  # 目标函数
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
        # 更新下一个epoch的学习率
        scheduler_lr.step()
    # 5.绘制学习率变化的曲线
    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()	
```

课堂代码：
```python
"""
案例:
    演示学习率衰减策略.

学习率衰减策略介绍:
    目的:
        较之于AdaGrad, RMSProp, Adam方式, 我们可以通过 等间隔, 指定间隔, 指数等方式, 来手动控制学习率的调整.

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

等间隔学习率衰减:
    step_size: 间隔的轮数, 即: 多少轮调整一次学习率.
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma

指定间隔学习率衰减:
    milestones = [50, 125, 160]     里边定义的是要调整学习率的 轮数.
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma

指数间隔学习率衰减:
    前期学习率衰减快, 中期慢, 后期更慢, 更符合梯度下降规律.
    公式:
        lr新 = lr旧 * gamma ** epoch

总结:
    等间隔学习率衰减:
        优点:
            直观, 易于调试, 适用于 大批量数据.
        缺点:
            学习率变化较大, 可能跳过最优解.
        应用场景:
            大型数据集, 较为简单的任务.

    指定间学习率衰减:
        优点:
            易于调试, 稳定训练过程.
        缺点:
            在某些情况下可能衰减过快, 导致优化提前停滞.
        应用场景:
            对训练平稳性要求较高的任务.
   指数学习率衰减:
        优点:
            平滑, 且考虑历史更新, 收敛稳定性较强.
        缺点:
            超参调节较为复杂, 可能需要更多的资源.
        应用场景:
            高精度训练, 避免过快收敛.

"""

# 导包
import torch
from torch import optim
import matplotlib.pyplot as plt


# 1. 定义函数, 演示: 等间隔学习率衰减
def dm01():
    # 1. 定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数.
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集.  y_true, x, w
    # 真实值.
    y_true = torch.tensor([0])

    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)

    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象, 动量法 -> 加速模型的收敛, 减少震荡.
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建学习率衰减对象.
    # 思路1: 创建等间隔学习率衰减对象.
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮调整一次学习率), 参3: 学习率衰减系数.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.1, 0.1... 0.05...]

    # 5. 创建两个列表, 分别表示: 训练轮数, 每轮训练用的学习率
    # epoch_list = [0, 1, 2, 3.... 50, 51, 52...100, 101, 101... 150, 151...199]
    # lr_list =    [0.1, 0.1, 0.1, 0.05........,0.025.........,  0.0125...]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数, 进行具体的训练.
    for epoch in range(epochs):     # epoch: 0 ~ 199
        # 7. 获取当前轮数 和 学习率, 并保存到列表中.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())     # 获取最后的lr(learning rate, 学习率)

        # 8. 循环遍历, 每轮每批次进行训练.
        for batch in range(iteration):

            # 9. 先计算预测值, 然后基于损失函数计算损失.
            y_pred = w * x

            # 10. 计算损失, 最小二乘法.
            loss = (y_pred - y_true) ** 2

            # 11. 梯度清零 + 反向传播 + 优化器更新参数.(固定套路)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 12. 更新学习率.
        scheduler.step()

    # 13. 打印结果:
    print(f'lr_list: {lr_list}')        # [0.1, 0.1, 0.1..., 0.05........,0.025.........,  0.0125...]

    # 14. 可视化.
    # x轴: 训练的轮数, y轴: 每轮训练用的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
```


###  指定间隔学习率衰减

指定间隔学习率衰减的效果如下：

{% asset_img 04-03.png "Hexo 博客封面示例" %}

在PyTorch中实现时使用：

```python
# milestones：设定调整轮次:[50, 125, 160]
# gamma：调整系数
# 调整方式：lr = lr * gamma
optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)    
```

具体使用方式如下所示：

```python
import torch
from torch import optim
import matplotlib.pyplot as plt


def test_MultiStepLR():
    torch.manual_seed(1)
    LR = 0.1
    iteration = 10
    max_epoch = 200
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    optimizer = optim.SGD([w], lr=LR, momentum=0.9)
    # 设定调整时刻数
    milestones = [50, 125, 160]
    # 设置学习率下降策略
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):
            loss = (w*x-y_true)**2
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
        # 更新下一个epoch的学习率
        scheduler_lr.step()
    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()
```

课堂代码：
```python
"""
指定间隔学习率衰减:
    milestones = [50, 125, 160]     里边定义的是要调整学习率的 轮数.
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma
"""

# 2. 定义函数, 演示: 指定间隔学习率衰减
def dm02():
    # 1. 定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数.
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集.  y_true, x, w
    # 真实值.
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象, 动量法 -> 加速模型的收敛, 减少震荡.
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建学习率衰减对象.
    # 思路1: 创建等间隔学习率衰减对象.
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮调整一次学习率), 参3: 学习率衰减系数.
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.1, 0.1... 0.05...]

    # 思路2: 创建指定间隔学习率衰减对象.
    # 定义变量, 记录要修改学习率的轮数.
    milestones = [50, 125, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 5. 创建两个列表, 分别表示: 训练轮数, 每轮训练用的学习率
    # epoch_list = [0, 1, 2, 3.... 50, 51, 52...100, 101, 101... 150, 151...199]
    # lr_list =    [0.1, 0.1, 0.1, 0.05........,0.025.........,  0.0125...]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数, 进行具体的训练.
    for epoch in range(epochs):     # epoch: 0 ~ 199
        # 7. 获取当前轮数 和 学习率, 并保存到列表中.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())     # 获取最后的lr(learning rate, 学习率)

        # 8. 循环遍历, 每轮每批次进行训练.
        for batch in range(iteration):
            # 9. 先计算预测值, 然后基于损失函数计算损失.
            y_pred = w * x
            # 10. 计算损失, 最小二乘法.
            loss = (y_pred - y_true) ** 2
            # 11. 梯度清零 + 反向传播 + 优化器更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 12. 更新学习率.
        scheduler.step()
    # 13. 打印结果:
    print(f'lr_list: {lr_list}')        # [0.1, 0.1, 0.1..., 0.05........,0.025.........,  0.0125...]

    # 14. 可视化.
    # x轴: 训练的轮数, y轴: 每轮训练用的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
```

### 按指数学习率衰减

按指数衰减调整学习率的效果如下：

{% asset_img 04-04.png "Hexo 博客封面示例" %}

在PyTorch中实现时使用：

```python
# gamma：指数的底
# 调整方式
# lr= lr∗gamma^epoch
optim.lr_scheduler.ExponentialLR(optimizer, gamma)
```

gammma值不能太小，0.9几
API:ExponentialLR

具体使用方式如下所示：

```python
import torch
from torch import optim
import matplotlib.pyplot as plt


def test_ExponentialLR():
    # 0.参数初始化
    LR = 0.1  # 设置学习率初始化值为0.1
    iteration = 10
    max_epoch = 200
    # 1 初始化参数
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    # 2.优化器
    optimizer = optim.SGD([w], lr=LR, momentum=0.9)
    # 3.设置学习率下降策略
    gamma = 0.95
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # 4.获取学习率的值和当前的epoch
    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)
        for i in range(iteration):  # 遍历每一个batch数据
            loss = (w*x-y_true)**2
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
        # 更新下一个epoch的学习率
        scheduler_lr.step()
    # 5.绘制学习率变化的曲线
    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()
```

课堂代码：
```python
"""
指数间隔学习率衰减:
    前期学习率衰减快, 中期慢, 后期更慢, 更符合梯度下降规律.
    公式:
        lr新 = lr旧 * gamma ** epoch
"""

# 3. 定义函数, 演示: 指数学习率衰减
def dm03():
    # 1. 定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数.
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集.  y_true, x, w
    # 真实值.
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象, 动量法 -> 加速模型的收敛, 减少震荡.
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建学习率衰减对象.
    # 思路1: 创建等间隔学习率衰减对象.
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮调整一次学习率), 参3: 学习率衰减系数.
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.1, 0.1... 0.05...]

    # 思路2: 创建指定间隔学习率衰减对象.
    # 定义变量, 记录要修改学习率的轮数.
    # milestones = [50, 125, 160]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 思路3: 创建指数学习率衰减对象.
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 5. 创建两个列表, 分别表示: 训练轮数, 每轮训练用的学习率
    # epoch_list = [0, 1, 2, 3.... 50, 51, 52...100, 101, 101... 150, 151...199]
    # lr_list =    [0.1, 0.1, 0.1, 0.05........,0.025.........,  0.0125...]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数, 进行具体的训练.
    for epoch in range(epochs):     # epoch: 0 ~ 199
        # 7. 获取当前轮数 和 学习率, 并保存到列表中.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())     # 获取最后的lr(learning rate, 学习率)

        # 8. 循环遍历, 每轮每批次进行训练.
        for batch in range(iteration):
            # 9. 先计算预测值, 然后基于损失函数计算损失.
            y_pred = w * x
            # 10. 计算损失, 最小二乘法.
            loss = (y_pred - y_true) ** 2
            # 11. 梯度清零 + 反向传播 + 优化器更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 12. 更新学习率.
        scheduler.step()
    # 13. 打印结果:
    print(f'lr_list: {lr_list}')        # [0.1, 0.1, 0.1..., 0.05........,0.025.........,  0.0125...]

    # 14. 可视化.
    # x轴: 训练的轮数, y轴: 每轮训练用的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

```
### 小结

| 方法         | 等间隔学习率衰减 (Step Decay)    | 指定间隔学习率衰减 (Exponential Decay)     | 指数学习率衰减 (Exponential Moving Average Decay) |
| ------------ | -------------------------------- | ------------------------------------------ | ------------------------------------------------- |
| **衰减方式** | 固定步长衰减                     | 指定步长衰减                               | 平滑指数衰减，历史平均考虑                        |
| **实现难度** | 简单易实现                       | 相对简单，容易调整                         | 需要额外历史计算，较复杂                          |
| **适用场景** | 大型数据集、较为简单的任务       | 对训练平稳性要求较高的任务                 | 高精度训练，避免过快收敛                          |
| **优点**     | 直观，易于调试，适用于大批量数据 | 易于调试，稳定训练过程                     | 平滑且考虑历史更新，收敛稳定性较强                |
| **缺点**     | 学习率变化较大，可能跳过最优点   | 在某些情况下可能衰减过快，导致优化提前停滞 | 超参数调节较为复杂，可能需要更多的计算资源        |


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

### 梯度下降算法全面详解

梯度下降（Gradient Descent）是神经网络优化的核心算法。其基本思想是：沿着损失函数关于参数的负梯度方向更新参数，使损失逐步减小。然而，朴素的梯度下降在实际应用中面临收敛慢、易陷入鞍点、对学习率敏感等问题。围绕这些问题，研究者提出了一系列改进算法，本文按演进脉络逐一详解。

在展开之前，先明确一个统一的符号体系：

- \(\theta\)：模型参数（权重和偏置的统称）。
- \(L(\theta)\)：损失函数。
- \(g_t = \nabla_\theta L_t(\theta_{t-1})\)：第 \(t\) 步的梯度。
- \(\eta\)：学习率（learning rate）。
- \(t\)：当前迭代步数。


#### 四·一 梯度下降基础

##### 4.1.1 核心思想

梯度下降的基本更新规则为：

\[
\theta_t = \theta_{t-1} - \eta \cdot g_t
\]

**公式解释**：
- \(\theta_{t-1}\)：第 \(t-1\) 步（上一步）的参数值。
- \(g_t = \nabla_\theta L_t(\theta_{t-1})\)：损失函数在当前参数处关于 \(\theta\) 的梯度，指向损失增大最快的方向。
- \(\eta\)：学习率，控制每步更新的幅度。\(\eta\) 过大导致震荡甚至发散，过小导致收敛缓慢。
- 负号：沿梯度反方向更新，因为梯度指向损失增大的方向，反方向才是损失减小的方向。
- \(\theta_t\)：更新后的参数。

**几何直观**：将损失函数想象成一片山地，参数是当前位置，梯度是脚下最陡的上坡方向。梯度下降就是朝着最陡的下坡方向迈出一步，步长由学习率决定。重复此过程，最终到达山谷（局部最小值）。

##### 4.1.2 学习率的影响

学习率 \(\eta\) 是梯度下降中最重要的超参数：

| 学习率 | 表现 |
|--------|------|
| 过小（如 1e-6） | 收敛极慢，需要大量迭代 |
| 合适（如 0.01～0.1） | 平稳收敛到较优解 |
| 过大（如 10） | 震荡，甚至发散（损失增大） |
| 极大 | 参数更新到无穷，出现 NaN |


#### 四·二 批量梯度下降（BGD）

##### 4.2.1 算法描述

批量梯度下降（Batch Gradient Descent, BGD）在每次更新时使用**全部训练样本**计算梯度。

**公式**：

\[
\theta_t = \theta_{t-1} - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta L_i(\theta_{t-1})
\]

**公式解释**：
- \(n\)：训练集样本总数。
- \(L_i(\theta_{t-1})\)：第 \(i\) 个样本在当前参数下的损失。
- \(\frac{1}{n} \sum_{i=1}^{n} \nabla_\theta L_i\)：所有样本梯度的平均值，即全量梯度。
- 每次更新需要遍历整个数据集，计算代价高。

**优缺点**：
- 优点：梯度估计准确，收敛轨迹平滑，理论上能收敛到局部最小值。
- 缺点：每次更新需计算全部样本，大数据集（如百万级）不可行；内存无法一次性载入；无法在线学习。

```python
# 导入数值计算库
import numpy as np

# ==================== 批量梯度下降 - 从零实现 ====================
def batch_gradient_descent(X, y, lr=0.01, epochs=100):
    """批量梯度下降求解线性回归
    
    参数:
        X: 特征矩阵，形状(n, d)
        y: 标签向量，形状(n,)
        lr: 学习率
        epochs: 迭代轮数
    返回:
        theta: 优化后的参数，形状(d,)
        loss_history: 每轮损失值
    """
    # 获取样本数和特征数
    n, d = X.shape
    # 初始化参数为全零向量
    theta = np.zeros(d)
    # 记录损失历史
    loss_history = []
    
    # 迭代epochs轮
    for epoch in range(epochs):
        # 前向传播：计算预测值
        y_pred = X @ theta
        # 计算误差
        error = y_pred - y
        # 计算损失（MSE）
        loss = np.mean(error ** 2)
        # 记录损失
        loss_history.append(loss)
        # 计算全量梯度：dMSE/dtheta = 2/n * X^T @ error
        gradient = 2.0 / n * X.T @ error
        # 更新参数
        theta = theta - lr * gradient
    
    return theta, loss_history

# 构造示例数据：y = 3x1 + 2x2 + 1
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 2)
# 真实参数[3, 2]，加噪声
y = X @ np.array([3.0, 2.0]) + 1.0 + np.random.randn(n_samples) * 0.1

# 运行批量梯度下降
theta_bgd, loss_bgd = batch_gradient_descent(X, y, lr=0.1, epochs=50)
print(f"BGD估计参数: {theta_bgd}")
print(f"最终损失: {loss_bgd[-1]:.6f}")
```

**逐行解释**：

- `n, d = X.shape`：获取样本数 \(n\) 和特征数 \(d\)。
- `theta = np.zeros(d)`：参数初始化为全零（线性回归中零初始化即可，因为损失是凸的，不存在对称性问题）。
- `y_pred = X @ theta`：矩阵乘法，计算所有样本的预测值。形状 \((n, d) \times (d,) = (n,)\)。
- `error = y_pred - y`：预测值与真实值的差，形状 \((n,)\)。
- `loss = np.mean(error ** 2)`：MSE 损失。
- `gradient = 2.0 / n * X.T @ error`：对 MSE 关于 \(\theta\) 求导。推导：\(L = \frac{1}{n}\sum_i (x_i^\top \theta - y_i)^2\)，\(\frac{\partial L}{\partial \theta} = \frac{2}{n}\sum_i x_i (x_i^\top \theta - y_i) = \frac{2}{n} X^\top (X\theta - y)\)。
- `theta = theta - lr * gradient`：梯度下降更新。

**案例**：BGD 适用于小数据集（如几百到几千样本）和凸优化问题。在深度学习时代，BGD 几乎不再使用，因为数据集太大且模型非凸。


#### 四·三 随机梯度下降（SGD）

##### 4.3.1 算法描述

随机梯度下降（Stochastic Gradient Descent, SGD）每次仅使用**一个样本**计算梯度并更新参数。

**公式**：

\[
\theta_t = \theta_{t-1} - \eta \cdot \nabla_\theta L_i(\theta_{t-1})
\]

**公式解释**：
- \(L_i\)：第 \(i\) 个随机选取的样本的损失。
- \(\nabla_\theta L_i\)：单个样本的梯度，是真实梯度的**无偏估计**。
- 每次只计算一个样本的梯度，计算量极小。
- 由于随机采样，梯度含噪声，更新方向不完全指向最优方向。

**SGD 的数学性质**：

令全量梯度为 \(\bar{g} = \frac{1}{n}\sum_i \nabla L_i\)，单样本梯度 \(g_i = \nabla L_i\)。由于样本随机均匀采样：

\[
\mathbb{E}[g_i] = \bar{g}
\]

**公式解释**：
- \(\mathbb{E}[g_i]\)：单样本梯度的期望。
- 期望等于全量梯度，说明 SGD 是**无偏估计**。
- 但方差 \(\text{Var}(g_i)\) 不为零，导致更新路径含随机波动。

**优缺点**：
- 优点：每次更新计算量小，更新频率高；能逃离局部极小值和鞍点；支持在线学习。
- 缺点：梯度噪声大，损失曲线震荡剧烈；可能永远无法精确收敛到最小值，而是在最小值附近徘徊。

**学习率调度**：为缓解震荡，SGD 通常配合学习率衰减：

\[
\eta_t = \frac{\eta_0}{1 + \lambda t}
\]

或

\[
\eta_t = \eta_0 \cdot \gamma^t
\]

**公式解释**：
- \(\eta_0\)：初始学习率。
- \(\lambda\)：衰减速率。
- \(\gamma\)：衰减因子，通常取 0.9～0.99。
- 训练初期学习率大，快速接近最优解；后期学习率小，精细收敛。

```python
# ==================== 随机梯度下降 - 从零实现 ====================
def stochastic_gradient_descent(X, y, lr=0.01, epochs=50):
    """随机梯度下降求解线性回归
    
    参数:
        X: 特征矩阵，形状(n, d)
        y: 标签向量，形状(n,)
        lr: 学习率
        epochs: 迭代轮数
    返回:
        theta: 优化后的参数
        loss_history: 每轮平均损失
    """
    # 获取样本数和特征数
    n, d = X.shape
    # 初始化参数
    theta = np.zeros(d)
    loss_history = []
    
    # 迭代epochs轮
    for epoch in range(epochs):
        # 每轮开始前打乱样本顺序
        indices = np.random.permutation(n)
        # 遍历每个样本
        for i in indices:
            # 取出当前样本的特征和标签
            x_i = X[i]
            y_i = y[i]
            # 计算当前样本的预测值（标量）
            y_pred = x_i @ theta
            # 计算误差（标量）
            error = y_pred - y_i
            # 单样本梯度：2 * x_i * error
            gradient = 2.0 * x_i * error
            # 更新参数
            theta = theta - lr * gradient
        
        # 每轮结束后计算全量损失用于监控
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

# 运行随机梯度下降
theta_sgd, loss_sgd = stochastic_gradient_descent(X, y, lr=0.01, epochs=50)
print(f"SGD估计参数: {theta_sgd}")
print(f"最终损失: {loss_sgd[-1]:.6f}")
```

**逐行解释**：

- `indices = np.random.permutation(n)`：生成 0 到 n-1 的随机排列，实现每轮样本顺序的随机化，避免顺序偏差。
- `x_i = X[i]; y_i = y[i]`：取出单个样本。
- `y_pred = x_i @ theta`：单样本预测，结果是标量。
- `error = y_pred - y_i`：单样本误差。
- `gradient = 2.0 * x_i * error`：单样本 MSE 梯度。推导：\(L_i = (x_i^\top\theta - y_i)^2\)，\(\nabla L_i = 2(x_i^\top\theta - y_i) x_i\)。
- `theta = theta - lr * gradient`：立即更新参数。注意 SGD 是"逐样本更新"，一个 epoch 内更新 n 次。

**案例**：SGD 是深度学习早期的主流优化器。在 MNIST 等数据集上，SGD 配合学习率衰减可以取得不错的效果。但纯 SGD 对学习率非常敏感，实践中通常使用 Momentum 加速。


#### 四·四 小批量梯度下降（MBGD）

##### 4.4.1 算法描述

小批量梯度下降（Mini-Batch Gradient Descent, MBGD）是 BGD 和 SGD 的折中：每次使用一个小批量（通常 32～256 个样本）计算梯度。

**公式**：

\[
\theta_t = \theta_{t-1} - \eta \cdot \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta L_i(\theta_{t-1})
\]

**公式解释**：
- \(\mathcal{B}\)：当前小批量的样本索引集合。
- \(|\mathcal{B}|\)：批量大小（batch size）。
- \(\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta L_i\)：小批量内样本梯度的平均。
- 相比 BGD，计算量大幅减少；相比 SGD，梯度噪声降低。

**批量大小的影响**：

| 批量大小 | 特点 |
|---------|------|
| 1 | 等价于 SGD，噪声大，更新频繁 |
| 32～256 | 平衡噪声与效率，最常用 |
| 全部样本 | 等价于 BGD，无噪声，计算量大 |

**梯度噪声与批量大小的关系**：

\[
\text{Var}(\hat{g}_\mathcal{B}) = \frac{\sigma^2}{|\mathcal{B}|}
\]

**公式解释**：
- \(\hat{g}_\mathcal{B}\)：小批量梯度。
- \(\sigma^2\)：单样本梯度的方差。
- 批量越大，梯度方差越小，估计越准确。
- 但批量增大到一定程度后，方差下降的收益递减（因为 \(\frac{1}{|\mathcal{B}|}\) 是线性下降，但计算量线性增长）。

**为什么小批量适合 GPU**：

现代 GPU 擅长并行矩阵运算。批量大小为 \(B\) 时，前向传播计算 \(X \cdot W\)（\(X\) 形状 \((B, d_{in})\)），GPU 可以并行处理 \(B\) 个样本，计算时间几乎与 \(B=1\) 相同（只要显存足够）。因此，小批量既能利用 GPU 并行性，又能降低梯度噪声。

```python
# ==================== 小批量梯度下降 - 从零实现 ====================
def mini_batch_gradient_descent(X, y, batch_size=32, lr=0.01, epochs=50):
    """小批量梯度下降求解线性回归
    
    参数:
        X: 特征矩阵，形状(n, d)
        y: 标签向量，形状(n,)
        batch_size: 小批量大小
        lr: 学习率
        epochs: 迭代轮数
    返回:
        theta: 优化后的参数
        loss_history: 每轮平均损失
    """
    # 获取样本数和特征数
    n, d = X.shape
    # 初始化参数
    theta = np.zeros(d)
    loss_history = []
    
    # 迭代epochs轮
    for epoch in range(epochs):
        # 打乱样本顺序
        indices = np.random.permutation(n)
        # 按batch_size切分为多个小批量
        for start in range(0, n, batch_size):
            # 取当前批量的索引
            batch_indices = indices[start:start + batch_size]
            # 取当前批量的特征和标签
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 前向传播：计算批量预测值
            y_pred = X_batch @ theta
            # 计算批量误差
            error = y_pred - y_batch
            # 计算批量梯度：2/B * X_batch^T @ error
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新参数
            theta = theta - lr * gradient
        
        # 每轮结束后计算全量损失
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

# 运行小批量梯度下降
theta_mbgd, loss_mbgd = mini_batch_gradient_descent(X, y, batch_size=32, lr=0.1, epochs=50)
print(f"MBGD估计参数: {theta_mbgd}")
print(f"最终损失: {loss_mbgd[-1]:.6f}")
```

**逐行解释**：

- `indices = np.random.permutation(n)`：每轮打乱样本顺序，避免同一批量的样本总是相同。
- `for start in range(0, n, batch_size)`：按 batch_size 步长遍历，将 n 个样本切分为 \(\lceil n / B \rceil\) 个批量。
- `X_batch = X[batch_indices]`：高级索引，取出当前批量的样本，形状 \((B, d)\)。
- `y_pred = X_batch @ theta`：批量矩阵乘法，形状 \((B, d) \times (d,) = (B,)\)。
- `gradient = 2.0 / len(batch_indices) * X_batch.T @ error`：批量梯度，\(X_{batch}^\top\) 形状 \((d, B)\)，乘 error \((B,)\) 得 \((d,)\)。
- `theta = theta - lr * gradient`：批量更新。一个 epoch 内更新 \(\lceil n/B \rceil\) 次。

**案例**：小批量梯度下降是当前深度学习的**事实标准**。PyTorch 的 `DataLoader` 默认使用小批量，batch_size 通常设为 32、64、128、256。批量大小与学习率通常呈线性关系：批量翻倍，学习率也可翻倍（线性缩放规则）。


#### 四·五 Momentum（动量法）

##### 4.5.1 算法描述

动量法（Momentum）由 Polyak 于 1964 年提出，通过累积历史梯度形成"速度"，使更新方向更稳定、收敛更快。

**公式**：

\[
v_t = \beta v_{t-1} + (1-\beta) g_t
\]

\[
\theta_t = \theta_{t-1} - \eta v_t
\]

**公式解释**：
- \(v_t\)：第 \(t\) 步的速度（动量项），累积历史梯度信息。
- \(\beta\)：动量系数，典型值 0.9。\(\beta\) 越大，历史梯度影响越大，更新越平滑。
- \(v_{t-1}\)：上一步的速度。
- \(g_t\)：当前梯度。
- \((1-\beta) g_t\)：当前梯度的加权项。用 \((1-\beta)\) 而非 1 是为了使 \(v_t\) 成为梯度的指数移动平均，避免量级累积。
- \(\theta_t = \theta_{t-1} - \eta v_t\)：用速度而非梯度更新参数。

**展开形式**：

将递推式展开：

\[
v_t = (1-\beta) \sum_{k=1}^{t} \beta^{t-k} g_k
\]

**公式解释**：
- \(g_k\)：第 \(k\) 步的梯度。
- \(\beta^{t-k}\)：权重，越早的梯度权重越小（指数衰减）。
- \((1-\beta)\)：归一化因子，保证所有权重之和为 1：\((1-\beta)\sum_{k=1}^{t}\beta^{t-k} = 1 - \beta^t \approx 1\)。
- 因此 \(v_t\) 是历史梯度的指数加权移动平均（EWMA）。

**物理意义**：
- 当梯度方向持续一致时，历史梯度累加，速度增大，加速前进。
- 当梯度方向反复变化时，正负梯度抵消，速度减小，减少震荡。
- 类似于球从山坡滚下，动量使其在平坦区域继续前进，不易被困在小的坑洼中。

**Nesterov 动量**：

Nesterov 加速梯度（NAG）是动量法的改进，先按速度前进一步，再在该位置计算梯度：

\[
v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta L(\theta_{t-1} - \eta \beta v_{t-1})
\]

\[
\theta_t = \theta_{t-1} - \eta v_t
\]

**公式解释**：
- \(\theta_{t-1} - \eta\beta v_{t-1}\)：先按动量方向"前瞻"一步。
- \(\nabla_\theta L(\theta_{t-1} - \eta\beta v_{t-1})\)：在"前瞻"位置计算梯度，而非当前位置。
- 这使得 NAG 能提前感知前方的梯度变化，在接近最小值时自动减速，减少过冲。
- NAG 的收敛速度理论上优于标准 Momentum（\(O(1/t^2)\) vs \(O(1/t)\)）。

```python
# ==================== Momentum - 从零实现 ====================
def momentum_sgd(X, y, lr=0.01, beta=0.9, batch_size=32, epochs=50):
    """Momentum SGD求解线性回归
    
    参数:
        X: 特征矩阵
        y: 标签向量
        lr: 学习率
        beta: 动量系数
        batch_size: 批量大小
        epochs: 迭代轮数
    返回:
        theta: 优化后的参数
        loss_history: 每轮损失
    """
    n, d = X.shape
    theta = np.zeros(d)
    # 初始化速度为零
    v = np.zeros(d)
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 计算批量梯度
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新速度：v = beta * v_prev + (1-beta) * g
            v = beta * v + (1 - beta) * gradient
            # 用速度更新参数
            theta = theta - lr * v
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

# 对比普通SGD和Momentum
theta_mom, loss_mom = momentum_sgd(X, y, lr=0.1, beta=0.9, batch_size=32, epochs=50)
print(f"Momentum估计参数: {theta_mom}")
print(f"最终损失: {loss_mom[-1]:.6f}")
```

**逐行解释**：

- `v = np.zeros(d)`：速度初始化为零向量，与参数同形状。
- `v = beta * v + (1 - beta) * gradient`：核心动量更新。历史速度保留 \(\beta\) 比例，当前梯度贡献 \((1-\beta)\) 比例。
- `theta = theta - lr * v`：用速度更新参数，而非直接用梯度。

**案例**：Momentum 在图像分类任务中广泛使用。PyTorch 中设置 `momentum=0.9` 是 SGD 的标准配置。在 ResNet 训练中，Momentum 比纯 SGD 收敛快约 2～3 倍。

**Nesterov 实现**：

```python
# ==================== Nesterov 动量 - 从零实现 ====================
def nesterov_sgd(X, y, lr=0.01, beta=0.9, batch_size=32, epochs=50):
    """Nesterov加速梯度求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    v = np.zeros(d)
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 前瞻位置：theta_lookahead = theta - lr * beta * v
            theta_lookahead = theta - lr * beta * v
            # 在前瞻位置计算梯度
            error = X_batch @ theta_lookahead - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新速度
            v = beta * v + (1 - beta) * gradient
            # 更新参数
            theta = theta - lr * v
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_nag, loss_nag = nesterov_sgd(X, y, lr=0.1, beta=0.9, batch_size=32, epochs=50)
print(f"Nesterov估计参数: {theta_nag}")
print(f"最终损失: {loss_nag[-1]:.6f}")
```

**PyTorch 中的使用**：

```python
import torch.optim as optim
# 标准Momentum
optimizer_mom = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Nesterov动量
optimizer_nag = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```


#### 四·六 AdaGrad

##### 4.6.1 算法描述

AdaGrad（Adaptive Gradient）由 Duchi 等人于 2011 年提出，是第一个自适应学习率算法。它为每个参数维护一个累积梯度平方和，根据历史梯度动态调整学习率。

**公式**：

\[
s_t = s_{t-1} + g_t^2
\]

\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
\]

**公式解释**：
- \(s_t\)：到第 \(t\) 步为止，所有梯度平方的累积和（逐元素）。
- \(g_t^2\)：当前梯度的逐元素平方。
- \(s_t = s_{t-1} + g_t^2\)：累积梯度平方，单调递增。
- \(\frac{\eta}{\sqrt{s_t + \epsilon}}\)：自适应学习率。梯度大的参数，\(s_t\) 大，学习率被缩小；梯度小的参数，\(s_t\) 小，学习率被放大。
- \(\epsilon\)：极小常数（如 \(10^{-8}\)），防止分母为零。
- 效果：稀疏特征（如 NLP 中的词袋模型）获得更大的学习率，频繁特征获得更小的学习率。

**优缺点**：
- 优点：无需手动调节学习率；适合稀疏数据；对每个参数自适应。
- 缺点：\(s_t\) 单调递增，导致学习率单调递减。训练后期学习率趋于零，参数几乎不再更新，无法继续学习。

**适用场景**：AdaGrad 适合**凸优化**和**稀疏梯度**问题，如词袋模型、推荐系统。在深度学习中，由于学习率过早衰减，效果通常不如 RMSprop 和 Adam。

```python
# ==================== AdaGrad - 从零实现 ====================
def adagrad(X, y, lr=0.01, epsilon=1e-8, batch_size=32, epochs=50):
    """AdaGrad求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    # 初始化累积梯度平方和
    s = np.zeros(d)
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 计算梯度
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 累积梯度平方
            s = s + gradient ** 2
            # 自适应学习率更新
            theta = theta - lr / np.sqrt(s + epsilon) * gradient
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_adagrad, loss_adagrad = adagrad(X, y, lr=0.5, batch_size=32, epochs=50)
print(f"AdaGrad估计参数: {theta_adagrad}")
print(f"最终损失: {loss_adagrad[-1]:.6f}")
```

**逐行解释**：

- `s = np.zeros(d)`：累积梯度平方和，与参数同形状。
- `s = s + gradient ** 2`：逐元素累积。注意是累加，不是移动平均，因此 \(s\) 只增不减。
- `theta = theta - lr / np.sqrt(s + epsilon) * gradient`：自适应更新。若某参数梯度一直很大，\(s\) 大，学习率被压缩；反之被放大。

**PyTorch 中的使用**：

```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10)
```


#### 四·七 RMSprop

##### 4.7.1 算法描述

RMSprop（Root Mean Square Propagation）由 Hinton 在 Coursera 课程中提出（未正式发表），解决了 AdaGrad 学习率单调递减的问题。它将累积梯度平方改为**指数移动平均**，使 \(s_t\) 不会无限增大。

**公式**：

\[
s_t = \gamma s_{t-1} + (1-\gamma) g_t^2
\]

\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
\]

**公式解释**：
- \(\gamma\)：衰减率，典型值 0.9 或 0.99。
- \(s_t = \gamma s_{t-1} + (1-\gamma) g_t^2\)：梯度平方的指数移动平均（EWMA）。
- 与 AdaGrad 的累积不同，RMSprop 只保留最近一段时间梯度的信息，旧的梯度逐渐被遗忘。
- \(\sqrt{s_t}\)：梯度平方的平均的平方根，即 RMS（Root Mean Square）。
- 学习率被 \(1/\sqrt{s_t}\) 缩放，梯度大时学习率小，梯度小时学习率大。

**指数移动平均展开**：

\[
s_t = (1-\gamma) \sum_{k=1}^{t} \gamma^{t-k} g_k^2
\]

**公式解释**：
- \(\gamma^{t-k}\)：越早的梯度权重越小。
- 有效记忆长度约为 \(\frac{1}{1-\gamma}\)。当 \(\gamma=0.9\) 时，约记忆最近 10 步的梯度。

**优缺点**：
- 优点：解决了 AdaGrad 学习率过早衰减的问题；适合非平稳目标（梯度分布随时间变化）；对 RNN 等序列模型效果良好。
- 缺点：仍需手动设置全局学习率；在部分任务上泛化性能不如 SGD+Momentum。

```python
# ==================== RMSprop - 从零实现 ====================
def rmsprop(X, y, lr=0.01, gamma=0.9, epsilon=1e-8, batch_size=32, epochs=50):
    """RMSprop求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    # 初始化梯度平方的指数移动平均
    s = np.zeros(d)
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 计算梯度
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新梯度平方的指数移动平均
            s = gamma * s + (1 - gamma) * gradient ** 2
            # 自适应更新
            theta = theta - lr / np.sqrt(s + epsilon) * gradient
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_rmsprop, loss_rmsprop = rmsprop(X, y, lr=0.05, gamma=0.9, batch_size=32, epochs=50)
print(f"RMSprop估计参数: {theta_rmsprop}")
print(f"最终损失: {loss_rmsprop[-1]:.6f}")
```

**逐行解释**：

- `s = np.zeros(d)`：初始化梯度平方的 EWMA。
- `s = gamma * s + (1 - gamma) * gradient ** 2`：EWMA 更新。历史值保留 \(\gamma\) 比例，当前梯度平方贡献 \(1-\gamma\) 比例。
- `theta = theta - lr / np.sqrt(s + epsilon) * gradient`：自适应更新，与 AdaGrad 相同。

**PyTorch 中的使用**：

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)
```

**案例**：RMSprop 在训练 RNN、LSTM 等序列模型时表现良好，因为序列模型的梯度分布随序列位置变化，RMSprop 的自适应特性可以有效应对。


#### 四·八 AdaDelta

##### 4.8.1 算法描述

AdaDelta 由 Zeiler 于 2012 年提出，是 AdaGrad 和 RMSprop 的进一步改进。它的核心创新是**去除全局学习率**，完全依赖自适应机制。

**公式**：

\[
s_t = \gamma s_{t-1} + (1-\gamma) g_t^2
\]

\[
\Delta\theta_t = -\frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}} g_t
\]

\[
u_t = \gamma u_{t-1} + (1-\gamma) \Delta\theta_t^2
\]

\[
\theta_t = \theta_{t-1} + \Delta\theta_t
\]

**公式解释**：
- \(s_t\)：梯度平方的 EWMA（与 RMSprop 相同）。
- \(u_t\)：参数更新量平方的 EWMA。
- \(\Delta\theta_t\)：第 \(t\) 步的参数更新量。
- \(\frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}}\)：用历史更新量的 RMS 替代学习率。其思想是：如果历史更新量较大，说明当前学习率可能合适或偏大，用历史 RMS 作为参照。
- 无需设置学习率 \(\eta\)，这是 AdaDelta 的最大特点。

**优缺点**：
- 优点：完全免去学习率调节；对超参数不敏感。
- 缺点：在深度学习中，去除学习率后灵活性下降；实践中 Adam 更受欢迎。

```python
# ==================== AdaDelta - 从零实现 ====================
def adadelta(X, y, gamma=0.95, epsilon=1e-6, batch_size=32, epochs=50):
    """AdaDelta求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    # 梯度平方的EWMA
    s = np.zeros(d)
    # 更新量平方的EWMA
    u = np.zeros(d)
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 计算梯度
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新梯度平方的EWMA
            s = gamma * s + (1 - gamma) * gradient ** 2
            # 计算参数更新量
            delta = -np.sqrt(u + epsilon) / np.sqrt(s + epsilon) * gradient
            # 更新更新量平方的EWMA
            u = gamma * u + (1 - gamma) * delta ** 2
            # 更新参数
            theta = theta + delta
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_adadelta, loss_adadelta = adadelta(X, y, gamma=0.95, batch_size=32, epochs=50)
print(f"AdaDelta估计参数: {theta_adadelta}")
print(f"最终损失: {loss_adadelta[-1]:.6f}")
```

**逐行解释**：

- `s = np.zeros(d)`：梯度平方的 EWMA。
- `u = np.zeros(d)`：更新量平方的 EWMA。
- `delta = -np.sqrt(u + epsilon) / np.sqrt(s + epsilon) * gradient`：计算更新量。分子 \(\sqrt{u}\) 是历史更新量的 RMS，替代了学习率。
- `u = gamma * u + (1 - gamma) * delta ** 2`：更新 \(u\)。
- `theta = theta + delta`：应用更新量。注意此时 delta 已含负号。

**PyTorch 中的使用**：

```python
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
```


#### 四·九 Adam

##### 4.9.1 算法描述

Adam（Adaptive Moment Estimation）由 Kingma 和 Ba 于 2014 年提出，结合了 Momentum 和 RMSprop 的优点：同时使用一阶动量（梯度的 EWMA）和二阶动量（梯度平方的 EWMA）。

**公式**：

**一阶动量**：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

**二阶动量**：

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

**偏差修正**：

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\]

\[
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

**参数更新**：

\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\]

**公式解释**：
- \(m_t\)：梯度的一阶动量，类似 Momentum 中的速度。
- \(v_t\)：梯度平方的二阶动量，类似 RMSprop 中的 \(s_t\)。
- \(\beta_1\)：一阶动量衰减率，典型值 0.9。
- \(\beta_2\)：二阶动量衰减率，典型值 0.999。
- \(\hat{m}_t, \hat{v}_t\)：偏差修正后的动量。
- \(\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t\)：结合一阶动量的方向和二阶动量的自适应尺度。

**偏差修正的必要性**：

初始化 \(m_0 = 0, v_0 = 0\)。第 1 步：

\[
m_1 = (1-\beta_1) g_1
\]

由于 \(\beta_1 = 0.9\)，\(m_1 = 0.1 g_1\)，仅为真实梯度的 1/10。若不加修正，初期更新步长过小。

偏差修正除以 \(1-\beta_1^t\)：

\[
\hat{m}_1 = \frac{0.1 g_1}{1 - 0.9^1} = \frac{0.1 g_1}{0.1} = g_1
\]

修正后 \(\hat{m}_1 = g_1\)，恢复了真实梯度。随着 \(t\) 增大，\(\beta_1^t \to 0\)，修正项趋近于 1，影响消失。

**优缺点**：
- 优点：收敛快，稳定性好；对超参数不敏感（\(\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}\) 几乎通用）；适合大多数任务。
- 缺点：在某些任务上泛化性能略逊于 SGD+Momentum；可能收敛到尖锐的局部最小值。

```python
# ==================== Adam - 从零实现 ====================
def adam(X, y, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, epochs=50):
    """Adam求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    # 一阶动量
    m = np.zeros(d)
    # 二阶动量
    v = np.zeros(d)
    # 步数计数器
    t = 0
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            # 步数加1
            t += 1
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # 计算梯度
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新一阶动量
            m = beta1 * m + (1 - beta1) * gradient
            # 更新二阶动量
            v = beta2 * v + (1 - beta2) * gradient ** 2
            # 偏差修正
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            # 参数更新
            theta = theta - lr / (np.sqrt(v_hat) + epsilon) * m_hat
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_adam, loss_adam = adam(X, y, lr=0.05, batch_size=32, epochs=50)
print(f"Adam估计参数: {theta_adam}")
print(f"最终损失: {loss_adam[-1]:.6f}")
```

**逐行解释**：

- `m = np.zeros(d)`：一阶动量初始化。
- `v = np.zeros(d)`：二阶动量初始化。
- `t = 0`：步数计数器，用于偏差修正。
- `t += 1`：每次参数更新前递增。
- `m = beta1 * m + (1 - beta1) * gradient`：一阶动量 EWMA。
- `v = beta2 * v + (1 - beta2) * gradient ** 2`：二阶动量 EWMA。
- `m_hat = m / (1 - beta1 ** t)`：一阶动量偏差修正。
- `v_hat = v / (1 - beta2 ** t)`：二阶动量偏差修正。
- `theta = theta - lr / (np.sqrt(v_hat) + epsilon) * m_hat`：Adam 更新。

**PyTorch 中的使用**：

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

**案例**：Adam 是当前深度学习中最常用的优化器。在 Transformer、BERT、GPT 等模型中，Adam 是默认选择。对于大多数任务，Adam 配合学习率 1e-3 或 1e-4 即可获得良好效果。


#### 四·十 Adamax

##### 4.10.1 算法描述

Adamax 是 Adam 的变体，由 Kingma 和 Ba 在同一论文中提出。它将 Adam 中的二阶动量从 L2 范数推广到无穷范数（L∞ 范数）。

**公式**：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

\[
u_t = \max(\beta_2 u_{t-1}, |g_t|)
\]

\[
\theta_t = \theta_{t-1} - \frac{\eta}{u_t} \hat{m}_t
\]

**公式解释**：
- \(m_t\)：一阶动量，与 Adam 相同。
- \(u_t\)：二阶动量，使用梯度的无穷范数（即历史梯度的最大绝对值），而非 Adam 中的平方 EWMA。
- \(\max(\beta_2 u_{t-1}, |g_t|)\)：取历史最大值与当前梯度绝对值的较大者。
- 使用 \(u_t\) 而非 \(\sqrt{v_t}\)，不需要开方。
- 偏差修正与 Adam 类似：\(\hat{m}_t = \frac{m_t}{1-\beta_1^t}\)。

**与 Adam 的对比**：
- Adam 的二阶动量是梯度平方的指数衰减平均，对历史梯度都有记忆。
- Adamax 的二阶动量是历史梯度的最大绝对值，更关注极端值。
- 当梯度分布有重尾时，Adamax 可能更稳定。

```python
# ==================== Adamax - 从零实现 ====================
def adamax(X, y, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, epochs=50):
    """Adamax求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)
    # 二阶动量初始化为零
    u = np.zeros(d)
    t = 0
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            t += 1
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 一阶动量
            m = beta1 * m + (1 - beta1) * gradient
            # 二阶动量：取历史最大值
            u = np.maximum(beta2 * u, np.abs(gradient))
            # 偏差修正
            m_hat = m / (1 - beta1 ** t)
            # 参数更新（注意分母为u，无sqrt）
            theta = theta - lr / (u + epsilon) * m_hat
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_adamax, loss_adamax = adamax(X, y, lr=0.05, batch_size=32, epochs=50)
print(f"Adamax估计参数: {theta_adamax}")
print(f"最终损失: {loss_adamax[-1]:.6f}")
```

**逐行解释**：

- `u = np.zeros(d)`：二阶动量初始化。
- `u = np.maximum(beta2 * u, np.abs(gradient))`：更新二阶动量。先对历史值乘以 \(\beta_2\) 衰减，再与当前梯度绝对值取最大。
- `theta = theta - lr / (u + epsilon) * m_hat`：分母为 \(u\) 而非 \(\sqrt{v}\)，不需开方。

**PyTorch 中的使用**：

```python
optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-8)
```


#### 四·十一 Nadam

##### 4.11.1 算法描述

Nadam（Nesterov-accelerated Adam）由 Dozat 于 2016 年提出，将 Nesterov 动量引入 Adam。

**公式**：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

**Nesterov 修正**：

\[
\hat{m}_t = \frac{\beta_1 m_t}{1 - \beta_1^{t+1}} + \frac{(1-\beta_1) g_t}{1 - \beta_1^t}
\]

\[
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

**参数更新**：

\[
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\]

**公式解释**：
- \(\hat{m}_t\)：Nesterov 修正后的一阶动量。
- 第一项 \(\frac{\beta_1 m_t}{1-\beta_1^{t+1}}\)：当前动量经偏差修正。
- 第二项 \(\frac{(1-\beta_1) g_t}{1-\beta_1^t}\)：当前梯度的修正项。
- 相比 Adam，Nadam 在动量项中加入了当前梯度的"前瞻"信息，使更新方向更准确。
- 实践中，Nadam 通常比 Adam 收敛稍快，尤其在小批量任务中。

```python
# ==================== Nadam - 从零实现 ====================
def nadam(X, y, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, epochs=50):
    """Nadam求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    t = 0
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            t += 1
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新一阶和二阶动量
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            # 偏差修正
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            # Nesterov修正：加入当前梯度项
            m_nesterov = beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1 ** t)
            # 参数更新
            theta = theta - lr / (np.sqrt(v_hat) + epsilon) * m_nesterov
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_nadam, loss_nadam = nadam(X, y, lr=0.05, batch_size=32, epochs=50)
print(f"Nadam估计参数: {theta_nadam}")
print(f"最终损失: {loss_nadam[-1]:.6f}")
```

**逐行解释**：

- `m_nesterov = beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1 ** t)`：Nesterov 修正。相比 Adam 的 \(\hat{m}\)，这里加入了当前梯度的修正项。
- 其余与 Adam 相同。

**PyTorch 中的使用**：

```python
optimizer = optim.NAdam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-8)
```


#### 四·十二 RAdam

##### 4.12.1 算法描述

RAdam（Rectified Adam）由 Liu 等人于 2019 年提出，针对 Adam 在训练初期方差过大、更新不稳定的问题。RAdam 通过一个"整流项"自动调整自适应学习率的启用时机：在训练初期（方差大）退化为 SGD+Momentum，在训练后期（方差小）切换为 Adam。

**公式**：

**一阶动量**：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]

**二阶动量**：

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

**未中心化方差估计**：

\[
\rho_t = \rho_\infty - \frac{2t \beta_2^t}{1 - \beta_2^t}
\]

其中 \(\rho_\infty = \frac{2}{1-\beta_2} - 1\)。

**整流项**：

\[
r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}
\]

**参数更新**：

\[
\theta_t = \begin{cases} \theta_{t-1} - \eta \cdot \hat{m}_t, & \rho_t \le 4 \\ \theta_{t-1} - \frac{\eta \cdot r_t}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t, & \rho_t > 4 \end{cases}
\]

**公式解释**：
- \(\rho_t\)：有效样本量（effective sample size）的估计，反映当前梯度估计的可靠程度。
- \(\rho_\infty\)：\(\rho_t\) 的渐近上界，当 \(\beta_2 = 0.999\) 时约为 1999。
- \(r_t\)：整流系数，当 \(\rho_t \le 4\) 时，梯度方差过大，不足以支持自适应学习率，RAdam 退化为 SGD+Momentum（不带分母的 Adam）。
- 当 \(\rho_t > 4\) 后，\(r_t\) 逐渐接近 1，RAdam 行为接近 Adam。
- 这解决了 Adam 在训练初期因偏差修正导致的更新不稳定问题。

**优缺点**：
- 优点：无需学习率预热（warmup）；训练初期更稳定；对学习率不敏感。
- 缺点：实现复杂；部分任务上与 Adam 效果差异不大。

```python
# ==================== RAdam - 从零实现 ====================
def radam(X, y, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, epochs=50):
    """RAdam求解线性回归"""
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    t = 0
    loss_history = []
    # 计算rho_inf
    rho_inf = 2.0 / (1 - beta2) - 1
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            t += 1
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            error = X_batch @ theta - y_batch
            gradient = 2.0 / len(batch_indices) * X_batch.T @ error
            # 更新一阶和二阶动量
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            # 偏差修正
            m_hat = m / (1 - beta1 ** t)
            # 计算rho_t
            rho_t = rho_inf - 2.0 * t * beta2 ** t / (1 - beta2 ** t)
            # 根据rho_t选择更新方式
            if rho_t > 4:
                # 计算整流项
                r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                              ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                # 计算v_hat
                v_hat = v / (1 - beta2 ** t)
                # Adam式更新
                theta = theta - lr * r_t / (np.sqrt(v_hat) + epsilon) * m_hat
            else:
                # 退化为SGD+Momentum
                theta = theta - lr * m_hat
        
        y_pred_all = X @ theta
        loss = np.mean((y_pred_all - y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

theta_radam, loss_radam = radam(X, y, lr=0.05, batch_size=32, epochs=50)
print(f"RAdam估计参数: {theta_radam}")
print(f"最终损失: {loss_radam[-1]:.6f}")
```

**逐行解释**：

- `rho_inf = 2.0 / (1 - beta2) - 1`：\(\rho_\infty\) 的计算公式，当 \(\beta_2=0.999\) 时约为 1999。
- `rho_t = rho_inf - 2.0 * t * beta2 ** t / (1 - beta2 ** t)`：\(\rho_t\) 的计算。随 \(t\) 增大，第二项趋近于 0，\(\rho_t \to \rho_\infty\)。
- `if rho_t > 4`：判断是否启用自适应学习率。
- `r_t = ...`：整流项计算。
- 更新分支：\(\rho_t \le 4\) 时用 SGD+Momentum，否则用带整流项的 Adam。

**PyTorch 中的使用**：

```python
optimizer = optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```


#### 四·十三 优化器对比实验

以下代码在同一数据集上对比各优化器的收敛速度和最终损失：

```python
# 导入PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 构造非线性回归数据：y = sin(x1) + cos(x2) + 噪声
n_samples = 2000
X_data = torch.randn(n_samples, 2)
y_data = torch.sin(X_data[:, 0]) + torch.cos(X_data[:, 1]) + 0.1 * torch.randn(n_samples)

# 定义两层MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze()

# 定义优化器字典
optimizers_config = {
    'SGD': lambda p: optim.SGD(p, lr=0.01),
    'Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    'Nesterov': lambda p: optim.SGD(p, lr=0.01, momentum=0.9, nesterov=True),
    'AdaGrad': lambda p: optim.Adagrad(p, lr=0.1),
    'RMSprop': lambda p: optim.RMSprop(p, lr=0.01),
    'AdaDelta': lambda p: optim.Adadelta(p, lr=1.0),
    'Adam': lambda p: optim.Adam(p, lr=0.01),
    'Adamax': lambda p: optim.Adamax(p, lr=0.01),
    'NAdam': lambda p: optim.NAdam(p, lr=0.01),
    'RAdam': lambda p: optim.RAdam(p, lr=0.01),
}

# 批量大小
batch_size = 64
# 训练轮数
epochs = 100

# 记录每个优化器的损失曲线
loss_curves = {}

# 逐个训练
for name, opt_fn in optimizers_config.items():
    # 每次重新初始化模型
    torch.manual_seed(42)
    model = MLP()
    # 创建优化器
    optimizer = opt_fn(model.parameters())
    # MSE损失
    criterion = nn.MSELoss()
    # 记录损失
    losses = []
    
    for epoch in range(epochs):
        # 打乱数据
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        # 小批量训练
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_data[idx]
            y_batch = y_data[idx]
            # 前向传播
            pred = model(x_batch)
            # 计算损失
            loss = criterion(pred, y_batch)
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计损失
            epoch_loss += loss.item() * len(idx)
        # 记录平均损失
        losses.append(epoch_loss / n_samples)
    
    loss_curves[name] = losses
    print(f"{name:10s} 最终损失: {losses[-1]:.6f}")

# 绘制损失曲线
plt.figure(figsize=(12, 6))
for name, losses in loss_curves.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True)
# 纵轴取对数，便于观察差异
plt.yscale('log')
plt.show()
```

**实验结果解读**：

- **SGD**：收敛最慢，损失下降缓慢。
- **Momentum / Nesterov**：比 SGD 快，Nesterov 略优于标准 Momentum。
- **AdaGrad**：初期收敛快，后期学习率过小，损失下降停滞。
- **RMSprop / AdaDelta**：自适应学习率，收敛平稳。
- **Adam / Adamax / NAdam / RAdam**：收敛最快，最终损失最低。NAdam 和 RAdam 在小批量时略优于 Adam。


#### 四·十四 优化器选择指南

| 优化器 | 提出年份 | 核心思想 | 适用场景 | PyTorch API |
|--------|---------|---------|---------|-------------|
| BGD | — | 全量梯度 | 小数据集、凸问题 | 手动实现 |
| SGD | — | 单样本梯度 | 在线学习、理论分析 | `optim.SGD` |
| MBGD | — | 小批量梯度 | 深度学习标准 | `optim.SGD` + DataLoader |
| Momentum | 1964 | 累积历史梯度 | 加速收敛、逃离鞍点 | `momentum=0.9` |
| Nesterov | 1983 | 前瞻梯度 | 比 Momentum 更快 | `nesterov=True` |
| AdaGrad | 2011 | 累积梯度平方 | 稀疏数据、凸问题 | `optim.Adagrad` |
| RMSprop | 2012 | 梯度平方 EWMA | RNN、非平稳目标 | `optim.RMSprop` |
| AdaDelta | 2012 | 更新量 RMS 替代学习率 | 免学习率调节 | `optim.Adadelta` |
| Adam | 2014 | 一阶+二阶动量 | 大多数任务默认 | `optim.Adam` |
| Adamax | 2014 | 无穷范数二阶动量 | 重尾梯度 | `optim.Adamax` |
| Nadam | 2016 | Nesterov + Adam | 小批量、快速收敛 | `optim.NAdam` |
| RAdam | 2019 | 整流自适应学习率 | 免预热、稳定训练 | `optim.RAdam` |

**实践建议**：

1. **默认选择 Adam**：学习率 1e-3 或 1e-4，适用于大多数任务。Adam 对超参数不敏感，是快速获得 baseline 的首选。

2. **追求最优泛化性能时选择 SGD + Momentum**：学习率 0.01～0.1，momentum=0.9。在图像分类（ResNet、VGG）中，SGD+Momentum 配合学习率衰减通常比 Adam 获得更好的测试精度。

3. **训练不稳定时尝试 RAdam 或 AdamW**：RAdam 无需学习率预热，适合训练初期不稳定的场景。AdamW 修正了 Adam 中权重衰减的实现方式，是 Transformer 训练的标准选择。

4. **RNN/LSTM 任务使用 RMSprop 或 Adam**：序列模型的梯度分布随位置变化，自适应学习率算法表现更好。

5. **稀疏数据使用 AdaGrad**：如词袋模型、推荐系统，AdaGrad 对稀疏特征的自适应学习率非常有效。

6. **学习率调度**：无论使用哪种优化器，配合学习率调度（如余弦退火、阶梯衰减、OneCycle）通常能进一步提升效果。


#### 四·十五 核心结论

梯度下降算法的演进围绕三个核心问题展开：

1. **收敛速度**：从 BGD/SGD 到 Momentum/Nesterov，通过累积历史梯度加速收敛。动量法的本质是对梯度做指数移动平均，平滑更新方向。

2. **学习率自适应**：从 AdaGrad 到 RMSprop/Adam，通过梯度平方的历史信息自动调整每个参数的学习率。梯度大的参数学习率小，梯度小的参数学习率大。

3. **训练稳定性**：从 Adam 到 RAdam，通过偏差修正、整流项等机制解决训练初期的不稳定问题。Adam 的偏差修正是关键创新，使初期更新步长不被低估。

理解这些算法的最佳方式是**动手对比**：在同一个数据集上运行不同优化器，观察损失曲线的差异。理论分析告诉你"为什么"，实验对比告诉你"效果如何"，两者结合才能建立真正的直觉。


## 五、网络优化方法(精简版)

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

### 什么是正则化

- 防止模型过拟合(训练集效果好, 测试集效果差), 提高模型泛化能力
- 一种防止过拟合, 提高模型泛化能力的策略
  - L1正则: 需要通过手动写代码实现  
  - L2正则: SGD(weight_decay=)
  - dropout
  - BN

过拟合是深度学习中常见的核心问题——模型在训练集上表现极好，但在测试集上表现很差。正则化技术通过控制模型复杂度来缓解过拟合。

{% asset_img Snipaste_2026-10-08_21-13-59.png "Hexo 博客封面示例" %}



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

### 6.2 Dropout1

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

###  Dropout正则化2

在神经网络中模型参数较多，在数据量不足的情况下，很容易过拟合。Dropout（随机失活）是一个简单有效的正则化方法。

在训练过程中，Dropout的实现是让神经元以超参数p的概率停止工作或者激活被置为0,未被置为0的进行缩放，缩放比例为1/(1-p)。训练过程可以认为是对完整的神经网络的一些子集进行训练，每次基于输入数据只更新子网络的参数。

在测试过程中，随机失活不起作用。

{% asset_img image-20220517141847967.png "Hexo 博客封面示例" %}

- 让神经元以p概率随机死亡, 每批次样本训练模型时, 死亡的神经元都是随机, 防止预测结果受某个神经元影响(防止过拟合)

- p概率->[0.2, 0.5], 简单模型概率低, 复杂模型概率高

- 不失活的神经元计算结果除以(1-p), 让训练时输出结果和测试时(dropout不生效)结果一致

  - 训练模型 -> model.train()
  - 测试模型 -> model.eval()

- dropout是在激活层后使用

  ```python
  import torch
  import torch.nn as nn
  
  
  # dropout随机失活: 每批次样本训练时,随机让一部分神经元死亡,防止一些特征对结果影响大(防止过拟合)
  def dm01():
  	# todo:1-创建隐藏层输出结果
  	# float(): 转换成浮点类型张量
  	t1 = torch.randint(low=0, high=10, size=(1, 4)).float()
  	print('t1->', t1)
  	# todo:2-进行下一层加权求和计算
  	linear1 = nn.Linear(in_features=4, out_features=4)
  	l1 = linear1(t1)
  	print('l1->', l1)
  	# todo:3-进行激活值计算
  	output = torch.sigmoid(l1)
  	print('output->', output)
  	# todo:4-对激活值进行dropout处理  训练阶段
  	# p: 失活概率
  	dropout = nn.Dropout(p=0.4)
  	d1 = dropout(output)
  	print('d1->', d1)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

课堂代码：
```python
"""
案例:
    代码演示 随机失活.

正则化的作用:
    缓解模型的过拟合情况.

正则化的方式:
    L1正则化: 权重可以变为0, 相当于: 降维.
    L2正则化: 权重可以无限接近0
    DropOut: 随机失活, 每批次样本训练时, 随机让一部分神经元死亡, 防止一些特征对结果的影响较大(防止过拟合)
    BN(批量归一化): ...
"""

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 演示: 随机失活(DropOut)
def dm01():
    # 1. 创建隐藏层输出结果.
    t1 = torch.randint(0, 10, size=(1, 4)).float()
    print(f't1: {t1}')      # t1: tensor([[0., 5., 6., 3.]])

    # 2. 进行下一层 加权求和 和 激活函数计算.
    # 2.1 创建全连接层(充当线性层)
    # 参1: 输入特征维度, 参2: 输出特征维度.
    linear1 = nn.Linear(4, 5)

    # 2.2 加权求和.
    l1 = linear1(t1)
    print(f'l1: {l1}')

    # 2.3 激活函数.
    output = torch.relu(l1)
    print(f'output: {output}')

    # 3. 对激活值进行随机失活dropout处理 -> 只有训练阶段有, 测试阶段没有.
    dropout = nn.Dropout(p=0.5) # 每个神经元都有50%的概率被 kill.
    # 具体的 随机失活动作.
    d1 = dropout(output)
    print(f'd1(随机失活后的数据): {d1}')        # 未被失活的进行缩放, 缩放比例为: 1 / (1 - p) = 2


# 2. 测试
if __name__ == '__main__':
    dm01()
```

{% asset_img Snipaste_2026-10-08_21-30-18.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-08_21-40-37.png "Hexo 博客封面示例" %}

### 6.3 批量归一化1

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

### 批量归一正则化2(Batch Normalization)

以批次为单位，对不同批次的数据做处理回到均值0、方差1，减少批次之间的差距（内部协方差）

BN能减少“内部协方差偏移
深层网络在反向传播时，浅层参数的更新会导致深层输入分布不断变化，深层网络需要不断去适应这种变化。BN强行把每层输入的分布拉回到均值0、方差1（然后再通过γ,β 调整），使得每层的输入分布相对稳定，深层网络不需要频繁适应新分布，从而加速训练。

加速训练：
在没有批量归一化的情况下，神经网络的训练通常会变慢，尤其是深度网络。因为在每层的训练过程中，输入数据的分布（特别是前几层）会不断变化，这会导致网络学习速度缓慢。

批量归一化通过确保每层的输入数据在训练时分布稳定，有效提高了学习率的上限，加速了网络的收敛过程。（补全后半句）

起到正则化作用：
批量归一化可以视作一种正则化方法，因为它在每次训练迭代中仅使用当前小批次（mini-batch）的数据来计算均值和方差，这引入了轻微的噪声。同批（指同一个批次内），批次较小的均值和方差估计会更加不准确），使得模型不容易过度依赖特定的训练样本，从而产生类似Dropout的正则化效果，减少了对其他正则化技术（如Dropout）的需求。

提升泛化能力： 由于其正则化效果，批量归一化能帮助网络在测试集上表现得更加稳定，从而提升模型的整体泛化能力。





通常用在计算机视觉领域，图像的像素差别通常很大，归一化会使他们稳定在一个区间

{% asset_img Snipaste_2026-10-08_21-40-37.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-08_21-58-28.png "Hexo 博客封面示例" %}

- 计算每个batch样本的均值和标准差, 利用均值和标准差计算出标准化的值

- 每个batch的均值和标准差都不一样, 会引入噪声样本数据, 降低训练模型效果(防止过拟合)

- 引入两个自学习的γ和β参数, 让每层的样本分布不一样(每层的激活函数可以不一样)

- 加速模型训练效果, 数据分布越均匀, 加权求和结果落入到合理区间(导数最大)

- 训练时进行标准化, 测试时不进行标准化

  ```python
  """
  正则化: 每批样本的均值和方差不一样, 引入噪声样本
  加快模型收敛: 样本标准化后, 落入激活函数的合理区间, 导数尽可能最大
  """
  import torch
  import torch.nn as nn
  
  
  # nn.BatchNorm1d(): 处理一维样本, 每批样本数最少是2个, 否则无法计算均值和标准差
  # nn.BatchNorm2d(): 处理二维样本, 图像(每个通道由二维矩阵组成), 计算二维矩阵每列均值和标准差
  # nn.BatchNorm3d(): 处理三维样本, 视频
  # 处理二维数据
  def dm01():
  	# todo:1-创建图像样本数据集 2个通道,每个通道3*4列特征图, 卷积层处理的特征图样本
  	# 数据集只有一张图像, 图像是由2个通道组成, 每个通道由3*4像素矩阵
  	input_2d = torch.randn(size=(1, 2, 3, 4))
  	print('input_2d->', input_2d)
  	# todo:2-创建BN层, 标准化 ->一定是在激活函数前进行标准化
  	# num_features: 输入样本的通道数
  	# eps: 小常数, 避免除0
  	# momentum: 指数移动加权平均值
  	# affine: 默认True, 引入可学习的γ和β参数
  	bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
  	ouput_2d = bn2d(input_2d)
  	print('ouput_2d->', ouput_2d)
  
  # 处理一维数据
  def dm02():
  	# 创建样本数据集
  	input_1d = torch.randn(size=(2, 2))
  	# 创建线性层
  	linear1 = nn.Linear(in_features=2, out_features=4)
  	l1 = linear1(input_1d)
  	print('l1->', l1)
  	# 创建BN层
  	bn1d = nn.BatchNorm1d(num_features=4)
  	# 对线性层的结果进行标准化处理
  	output_1d = bn1d(l1)
  	print('output_1d->', output_1d)
  
  
  
  if __name__ == '__main__':
  	# dm01()
  	dm02()
  ```

课堂代码：
```python
"""
案例:
    代码演示批量归一化,  它(批量归一化)也属于正则化的一种, 也是用于 缓解模型的 过拟合情况的.

批量归一化:
    思路:
        先对数据做标准化(会丢失一些信息), 然后再对数据做 缩放(λ, 理解为: w权重) 和 平移(β, 理解为: b偏置), 再找补回一些信息.
    应用场景:
        批量归一化在计算机视觉领域使用较多.

        BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
        BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
        BatchNorm3d：主要用于三维卷积神经网络 (3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。
"""

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 处理 二维数据.
def dm01():
    # 1. 创建图像样本数据.
    # 1张图片, 2个通道, 3行4列(像素点)
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print(f'input_2d: {input_2d}')

    # 2. 创建批量归一化层(BN层)
    # 参1: 输入特征数 = 图片的通道数.
    # 参2: 噪声值(小常数), 默认为1e-5.防止分母变为0
    # 参3: 动量值, 用于计算移动平局统计量的  动量值.
    # 参4: 表示使用可学习的变换参数(λ, β) 对归一化(标准化)后的数据进行 缩放和平移.
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    # 3. 对数据进行 批量归一化处理.
    output_2d = bn2d(input_2d)
    print(f'output_2d: {output_2d}')


# 2. 定义函数, 处理: 一维数据.
def dm02():
    # 1. 创建样本数据.
    # 2行2列, 2条样本, 每个样本有2个特征
    input_1d = torch.randn(size=(2, 2))
    print(f'input_1d: {input_1d}')

    # 2. 创建线性层.
    linear1 = nn.Linear(2, 4)

    # 3. 对数据进行 线性变换.
    l1 = linear1(input_1d)
    print(f'l1: {l1}')

    # 4. 创建批量归一化层.
    bn1d = nn.BatchNorm1d(num_features=4)
    # 5. 对线性处理结果l1 进行 批量归一化处理.
    output_1d = bn1d(l1)
    print(f'output_1d: {output_1d}')



# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()

```

### 6.4 其他正则化策略

- **早停法**：在验证集损失开始上升时停止训练，防止过拟合。
- **数据增强**：通过对训练样本进行随机变换（旋转、翻转、裁剪等）增加数据多样性。
- **L1正则化**：使用权重绝对值之和作为惩罚项，倾向于产生稀疏解。


## 实战案例

## 七、实战案例

### 案例需求

- 分类问题 0,1,2,3 四个类别
- 实现步骤
  - 准备数据集 -> 数据集分割, 转换成张量数据集
  - 构建神经网络模型 -> 继承nn.module
  - 模型训练
  - 模型评估

### 构建张量数据集

```python
# 导入相关模块
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


# todo:1-构建数据集
def create_dataset():
    print('===========================构建张量数据集对象===========================')
	# todo:1-1 加载csv文件数据集
	data = pd.read_csv('data/手机价格预测.csv')
	print('data.head()->', data.head())
	print('data.shape->', data.shape)
	# todo:1-2 获取x特征列数据集和y目标列数据集
	# iloc属性 下标取值
	x, y = data.iloc[:, :-1], data.iloc[:, -1]
	# 将特征列转换成浮点类型
	x = x.astype(np.float32)
	print('x->', x.head())
	print('y->', y.head())
	# todo:1-3 数据集分割 8:2
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
	# todo:1-4 数据集转换成张量数据集
	# x_train,y_train类型是df对象, df不能直接转换成张量对象
	# x_train.values():获取df对象的数据值, 得到numpy数组
	# torch.tensor(): numpy数组对象转换成张量对象
	train_dataset = TensorDataset(torch.tensor(data=x_train.values), torch.tensor(data=y_train.values))
	valid_dataset = TensorDataset(torch.tensor(data=x_valid.values), torch.tensor(data=y_valid.values))
	# todo:1-5 返回训练数据集, 测试数据集, 特征数, 类别数
	# shape->(行数, 列数) [1]->元组下标取值
	# np.unique()->去重 len()->去重后的长度 类别数
	print('x.shape[1]->', x.shape[1])
	print('len(np.unique(y)->', len(np.unique(y)))
	return train_dataset, valid_dataset, x.shape[1], len(np.unique(y))


if __name__ == '__main__':
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
```

### 构建分类神经网络模型

```python
# todo:2-构建神经网络分类模型
class PhonePriceModel(nn.Module):
	print('===========================构建神经网络分类模型===========================')
	# todo:2-1 构建神经网络  __init__()
	def __init__(self, input_dim, output_dim):
		# 继承父类的构造方法
		super().__init__()
		# 第一层隐藏层
		self.linear1 = nn.Linear(in_features=input_dim, out_features=128)
		# 第二层隐藏层
		self.linear2 = nn.Linear(in_features=128, out_features=256)
		# 输出层
		self.output = nn.Linear(in_features=256, out_features=output_dim)
	# todo:2-2 前向传播方法 forward()
	def forward(self, x):
		# 第一层隐藏层计算
		x = torch.relu(input=self.linear1(x))
		# 第二层隐藏层计算
		x = torch.relu(input=self.linear2(x))
		# 输出层计算
		# 没有进行softmax激活计算, 后续创建损失函数时CrossEntropyLoss=softmax+损失计算
		output = self.output(x)
		return output
# todo:3-模型训练
# todo:4-模型评估

if __name__ == '__main__':
	# 创建张量数据集对象
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# 创建模型对象
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# 计算模型参数
	# input_size: 输入层样本形状
	summary(model, input_size=(16, input_dim))
```
###  模型训练

```python
# todo:3-模型训练
def train(train_dataset, input_dim, class_num):
	print('===========================模型训练===========================')
	# todo:3-1 创建数据加载器 批量训练
	dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
	# todo:3-2 创建神经网络分类模型对象, 初始化w和b
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	print("======查看模型参数w和b======")
	for name, parameter in model.named_parameters():
		print(name, parameter)
	# todo:3-3 创建损失函数对象 多分类交叉熵损失=softmax+损失计算
	criterion = nn.CrossEntropyLoss()
	# todo:3-4 创建优化器对象 SGD
	optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
	# todo:3-5 模型训练 min-batch 随机梯度下降
	# 训练轮数
	num_epoch = 50
	for epoch in range(num_epoch):
		# 定义变量统计每次训练的损失值, 训练batch数
		total_loss = 0.0
		batch_num = 0
		# 训练开始的时间
		start = time.time()
		# 批次训练
		for x, y in dataloader:
			# 切换模型模式
			model.train()
			# 模型预测 y预测值
			y_pred = model(x)
			# print('y_pred->', y_pred)
			# 计算损失值
			loss = criterion(y_pred, y)
			# print('loss->', loss)
			# 梯度清零
			optimizer.zero_grad()
			# 计算梯度
			loss.backward()
			# 更新参数 梯度下降法
			optimizer.step()
			# 统计每次训练的所有batch的平均损失值和和batch数
			# item(): 获取标量张量的数值
			total_loss += loss.item()
			batch_num += 1
		# 打印损失变换结果
		print('epoch: %4s loss: %.2f, time: %.2fs' % (epoch + 1, total_loss / batch_num, time.time() - start))
	# todo:3-6 模型保存, 将模型参数保存到字典, 再将字典保存到文件
	torch.save(model.state_dict(), 'model/phone.pth')


if __name__ == '__main__':
	# 创建张量数据集对象
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# 创建模型对象
	# model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# 计算模型参数
	# input_size: 输入层样本形状
	# summary(model, input_size=(16, input_dim))
	# 模型训练
	train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
```

### 模型评估

```python
# todo:4-模型评估
def test(valid_dataset, input_dim, class_num):
	# todo:4-1 创建神经网络分类模型对象
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# todo:4-2 加载训练模型的参数字典
	model.load_state_dict(torch.load(f='model/phone.pth'))
	# todo:4-3 创建测试集数据加载器
	# shuffle: 不需要为True, 预测, 不是训练
	dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
	# todo:4-4 定义变量, 初始值为0, 统计预测正确的样本个数
	correct = 0
	# todo:4-5 按batch进行预测
	for x, y in dataloader:
		print('y->', y)
		# 切换模型模式为预测模式
		model.eval()
		# 模型预测 y预测值 -> 输出层的加权求和值
		output = model(x)
		print('output->', output)
		# 根据加权求和值得到类别, argmax() 获取最大值对应的下标就是类别 y->0,1,2,3
		# dim=1:一行一行处理, 一个样本一个样本
		y_pred = torch.argmax(input=output, dim=1)
		print('y_pred->', y_pred)
		# 统计预测正确的样本个数
		print(y_pred == y)
		# 对布尔值求和, True->1 False->0
		print((y_pred == y).sum())
		correct += (y_pred == y).sum()
		print('correct->', correct)
	# 计算预测精度 准确率
	print('Acc: %.5f' % (correct.item() / len(valid_dataset)))


if __name__ == '__main__':
	# 创建张量数据集对象
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# 创建模型对象
	# model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# 计算模型参数
	# input_size: 输入层样本形状
	# summary(model, input_size=(16, input_dim))
	# 模型训练
	# train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
	# 模型评估
	test(valid_dataset=valid_dataset, input_dim=input_dim, class_num=class_num)
```

课堂代码：
```python
"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例.

背景:
    基于手机的20列特征 -> 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用 深度学习做(推荐)

ANN案例的实现步骤:
    1. 构建数据集.
    2. 搭建神经网络.
    3. 模型训练.
    4. 模型测试.
"""

# 导包
import torch                                    # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         # 数据加载器.
import torch.nn as nn                           # neural network, 封装了神经网络的各种操作
import torch.optim as optim                     # 优化器
from sklearn.model_selection import train_test_split    # 训练集和测试集的划分
import matplotlib.pyplot as plt                 # 绘图
import numpy as np                              # 数组(矩阵)操作
import pandas as pd                             # 数据处理
import time                                     # 时间模块

# todo 1. 定义函数, 构建数据集.
def create_dataset():
    # 1. 加载csv文件数据集.
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f'data: {data.head()}')
    # print(f'data: {data.shape}')    # (2000, 21)

    # 2. 获取x特征列 和 y标签列.
    x, y = data.iloc[:, :-1], data.iloc[:, -1] # 切片出特征列
    # print(f'x: {x.head()}, {x.shape}')  # (2000, 20)
    # print(f'y: {y.head()}, {y.shape}')  # (2000, )

    # 3. 把特征列转成浮点型.
    x = x.astype(np.float32)
    # print(f'x: {x.head()}, {x.shape}')   # (2000, 20)

    # 4. 切分训练集和测试集.
    # 参1: 特征, 参2: 标签, 参3: 测试集所占比例, 参4: 随机种子, 参5: 样本的分布(即: 参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 5. 把数据集封装成 张量数据集.  思路: 数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    # 6. 返回结果                         20(充当 输入特征数)     4(充当 输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# todo 2. 搭建神经网络.
# todo 3. 模型训练.
# todo 4. 模型测试.


# todo 5. 测试
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集 数据集对象: {train_dataset}')
    print(f'测试集 数据集对象: {test_dataset}')
    print(f'输入特征数: {input_dim}')    # 20
    print(f'输出标签数: {output_dim}')   # 4
```

### 网络性能优化

- 输入层数据进行标准化
- 神经网络层数增加, 神经元个数增加
- 梯度下降优化方法由SGD调整为Aam
- 学习率由1e-3调整为1e-4
- 正则化
- 增加训练轮数
- ...

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
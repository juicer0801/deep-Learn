---
title: 微积分
date: 2026-09-26 19:35:17
tags:
---
# 第 6 节：微积分，模型怎么学习

---

## 6.1 导数与微分

### 导数是什么

导数衡量的是**变化率**。函数 \(f(x)\) 在 \(x\) 点的导数记作 \(f'(x)\) 或 \(\frac{df}{dx}\)，定义是：

\[
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\]


如果 \(f'(x) > 0\)，说明 \(x\) 增大时 \(f(x)\) 也增大。如果 \(f'(x) < 0\)，说明 \(x\) 增大时 \(f(x)\) 减小。如果 \(f'(x) = 0\)，说明 \(x\) 处是极值点或者鞍点。

几何上，导数就是曲线在这一点切线的斜率。

```python
import torch

# 用 PyTorch 验证导数
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # tensor(4.)  因为 f'(x) = 2x，在 x=2 处是 4
```

### 微分是什么

微分和导数严格讲有区别。

微分 \(df\) 指的是函数变化的**线性主部**：

\[
df = f'(x) \, dx
\]

\(dx\) 是自变量的微小变化，\(df\) 是函数值的近似变化。导数 \(f'(x)\) 是微分和自变量微分的比值，所以也叫“微商”。

在深度学习里，基本是一回事。**注意**：导数是局部的，它只在某一点附近有效。理解梯度下降有用。

### 深度学习中为什么需要导数

模型训练的目标是最小化损失函数 \(L(\theta)\)，\(\theta\) 是模型参数。
即沿导数相反的方向调整参数。

如果 \(L'(\theta) > 0\)，说明 \(\theta\) 增大时损失也增大，那就应该减小 \(\theta\)。如果 \(L'(\theta) < 0\)，说明 \(\theta\) 增大时损失减小，那就应该增大 \(\theta\)。

这也就是梯度下降原理：**最小化一个损失函数**

```python
# 手动梯度下降
x = torch.tensor(2.0, requires_grad=True)
lr = 0.1

for i in range(5):
    y = x ** 2
    y.backward()
    print(f"x={x.item():.4f}, y={y.item():.4f}, grad={x.grad.item():.4f}")
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()
```

输出：

```
x=2.0000, y=4.0000, grad=4.0000
x=1.6000, y=2.5600, grad=3.2000
x=1.2800, y=1.6384, grad=2.5600
x=1.0240, y=1.0486, grad=2.0480
x=0.8192, y=0.6711, grad=1.6384
```

每次 \(x\) 都往导数反方向走一点，\(y\) 越来越小。这就是训练的过程。

### 常用导数公式（高中学的）

这些是基础，得记住：

| 函数 | 导数 |
|---|---|
| \(x^n\) | \(n x^{n-1}\) |
| \(e^x\) | \(e^x\) |
| \(\ln x\) | \(1/x\) |
| \(\sin x\) | \(\cos x\) |
| \(\cos x\) | \(-\sin x\) |
| 常数 \(C\) | 0 |

还有几条运算法则：

- 加法：\((f + g)' = f' + g'\)
- 乘法：\((fg)' = f'g + fg'\)
- 除法：\((f/g)' = (f'g - fg') / g^2\)
- 复合：链式法则

---

## 6.2 偏导数

### 为什么需要偏导数

现实中的函数很少只有一个变量。损失函数 \(L\) 依赖成千上万个参数 \(\theta_1, \theta_2, \ldots, \theta_n\)。
分别知道损失对每个参数的变化率。
这就是偏导数。

偏导数 \(\frac{\partial f}{\partial x_i}\) 的意思是：**只让 \(x_i\) 动，其他变量全部固定，看 \(f\) 怎么变。**

符号从 \(d\) 变成 \(\partial\)

### 举个栗子

\[
f(x, y) = x^2 + y^2 + xy
\]

对 \(x\) 求偏导，把 \(y\) 当常数：

\[
\frac{\partial f}{\partial x} = 2x + y
\]

对 \(y\) 求偏导，把 \(x\) 当常数：

\[
\frac{\partial f}{\partial y} = 2y + x
\]

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
f = x**2 + y**2 + x*y
f.backward()

print(x.grad)   # tensor(4.)  2*1 + 2 = 4
print(y.grad)   # tensor(5.)  2*2 + 1 = 5
```

PyTorch 的 `backward()` 会自动把所有偏导数都算出来，分别存在各个张量的 `.grad` 里。

### 偏导数的几何意义

一维的时候，导数是切线斜率。二维的时候，偏导数是在某个方向上的切线斜率。

想象你站在一座山上。\(x\) 方向是东西，\(y\) 方向是南北。\(\frac{\partial f}{\partial x}\) 是你只往东走时的高度变化率，\(\frac{\partial f}{\partial y}\) 是你只往北走时的高度变化率。

但实际上你不想只往东或只往北走，你想往最陡的方向走。那就需要梯度。

---

## 6.3 梯度

### 梯度的定义

梯度是所有偏导数组成的向量：

\[
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)
\]

李沐大神说：
# 2.4.3 梯度
我们可以连接一个多元函数对其所有变量的偏导数，以得到该函数的梯度（gradient）向量。具体而言，设函数$f: \mathbb{R}^n \to \mathbb{R}$的输入是一个$n$维向量$\boldsymbol{\(\boldsymbol{x}\)}=[\(\boldsymbol{x}\)_1,\(\boldsymbol{x}\)_2,\cdots,\(\boldsymbol{x}\)_n]^\mathrm{T}$，输出是一个标量。函数$f(\boldsymbol{\(\boldsymbol{x}\)})$相对于$\boldsymbol{\(\boldsymbol{x}\)}$的梯度是一个包含$n$个偏导数的向量：

$$
\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} f(\boldsymbol{\(\boldsymbol{x}\)})=\left[\frac{\partial f(\boldsymbol{\(\boldsymbol{x}\)})}{\partial \(\boldsymbol{x}\)_1},\frac{\partial f(\boldsymbol{\(\boldsymbol{x}\)})}{\partial \(\boldsymbol{x}\)_2},\cdots,\frac{\partial f(\boldsymbol{\(\boldsymbol{x}\)})}{\partial \(\boldsymbol{x}\)_n}\right]^\mathrm{T} \tag{2.26}
$$

其中，$\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} f(\boldsymbol{\(\boldsymbol{x}\)})$在没有歧义时通常被$\nabla f(\boldsymbol{\(\boldsymbol{x}\)})$取代。

假设$\boldsymbol{\(\boldsymbol{x}\)}$为$n$维向量，在对多元函数求微分时经常使用以下规则：
- 对于所有$\boldsymbol{A}\in\mathbb{R}^{m\times n}$，都有$\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} \boldsymbol{A}\boldsymbol{\(\boldsymbol{x}\)} = \boldsymbol{A}^\mathrm{T}$；
- 对于所有$\boldsymbol{A}\in\mathbb{R}^{n\times m}$，都有$\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} \boldsymbol{\(\boldsymbol{x}\)}^\mathrm{T}\boldsymbol{A} = \boldsymbol{A}$；
- 对于所有$\boldsymbol{A}\in\mathbb{R}^{n\times n}$，都有$\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} \boldsymbol{\(\boldsymbol{x}\)}^\mathrm{T}\boldsymbol{A}\boldsymbol{\(\boldsymbol{x}\)} = (\boldsymbol{A}+\boldsymbol{A}^\mathrm{T})\boldsymbol{\(\boldsymbol{x}\)}$；
- $\nabla_{\boldsymbol{\(\boldsymbol{x}\)}} \|\boldsymbol{\(\boldsymbol{x}\)}\|^2 = \nabla_{\boldsymbol{\(\boldsymbol{x}\)}} \boldsymbol{\(\boldsymbol{x}\)}^\mathrm{T}\boldsymbol{\(\boldsymbol{x}\)}=2\boldsymbol{\(\boldsymbol{x}\)}$。

同样，对于任何矩阵$\boldsymbol{\(\boldsymbol{X}\)}$，都有$\nabla_{\boldsymbol{\(\boldsymbol{X}\)}} \|\boldsymbol{\(\boldsymbol{X}\)}\|_F^2=2\boldsymbol{\(\boldsymbol{X}\)}$。正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。

---



它是一个**向量**，指向函数增长最快的方向。

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = x[0]**2 + x[1]**2 + x[0]*x[1]
f.backward()

print(x.grad)   # tensor([4., 5.])
```

这里 `x.grad` 就是梯度向量 `[∂f/∂x₀, ∂f/∂x₁] = [4, 5]`。

### 梯度为什么指向最陡方向

这个结论可以用方向导数来证明，但直觉上可以这样理解：

假设你站在点 \(\mathbf{x}\)，想找一个方向 \(\mathbf{v}\)（单位向量），使得函数在这个方向上的变化率最大。方向导数等于 \(\nabla f \cdot \mathbf{v}\)。

两个向量的点积，在方向相同时最大。所以当 \(\mathbf{v}\) 和 \(\nabla f\) 同方向时，变化率最大。

所以梯度方向就是函数上升最快的方向，反梯度方向就是下降最快的方向。

### 梯度下降

梯度下降的更新公式：

\[
\theta \leftarrow \theta - \eta \nabla L(\theta)
\]

\(\eta\) 是学习率。每一步都往梯度的反方向走，损失就会下降。

```python
# 用梯度下降最小化 f(x,y) = x^2 + y^2 + xy
x = torch.tensor([1.0, 2.0], requires_grad=True)
lr = 0.1

for i in range(10):
    f = x[0]**2 + x[1]**2 + x[0]*x[1]
    f.backward()
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()
    print(f"step {i}: x={x.tolist()}, f={f.item():.4f}")
```

这个函数的最小值在原点。每次沿梯度反方向走一步，\(x\) 就会越来越接近原点。

### 梯度为零的点

梯度为零的点叫**驻点**。驻点可能是：

- **极小值**：周围都比它大（这是我们要找的）
- **极大值**：周围都比它小。
- **鞍点**：一个方向上是极小值，另一个方向上是极大值。

深度学习里，鞍点比局部极小值更常见。因为参数维度太高了，要所有方向都是极小值太难了。好在鞍点附近梯度很小，优化器通常能慢慢走出去。

### 注意：梯度是局部信息

梯度只告诉你“在当前点附近，往哪个方向走下降最快”。它不知道远处的情况。

这就是为什么学习率不能太大。步子迈太大，可能直接跨过最小值，甚至跑到更高的地方。但学习率太小，又走得太慢。


---

## 6.4 链式法则

### 链式法则是什么

链式法则处理的是复合函数求导。

如果 \(y = f(u)\)，\(u = g(x)\)，那么：

\[
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
\]

复合函数的导数，等于外层函数的导数乘以内层函数的导数。

多层嵌套也一样：

\[
y = f(g(h(x))) \implies \frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)
\]

### 为什么链式法则是深度学习的核心

神经网络就是一堆函数的嵌套。一层接一层，每一层都是对上一层输出的变换。

\[
\mathbf{h}_1 = f_1(\mathbf{x})
\]
\[
\mathbf{h}_2 = f_2(\mathbf{h}_1)
\]
\[
\vdots
\]
\[
L = \text{loss}(f_n(\mathbf{h}_{n-1}), y)
\]

损失 \(L\) 对第一层参数 \(W_1\) 的偏导，要用链式法则一层一层往回传：

\[
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{h}_n} \cdot \frac{\partial \mathbf{h}_n}{\partial \mathbf{h}_{n-1}} \cdots \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}
\]

**反向传播就是链式法则的系统化应用。** 从损失开始，一层一层往回想，把梯度传回去。

### 一个手算例子

设：

\[
z = (2x + 3)^2
\]

令 \(u = 2x + 3\)，则 \(z = u^2\)。

链式法则：

\[
\frac{dz}{dx} = \frac{dz}{du} \cdot \frac{du}{dx} = 2u \cdot 2 = 4u = 4(2x + 3)
\]

当 \(x = 1\) 时，\(u = 5\)，\(\frac{dz}{dx} = 20\)。

```python
x = torch.tensor(1.0, requires_grad=True)
z = (2*x + 3) ** 2
z.backward()
print(x.grad)   # tensor(20.)
```

### 链式法则的计算图视角

PyTorch 的自动微分就是靠计算图实现的。

前向传播的时候，PyTorch 记录每一步运算，构建一张图。反向传播的时候，从损失节点开始，沿图反向走，每经过一个节点就应用一次链式法则。

```python
x = torch.tensor(2.0, requires_grad=True)

# 前向传播：每一步都被记录
a = x * 3        # a = 6
b = a + 1        # b = 7
c = b ** 2       # c = 49

# 反向传播：沿计算图往回走
c.backward()

# dc/dx = dc/db * db/da * da/dx
#       = 2b * 1 * 3
#       = 2*7 * 3
#       = 42
print(x.grad)    # tensor(42.)
```

计算图长这样：

```
x → (×3) → a → (+1) → b → (²) → c
```

反向走：

- \(c\) 对 \(b\) 的梯度：\(2b = 14\)
- \(b\) 对 \(a\) 的梯度：\(1\)
- \(a\) 对 \(x\) 的梯度：\(3\)
- 乘起来：\(14 \times 1 \times 3 = 42\)

这就是反向传播。就是链式法则。

完整例子
5.3 实现完整反向传播
在PyTorch中实现：

```python

import torch
​
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
x3 = torch.tensor(4.0, requires_grad=True)
​
y = (x1 + x2) * x3
y.backward()
​
print(x1.grad)  # 4.0
print(x2.grad)  # 4.0
print(x3.grad)  # 5.0
```

### 链式法则在向量情况下的形式

标量情况简单，但深度学习里都是向量和矩阵。向量情况下的链式法则要用雅可比矩阵。

如果 \(\mathbf{y} = f(\mathbf{x})\)，\(\mathbf{z} = g(\mathbf{y})\)，那么：

\[
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
\]

这里的乘法是矩阵乘法，\(\frac{\partial \mathbf{z}}{\partial \mathbf{y}}\) 和 \(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\) 都是雅可比矩阵。

雅可比矩阵 \(J\) 的第 \(i\) 行第 \(j\) 列是 \(\frac{\partial y_i}{\partial x_j}\)。

实际写代码的时候你不需要手动算雅可比，PyTorch 的 `backward()` 即可。
---

## 6.5 把四个概念关系

**导数**告诉我们单个变量怎么变。  
**偏导数**告诉我们多变量里，每个变量单独怎么变。  
**梯度**把所有偏导数打包成一个向量，指向最陡上升方向。  
**链式法则**告诉我们复合函数怎么求导，也就是梯度怎么从最后一层传回第一层。

深度学习训练就是一个循环：

1. 前向传播：数据过一遍网络，算出预测和损失。
2. 反向传播：用链式法则从损失开始，一层层算梯度。
3. 更新参数：沿梯度反方向走一步。
4. 重复。

这个循环的核心就是微积分。没有微积分，就没有反向传播，就没有深度学习。

---

## 6.6 一些实际会用到的导数

最后列几个深度学习里高频出现的导数。

**Sigmoid**：

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]
\[
\sigma'(x) = \sigma(x)(1 - \sigma(x))
\]



**Tanh**：

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]
\[
\tanh'(x) = 1 - \tanh^2(x)
\]

**ReLU**：

\[
\text{ReLU}(x) = \max(0, x)
\]
\[
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}
\]

ReLU 在 0 处不可导，但实际用的时候没人管这个点，随便定一个值就行（通常用 0 或 1）。

**Softmax + 交叉熵**：预测概率减去真实标签：

\[
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
\]

这也是为什么分类任务我喜欢用 softmax + 交叉熵，梯度好算，数值也稳定。

---

## 本节小结

- **导数**是变化率，告诉我们函数在某点附近怎么变。
- **偏导数**是多变量函数里，只对一个变量求导，其他固定。
- **梯度**是所有偏导数组成的向量，指向函数上升最快的方向。
- **链式法则**是复合函数求导的规则，是反向传播的数学基础。
- **梯度下降**沿梯度反方向更新参数，是训练的基本方法。
- **反向传播**就是链式法则在计算图上的系统化应用。

## 6.6 扩展：微积分在深度学习中的高级应用

前面讲的导数、偏导数、梯度、链式法则，是最基础的部分。实际做深度学习，还会碰到一些更深入的东西。
### 雅可比矩阵

前面说链式法则的时候提过一嘴，这里展开讲。

如果函数是 \(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m\)，输入是 n 维向量，输出是 m 维向量，那它的导数不能只用一组偏导数表示。每一个输出对每一个输入都有一个偏导数，总共有 \(m \times n\) 个。把这些偏导数排成一个矩阵，就是**雅可比矩阵**：

\[
J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
\]

第 i 行第 j 列是 \(\frac{\partial f_i}{\partial x_j}\)。

**在神经网络里，每一层都是一个向量到向量的函数。** 一层的雅可比矩阵描述了这一层的输出对输入的敏感度。反向传播的本质，就是把损失对输出的梯度，乘以这一层的雅可比矩阵，得到损失对输入的梯度。

```python
# 用 PyTorch 的 jacobian 函数算雅可比矩阵
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] + x[1]**2])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)
print(J)
# tensor([[2., 1.],
#         [1., 4.]])
```

验证一下：\(f_1 = x_1^2 + x_2\)，\(\partial f_1/\partial x_1 = 2x_1 = 2\)，\(\partial f_1/\partial x_2 = 1\)。\(f_2 = x_1 + x_2^2\)，\(\partial f_2/\partial x_1 = 1\)，\(\partial f_2/\partial x_2 = 2x_2 = 4\)。

**不显式算雅可比矩阵** 一个层如果有 1000 维输入、1000 维输出，雅可比矩阵就是 100 万个元素。反向传播用的是**向量-雅可比积**（VJP），不需要显式构造矩阵，只需要算“雅可比矩阵转置乘以一个向量”，计算量小得多。。

### 海森矩阵

雅可比矩阵是“一阶导数”的推广。**海森矩阵**是“二阶导数”的推广。

对于标量函数 \(f: \mathbb{R}^n \to \mathbb{R}\)，海森矩阵 \(H\) 的第 i 行第 j 列是：

\[
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
\]

它描述的是函数的曲率——梯度本身是怎么变化的。

```python
from torch.autograd.functional import hessian

def f(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

x = torch.tensor([1.0, 2.0])
H = hessian(f, x)
print(H)
# tensor([[2., 1.],
#         [1., 2.]])
```

**海森矩阵在深度学习的作用**

- 如果海森矩阵在某个方向的特征值很大，说明那个方向曲率大，梯度变化快，学习率要小一点。
- 如果特征值很小，说明那个方向很平，可以走大步一点。

**海森矩阵无法直接全应用** n 个参数，海森矩阵是 \(n \times n\)。一个百万参数的模型，海森矩阵就是一万亿个元素，根本存不下。

所以实际用的是海森矩阵的**近似**。Adam、RMSProp 这些自适应学习率方法，本质上是在用梯度平方的移动平均来近似海森矩阵的对角线。K-FAC、Sophia 这些二阶优化器，用的是更精细的低秩近似。

### 自动微分的两种模式

PyTorch 的 `backward()` 用的是**反向模式**自动微分。但自动微分不止这一种模式。

**前向模式**：从输入开始，沿计算图正向走，每一步算“输出对输入的导数”。一次前向能算出所有输出对一个输入的导数。

**反向模式**：从输出开始，沿计算图反向走，每一步算“输出对中间变量的导数”。一次反向能算出所有输出对所有输入的导数。

区别：

- 输入 n 维，输出 m 维。
- 前向模式一次算一列雅可比，要算 n 次才能得到完整雅可比。
- 反向模式一次算一行雅可比，要算 m 次才能得到完整雅可比。

**深度学习里，输入维度极高（百万参数），输出维度很低（一个损失值）。** 所以反向模式效率高得多——一次反向传播就能算出损失对所有参数的梯度。

如果输出维度比输入维度高，前向模式反而更合适。JAX 的 `jacfwd` 和 `jacrev` 就是这两种模式。

### 梯度消失与梯度爆炸

最经典的问题之一。

反向传播是连乘：

\[
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{h}_n} \cdot \frac{\partial \mathbf{h}_n}{\partial \mathbf{h}_{n-1}} \cdots \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}
\]

如果每一层的雅可比矩阵的范数都小于 1，连乘之后趋近于 0，梯度就消失了。前面的层几乎收不到梯度，参数更新不动。

如果每一层的雅可比矩阵的范数都大于 1，连乘之后指数增长，梯度就爆炸了。参数更新幅度巨大，训练直接崩掉。

```python
# 演示梯度消失
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(10, 10))
            layers.append(nn.Sigmoid())  # sigmoid 导数最大只有 0.25
        layers.append(nn.Linear(10, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 20 层 sigmoid 网络
model = DeepNet(20)
x = torch.randn(1, 10)
loss = model(x).sum()
loss.backward()

# 看第一层的梯度
first_layer_grad = model.net[0].weight.grad
print(f"第一层梯度范数: {first_layer_grad.norm().item():.2e}")
# 会非常小，接近 0
```

**解决方法**

- **梯度消失**：换激活函数（ReLU 代替 sigmoid）、用残差连接、用 BatchNorm、用 LSTM/GRU 的门控机制。
- **梯度爆炸**：梯度裁剪、权重初始化用更小的方差、用 LayerNorm（可以看看李沐大神的讲解）。

### 梯度裁剪

梯度爆炸的常用解法是梯度裁剪（gradient clipping）。思路很简单：如果梯度的范数超过某个阈值，就把它缩回去。

```python
loss.backward()

# 计算所有参数梯度的总范数
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"裁剪前范数: {total_norm:.4f}")

optimizer.step()
```

`clip_grad_norm_` 会把所有参数的梯度当成一个长向量，算它的 L2 范数。如果超过 `max_norm`，所有梯度按比例缩小，保持方向不变。

**为什么保持方向不变？** 因为方向是梯度下降的依据，缩放只影响步长。如果方向变了，更新方向就不对了。

**注意**：`clip_grad_norm_` 要在 `loss.backward()` 之后、`optimizer.step()` 之前调用。顺序错了就完蛋了。

### 二阶优化方法

梯度下降是一阶方法，只用了一阶导数。二阶方法用海森矩阵，收敛更快，但计算代价高。

**牛顿法**的更新公式：

\[
\theta \leftarrow \theta - H^{-1} \nabla L
\]

用海森矩阵的逆来调整更新方向。在一维情况下，牛顿法相当于用二次函数去拟合当前点，然后直接跳到二次函数的最小值。收敛速度是二次的，比梯度下降快得多。

**但牛顿法在深度学习里基本不可用**，因为：

1. 海森矩阵太大，存不下。
2. 求逆的复杂度是 \(O(n^3)\)，百万参数根本算不动。
3. 海森矩阵可能不是正定的，逆不一定存在。

**实际用的是拟牛顿法**，比如 L-BFGS。它不显式构造海森矩阵，而是用最近几次的梯度变化来估计曲率。L-BFGS 在参数量不大的时候效果不错，但在深度学习里因为 batch 训练的随机性，表现不稳定。

**真正在深度学习里广泛使用的“二阶信息”是自适应的**。Adam 用梯度平方的指数移动平均来近似每个参数方向的曲率，为每个参数单独调整学习率。这是对海森矩阵对角线的一个粗糙但有效的近似。

### 伴随方法与神经常微分方程

这块比较前沿（我还在了解）。

**伴随方法**（adjoint method）是求解微分方程约束下优化问题的经典方法。它的核心思想和反向传播一样：先正向求解状态，再从终点反向传播梯度。

**神经常微分方程**（Neural ODE）把神经网络的层数看成连续的。不是离散的“第 1 层、第 2 层、……”，而是用一个微分方程描述隐藏状态随“深度”的变化：

\[
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)
\]

前向传播就是解这个微分方程。反向传播用伴随方法，不需要存储中间状态，内存占用是常数级的。

```python
# 概念示意，需要 torchdiffeq 库
from torchdiffeq import odeint

class ODEBlock(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        # 从 t=0 积分到 t=1
        t = torch.tensor([0.0, 1.0])
        return odeint(self.func, x, t)[-1]
```

Neural ODE 的好处是内存高效、可以处理不规则时间序列、理论上更优雅。缺点是训练慢，因为要数值解微分方程。

这块属于了解即可，实际项目里用得不多。

### 隐函数定理与深度均衡模型

**深度均衡模型**（Deep Equilibrium Model, DEQ）是另一个前沿方向。它不显式定义多少层，而是找一个不动点：

\[
\mathbf{z}^* = f(\mathbf{z}^*, \mathbf{x})
\]

前向传播就是迭代直到收敛。反向传播不用存中间状态，而是用隐函数定理直接算梯度。

隐函数定理说的是：如果 \(F(\mathbf{z}, \theta) = 0\) 定义了隐函数 \(\mathbf{z}(\theta)\)，那么：

\[
\frac{\partial \mathbf{z}}{\partial \theta} = -\left( \frac{\partial F}{\partial \mathbf{z}} \right)^{-1} \frac{\partial F}{\partial \theta}
\]

这个公式让 DEQ 可以用常数级内存做反向传播。DEQ 在理论上可以表示无限深的网络，但实际训练稳定性还需要进一步研究。

以下还有，但我不会
###  流形优化与梯度下降
在深度学习中，参数空间实际上是高维流形。流形优化考虑了参数空间的几何结构，可以处理约束优化问题。

### 变分方法与最大期望
EM算法基于变分思想，优化期望对数似然函数。在深度学习中，变分自编码器（VAE）就是基于这种变分推断思想的。

### 能量基础模型与微积分
Boltzmann机等能量基础模型通过能量函数 $E(x, h)$ 描述数据分布，梯度由玻尔兹曼分布的期望给出：
$$\nabla \log p(x) = \langle \frac{\partial E}{\partial \theta} \rangle_{data} - \langle \frac{\partial E}{\partial \theta} \rangle_{model}$$



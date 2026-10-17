---
title: rnn三行代码的理解
date: 2026-10-10 15:29:04
tags:
---

# 把 RNN 的这几行代码讲透：从“小本本”到张量形状

如果你刚接触 PyTorch 的 RNN，下面这几行代码很容易让人绕晕：

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
x = torch.randn(size=(5, 32, 128))
h0 = torch.randn(size=(1, 32, 256))
output, h1 = rnn(x, h0)
```

明明参数不多，但 `128`、`256`、`1`、`5`、`32` 到处出现，形状还互相关联。  
这篇文章用两种方式讲：

1. **通俗类比**：把它想成一群人记小本本。
2. **严谨**：把每个张量维度对应到 API 参数。

最后再重点拓展：**那些数值相同的参数，到底分别是什么意思。**

---

## 一、一句话总览

> 给一个只有 **1 层**、**单向** 的 RNN，喂入 **5 个时间步**、**32 个样本**、每个样本每步 **128 维** 的输入。  
> RNN 的隐藏记忆是 **256 维**。  
> 初始隐藏状态是 **1 层、32 个样本、256 维记忆**。  
> 输出是 **5 个时间步、32 个样本、256 维记忆**。  
> 最终隐藏状态是 **1 层、32 个样本、256 维记忆**。

默认情况下，PyTorch 的顺序是：

```text
x:      (seq_len, batch, input_size)
h0/h1:  (num_layers * num_directions, batch, hidden_size)
output: (seq_len, batch, hidden_size * num_directions)
```

你的例子里：

```python
x.shape      # (5, 32, 128)
h0.shape     # (1, 32, 256)
output.shape # (5, 32, 256)
h1.shape     # (1, 32, 256)
```

---

## 二、通俗理解：32 个人，5 天，小本本

把 RNN 想成一群人，每个人有一个小本本，用来记记忆。

- **5**：有 5 天，也就是 5 个时间步。
- **32**：有 32 个人，也就是 batch 里有 32 个样本。
- **128**：每个人每天带 128 个数字来。
- **256**：每个人的小本本有 256 个格子，用来记记忆。
- **1**：只有 1 层 RNN，也就是只有一本小本本。

所以：

```python
x = (5, 32, 128)
```

就是：

> 5 天，32 个人，每人每天输入 128 个数字。

```python
h0 = (1, 32, 256)
```

就是：

> 给 32 个人各发一本初始小本本，每本 256 个格子；1 表示只有 1 层。

```python
output = (5, 32, 256)
```

就是：

> RNN 每天都会把每个人当天的小本本内容记下来。  
> 所以有 5 天，每天 32 个人，每人 256 个格子。

```python
h1 = (1, 32, 256)
```

就是：

> 5 天结束后，每个人最后的小本本内容。  
> 1 还是表示只有 1 层。

单层单向时，最后一天的输出就是最终隐藏状态：

```python
output[-1] == h1[0]
```

更准确地说：

```python
torch.allclose(output[-1], h1[0])  # True
```

---

## 三、严谨拆解：每个参数管什么

先看你写的 API：

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
```

它决定了 RNN 的内部结构：

| 参数 | 含义 | 对应维度 |
|---|---|---|
| `input_size=128` | 每个时间步输入的特征维度 | `x` 的最后一维 |
| `hidden_size=256` | 隐藏状态/记忆的维度 | `h0`、`h1`、`output` 的最后一维 |
| `num_layers=1` | RNN 层数 | `h0`、`h1` 的第一维 |
| `batch_first=False` | 默认 batch 不在第一维 | `x` 和 `output` 的第一维是时间 |
| `bidirectional=False` | 默认单向 | 方向数 `num_directions=1` |

RNN 内部每个时间步做类似这样的事：

```text
h_t = tanh( x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh )
```

维度对应：

```text
x_t:     (batch, input_size)      = (32, 128)
h_{t-1}: (batch, hidden_size)     = (32, 256)
W_ih:    (hidden_size, input_size)= (256, 128)
W_hh:    (hidden_size, hidden_size)= (256, 256)
h_t:     (batch, hidden_size)     = (32, 256)
```

你可以验证权重形状：

```python
rnn.weight_ih_l0.shape  # torch.Size([256, 128])
rnn.weight_hh_l0.shape  # torch.Size([256, 256])
```

所以：

- 输入 `128` 维，被映射成隐藏记忆 `256` 维。
- `output` 是把每个时间步的 `h_t` 沿时间维堆起来。
- `h1` 只保存每层最后一个时间步的隐藏状态。

---

## 四、重点：那些数值相同的参数，分别是什么含义？

你代码里有很多重复数字。它们不是随便写的，而是 **形状契约**。

### 1. `128` 出现在哪里？

```python
input_size=128
x = torch.randn(size=(5, 32, 128))
```

`128` 表示：

> 每个时间步，每个样本输入的特征维度。

它出现在：

- `nn.RNN(input_size=128)`
- `x` 的最后一维 `x.shape[-1]`

如果你把 `input_size` 改成 `64`，那么 `x` 的最后一维也必须是 `64`，否则会报错。

---

### 2. `256` 出现在哪里？

```python
hidden_size=256
h0 = torch.randn(size=(1, 32, 256))
output, h1 = rnn(x, h0)
```

`256` 表示：

> RNN 的隐藏状态/记忆维度。

它出现在：

- `hidden_size=256`
- `h0` 的最后一维
- `h1` 的最后一维
- `output` 的最后一维
- `weight_ih_l0` 的第一维
- `weight_hh_l0` 的两个维度

所以：

```python
h0.shape[-1]      # 256
h1.shape[-1]      # 256
output.shape[-1]  # 256
```

注意：如果是双向 RNN，`output` 最后一维会变成：

```text
hidden_size * num_directions = 256 * 2 = 512
```

但 `h0`、`h1` 的最后一维仍然是 `hidden_size=256`。

---

### 3. `1` 出现在哪里？

```python
num_layers=1
h0 = torch.randn(size=(1, 32, 256))
```

`1` 表示：

> 层数 × 方向数。

默认单向时：

```text
num_directions = 1
```

所以：

```text
num_layers * num_directions = 1 * 1 = 1
```

因此 `h0`、`h1` 的第一维是 `1`。

如果你设置：

```python
nn.RNN(128, 256, num_layers=1, bidirectional=True)
```

那么方向数变成 `2`，`h0`、`h1` 第一维就变成：

```text
1 * 2 = 2
```

如果你设置：

```python
nn.RNN(128, 256, num_layers=2)
```

那么 `h0`、`h1` 第一维就变成：

```text
2 * 1 = 2
```

所以 `1` 这个数字有两个相关含义：

- `num_layers=1`：只有 1 层。
- `h0.shape[0]=1`：层数 × 方向数 = 1。

单层单向时，它们看起来一样；多层或双向时，就不再一样。

---

### 4. `32` 出现在哪里？

```python
x = torch.randn(size=(5, 32, 128))
h0 = torch.randn(size=(1, 32, 256))
```

`32` 表示：

> batch size，也就是同时处理多少个独立样本。

它出现在：

- `x` 的第二维
- `h0` 的第二维
- `output` 的第二维
- `h1` 的第二维

默认 `batch_first=False`，所以 batch 在第二维：

```text
x:      (时间, 批, 特征)
h0/h1:  (层×方向, 批, 记忆)
output: (时间, 批, 记忆×方向)
```

如果你设置 `batch_first=True`，`x` 和 `output` 会变成：

```text
x:      (批, 时间, 特征)
output: (批, 时间, 记忆×方向)
```

但注意：`h0`、`h1` 的形状不变，仍然是：

```text
(层×方向, 批, 记忆)
```

---

### 5. `5` 出现在哪里？

```python
x = torch.randn(size=(5, 32, 128))
```

`5` 表示：

> 序列长度，也就是时间步数。

它出现在：

- `x` 的第一维
- `output` 的第一维

它不会出现在 `h0`、`h1` 里，因为隐藏状态只保存最终状态，不保存所有时间步。

单层单向时：

```python
output[-1] == h1[0]
```

因为最后一天的输出，就是最后一天的小本本。

---

## 五、形状关系总表

| 数值 | 出现位置 | 含义 | 变化条件 |
|---|---|---|---|
| `128` | `input_size`、`x.shape[-1]` | 每步输入特征维度 | 改 `input_size` 必须改 `x` 最后一维 |
| `256` | `hidden_size`、`h0/h1.shape[-1]`、`output.shape[-1]` | 隐藏记忆维度 | 双向时 `output` 最后一维变 `256*2` |
| `1` | `num_layers`、`h0/h1.shape[0]` | 层数 × 方向数 | 多层或双向会变大 |
| `32` | `x/h0/output/h1` 第二维 | batch 大小 | `batch_first=True` 时 `x/output` 的 batch 去第一维 |
| `5` | `x/output` 第一维 | 时间步/序列长度 | `batch_first=True` 时去第二维，`h` 没有它 |

记住三句口诀：

```text
x：时间、批、输入
h0/h1：层×方向、批、记忆
output：时间、批、记忆×方向
```

---

## 六、如果改成多层或双向，会怎么变？

### 1. 多层 RNN

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2)
x = torch.randn(5, 32, 128)
h0 = torch.randn(2, 32, 256)
output, h1 = rnn(x, h0)
```

形状：

```python
output.shape  # (5, 32, 256)
h1.shape      # (2, 32, 256)
```

含义：

- `h0[0]`：第 0 层初始隐藏状态。
- `h0[1]`：第 1 层初始隐藏状态。
- `h1[0]`：第 0 层最终隐藏状态。
- `h1[1]`：第 1 层最终隐藏状态。
- `output`：只包含最后一层每个时间步的输出。
- `output[-1] == h1[-1]`。

第 1 层的输入维度自动变成第 0 层的 `hidden_size=256`。

---

### 2. 双向 RNN

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1, bidirectional=True)
x = torch.randn(5, 32, 128)
h0 = torch.randn(2, 32, 256)
output, h1 = rnn(x, h0)
```

形状：

```python
output.shape  # (5, 32, 512)
h1.shape      # (2, 32, 256)
```

因为：

```text
num_directions = 2
output 最后一维 = hidden_size * 2 = 512
h0/h1 第一维 = num_layers * num_directions = 1 * 2 = 2
```

双向就是：

> 正着读一遍，反着读一遍，最后把两个方向的记忆拼起来。

---

## 七、默认 `batch_first=False` 

默认 batch 不在第一维。

默认：

```python
x:      (seq_len, batch, input_size)
output: (seq_len, batch, hidden_size * num_directions)
```

如果你写：

```python
rnn = nn.RNN(128, 256, batch_first=True)
x = torch.randn(32, 5, 128)
```

那么：

```python
x:      (batch, seq_len, input_size)
output: (batch, seq_len, hidden_size * num_directions)
```

但 `h0`、`h1` 仍然是：

```python
(num_layers * num_directions, batch, hidden_size)
```

也就是说，`batch_first=True` 只影响 `x` 和 `output`，不影响隐藏状态 `h`。

---

## 八、用代码验证一下

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
x = torch.randn(size=(5, 32, 128))
h0 = torch.randn(size=(1, 32, 256))

output, h1 = rnn(x, h0)

print(output.shape)  # torch.Size([5, 32, 256])
print(h1.shape)      # torch.Size([1, 32, 256])

print(torch.allclose(output[-1], h1[0]))  # True

print(rnn.weight_ih_l0.shape)  # torch.Size([256, 128])
print(rnn.weight_hh_l0.shape)  # torch.Size([256, 256])
```

---

## 九、最后总结

你这段代码可以翻译成：

```text
5 个时间步
32 个样本
每个样本每步输入 128 维
RNN 内部记忆 256 维
只有 1 层
output 保存所有 5 个时间步的最后一层输出
h1 保存每层最后一个时间步的隐藏状态
```

相同数值的含义：

- `128`：输入特征维度。
- `256`：隐藏记忆维度。
- `1`：层数 × 方向数，默认单向时就是 1。
- `32`：batch 大小。
- `5`：时间步/序列长度。


```text
x：时间、批、输入
h0/h1：层×方向、批、记忆
output：时间、批、记忆×方向
```

# 结合案例：（必看）
 `nn.RNN` 这个 API 本身

先只看这一行：

```python
self.rnn = nn.RNN(128, 256, 1)
```

它等价于：

```python
self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
```

这行代码只做一件事：

> **创建一个 RNN 层。**  
> 它规定了“输入长什么样、记忆多大、叠几层”，但还没真正开始跑数据。

真正跑数据是在 `forward` 里：

```python
output, hidden = self.rnn(x, h0)
```

所以要搞懂 RNN，只要搞懂两件事：

1. **构造函数 `nn.RNN(...)` 的参数**
2. **调用时 `rnn(x, h0)` 的输入和输出形状**

---

## 一、构造函数 `nn.RNN(...)` 的参数

`nn.RNN` 的完整签名（常用的几个）是：

```python
nn.RNN(
    input_size,          # 每个时间步输入的特征维度
    hidden_size,         # 隐藏记忆的维度
    num_layers=1,        # 叠几层
    nonlinearity='tanh', # 激活函数, 'tanh' 或 'relu'
    bias=True,           # 是否加偏置
    batch_first=False,   # 输入的第一维是不是 batch
    dropout=0,           # 层间 dropout
    bidirectional=False, # 是否双向
)
```

你这个案例只用了前三个：

```python
nn.RNN(128, 256, 1)
```

### 1. `input_size=128` 是什么

> 每个时间步，输入给 RNN 的向量有多少维。

在案例里，这个 `128` 从哪来？看它上一层：

```python
self.ebd = nn.Embedding(unique_word_count, 128)
```

Embedding 把每个词（一个整数索引）变成了 **128 维向量**。  
所以 RNN 每一步收到的就是这 128 维向量，`input_size` 就必须是 128。

如果你把 Embedding 改成：

```python
self.ebd = nn.Embedding(unique_word_count, 64)
```

那 RNN 也必须改成：

```python
self.rnn = nn.RNN(64, 256, 1)
```

否则会报错。**这是两个参数之间的第一个约束。**

### 2. `hidden_size=256` 是什么

> RNN 内部的“记忆”有多少维。

你可以把它理解成：RNN 每一步读完输入后，会把信息压缩成一个 256 维的向量，传给下一步。

这个 `256` 决定了很多东西的形状：

- RNN 输出的 `output` 最后一维 = 256
- RNN 返回的 `hidden` 最后一维 = 256
- 下一层 `nn.Linear(256, unique_word_count)` 的输入维度也必须是 256

**这是第二个约束：`hidden_size` 必须和下一层全连接层的输入维度一致。**

### 3. `num_layers=1` 是什么

> RNN 叠几层。

- `num_layers=1`：只有一层 RNN。
- `num_layers=2`：第一层的输出，喂给第二层，第二层再输出。

它影响的主要是 `hidden` 的第一维：

```text
hidden.shape[0] = num_layers * num_directions
```

单层单向时，就是 `1`。所以案例里 `init_hidden` 写的是：

```python
return torch.zeros(1, bs, 256)
```

这个 `1` 就是 `num_layers=1` 来的。

---

## 二、调用时的输入：`rnn(x, h0)`

调用的时候要传两样东西：

```python
output, hidden = self.rnn(x, h0)
```

### 1. `x`：输入序列

默认（`batch_first=False`）下，`x` 的形状必须是：

```text
(seq_len, batch, input_size)
```

在案例里就是：

```text
(32, 5, 128)
```

- **32**：序列长度，一个样本有多少个时间步。
- **5**：batch，一次喂 5 个句子。
- **128**：每个时间步的特征维度，等于 `input_size`。

这三个数分别在哪儿定义？

| x 的维度 | 值 | 来源 |
|---|---|---|
| `seq_len` | 32 | `LyricsDataset(corpus_idx, 32)` |
| `batch` | 5 | `DataLoader(..., batch_size=5)` |
| `input_size` | 128 | `nn.Embedding(unique_word_count, 128)` |

注意：**RNN 自己不知道 seq_len 和 batch 是多少**，它只看 `input_size` 对不对得上。  
`seq_len` 和 `batch` 可以每次不一样，但 `input_size` 必须严格等于 `128`。

### 2. `h0`：初始隐藏状态

默认形状：

```text
(num_layers * num_directions, batch, hidden_size)
```

案例里就是：

```text
(1, 5, 256)
```

- **1**：`num_layers * num_directions = 1 * 1 = 1`
- **5**：batch，必须和 `x` 的 batch 一致
- **256**：`hidden_size`

这个 `h0` 一般用全 0 初始化，案例里就是：

```python
def init_hidden(self, bs):
    return torch.zeros(1, bs, 256)
```

**这是第三个约束：`h0` 的 batch 维必须等于 `x` 的 batch 维。**

如果 `x` 的 batch 是 5，`h0` 的 batch 也必须是 5，否则报错。

---

## 三、调用后的输出：`output, hidden`

```python
output, hidden = self.rnn(x, h0)
```

RNN 返回两个东西。

### 1. `output`：所有时间步的输出

默认形状：

```text
(seq_len, batch, hidden_size * num_directions)
```

案例里单向单层就是：
num_layers=1 → 单层

bidirectional=False → 单向

```text
(32, 5, 256)
```

含义：

> 对每一个时间步、每一个样本，RNN 都给出一个 256 维的记忆向量。

所以 `output[0]` 是第 0 步所有样本的记忆，`output[-1]` 是最后一步所有样本的记忆。

### 2. `hidden`：最后一个时间步的隐藏状态

默认形状：

```text
(num_layers * num_directions, batch, hidden_size)
```

案例里就是：

```text
(1, 5, 256)
```

含义：

> 每一层最后一步的记忆。

单层单向时：

```python
output[-1] == hidden[0]
```

因为最后一层的最后一步，就是整段序列读完后的最终记忆。

**注意：**

- `output` 的第二维是 batch（和输入一样）
- `hidden` 的第二维也是 batch
- `output` 的最后一维是 `hidden_size`
- `hidden` 的最后一维也是 `hidden_size`
- 但 `output` 第一维是 `seq_len`
- `hidden` 第一维是 `num_layers * num_directions`

这是最容易混的地方。

---

## 四、一张图把输入输出串起来

```text
输入 x:   (seq_len=32, batch=5, input_size=128)
输入 h0:  (num_layers=1, batch=5, hidden_size=256)

              ↓  RNN

输出 output: (seq_len=32, batch=5, hidden_size=256)
输出 hidden: (num_layers=1, batch=5, hidden_size=256)
```

和案例代码一一对照：

| 张量 | 案例里的形状 | 由哪些参数决定 |
|---|---|---|
| `x` | `(32, 5, 128)` | seq_len, batch, `input_size` |
| `h0` | `(1, 5, 256)` | `num_layers`, batch, `hidden_size` |
| `output` | `(32, 5, 256)` | seq_len, batch, `hidden_size` |
| `hidden` | `(1, 5, 256)` | `num_layers`, batch, `hidden_size` |

---

## 五、参数之间的三个硬约束

这三条一旦违反，直接报错：

### 约束 1：上一层输出维度 = `input_size`

```python
nn.Embedding(unique_word_count, 128)   # 输出 128 维
nn.RNN(128, 256, 1)                     # input_size 必须是 128
```

### 约束 2：`hidden_size` = 下一层输入维度

```python
nn.RNN(128, 256, 1)                     # hidden_size = 256
nn.Linear(256, unique_word_count)       # 输入必须是 256
```

### 约束 3：`h0` 的 batch = `x` 的 batch

```python
hidden = model.init_hidden(5)           # batch = 5
output, hidden = model(x, hidden)       # x 的 batch 也必须是 5
```

另外还有一条**软约束**（不报错但逻辑会乱）：

> `h0` 的 `num_layers` 和 `hidden_size`，必须和 `nn.RNN` 里设置的 `num_layers`、`hidden_size` 一致。

案例里 `init_hidden` 写死了 `torch.zeros(1, bs, 256)`，就是因为模型是 1 层、256 维。  
如果你把模型改成 `nn.RNN(128, 512, 1)`，那 `init_hidden` 也要改成 `torch.zeros(1, bs, 512)`，同时 `nn.Linear` 也要改成 `nn.Linear(512, unique_word_count)`。

---

## 六、和案例里的两次 transpose/reshape 对上

案例里 forward 是这么写的：

```python
embd = self.ebd(inputs)                       # (5, 32, 128)
output, hidden = self.rnn(embd.transpose(0, 1), hidden)
output = self.out(output.reshape(-1, 256))
```

为什么要动形状？就是因为 RNN 的 API 规定死了：

- **进来**：`(seq_len, batch, input_size)`
- **出去**：`(seq_len, batch, hidden_size)`

而 Embedding 给你的是 `(batch, seq_len, embed_dim)`。  
所以要 `transpose(0, 1)`，把 batch 和 seq_len 换一下，才能喂给 RNN。

RNN 出来是 `(32, 5, 256)`，但 `nn.Linear` 只吃二维 `(N, features)`，  
所以要 `reshape(-1, 256)`，把前两维展平成 `160`。

这些都是 **为了满足 API 的形状要求**，不是为了模型本身。

---

## 七、小结

只看 API，记住这几句话就够了：

**构造函数 `nn.RNN(input_size, hidden_size, num_layers)`：**

- `input_size`：每一步输入几维
- `hidden_size`：记忆几维
- `num_layers`：叠几层

**调用 `rnn(x, h0)`：**

- `x`：`(seq_len, batch, input_size)`
- `h0`：`(num_layers, batch, hidden_size)`

**返回 `output, hidden`：**

- `output`：`(seq_len, batch, hidden_size)`
- `hidden`：`(num_layers, batch, hidden_size)`

**三条硬约束：**

1. 上一层输出维度 = `input_size`
2. `hidden_size` = 下一层输入维度
3. `h0` 的 batch = `x` 的 batch

你问的这两个点很关键，我单独拆出来讲。

---

## 1. 你现在这个是不是单层单向？

是的。

```python
self.rnn = nn.RNN(128, 256, 1)
```

等价于：

```python
self.rnn = nn.RNN(
    input_size=128,
    hidden_size=256,
    num_layers=1,
    bidirectional=False   # 默认就是 False
)
```

所以：

- `num_layers=1` → **单层**
- `bidirectional=False` → **单向**

因此 `hidden` 的第一维就是：

```text
num_layers * num_directions = 1 * 1 = 1
```

所以 `hidden` 的形状是：

```text
(1, batch, 256)
```

在案例里 `batch=5`，所以是 `(1, 5, 256)`。

---

## 2. “时间步”到底是什么意思？

其实是那个滑动窗口

时间步是**序列上的位置**。

举个例子，假设一个样本是一个句子：

```text
我  爱  你
```

分词后是 3 个词，那么这个样本的序列长度 `seq_len = 3`。

RNN 会这样处理：

```text
时间步 1：输入“我”   -> 更新隐藏状态 h1
时间步 2：输入“爱”   -> 结合 h1，更新为 h2
时间步 3：输入“你”   -> 结合 h2，更新为 h3
```

所以这个样本经历了 **3 个时间步**。

在你的案例里：

```python
LyricsDataset(corpus_idx, 32)
```

每个样本的长度是 32，所以每个样本都要经历 **32 个时间步**。

而 `batch=5` 表示：

> 一次同时处理 5 个这样的样本。

这 5 个样本是**并行**处理的。在每个时间步，RNN 同时收到 5 个样本的当前输入：
```text
第 1 个时间步：5 个样本同时各算一次
第 2 个时间步：5 个样本同时各算一次
...
第 32 个时间步：5 个样本同时各算一次
```


```text
时间步 1：同时输入 5 个样本的第 1 个词
时间步 2：同时输入 5 个样本的第 2 个词
...
时间步 32：同时输入 5 个样本的第 32 个词
```

所以：

- 对**单个样本**来说，它经过了 `seq_len` 个时间步。
- 对**整个 batch** 来说，每个时间步同时处理 `batch` 个样本。
- 总共处理的“样本-时间步”数量是 `seq_len * batch`。

但 RNN 神经元只有一个（共享权重），它在每个时间步对 batch 里所有样本并行计算。  
所以“时间步”描述的是**序列推进到了第几个位置**，而不是“所有样本总共经过了多少次”。

---

## 3. `hidden` 为什么是“最后一个时间步的隐藏状态”？

RNN 是按时间步顺序处理的。

假设单层单向，batch=1，序列长度=3：

```text
初始隐藏状态 h0

时间步 1：输入 x1 -> 计算 h1
时间步 2：输入 x2 -> 计算 h2
时间步 3：输入 x3 -> 计算 h3
```

处理完整个序列后，最后得到的隐藏状态是 `h3`。

`hidden` 返回的就是这个 `h3`。

因为 `h3` 是在最后一个时间步产生的，所以叫“最后一个时间步的隐藏状态”。

它包含了从第 1 步到第 3 步的所有信息。  
在语言模型里，通常就用这个最终隐藏状态来代表整个序列的“记忆”。

---

## 4. `output` 和 `hidden` 的关系

还是上面那个例子，序列长度=3：

```text
output = [h1, h2, h3]   # 所有时间步的隐藏状态
hidden = [h3]           # 最后一个时间步的隐藏状态
```

用形状表示：

```text
output: (seq_len, batch, hidden_size) = (3, batch, 256)
hidden: (num_layers, batch, hidden_size) = (1, batch, 256)
```

所以：

```python
output[-1] == hidden[0]
```

在你的案例里：

```text
seq_len = 32
batch = 5
hidden_size = 256

output.shape = (32, 5, 256)
hidden.shape = (1, 5, 256)
```

`output[-1]` 就是第 32 个时间步的隐藏状态，形状 `(5, 256)`。  
`hidden[0]` 也是第 32 个时间步的隐藏状态，形状 `(5, 256)`。  
它们完全相等。

---

## 5. 再走一遍

`forward` 里：

```python
embd = self.ebd(inputs)                       # (5, 32, 128)
output, hidden = self.rnn(embd.transpose(0, 1), hidden)
```

`embd.transpose(0, 1)` 后：

```text
(32, 5, 128)   -> (seq_len, batch, input_size)
```

RNN 处理这 32 个时间步，每个时间步同时处理 5 个样本。

处理完：

```text
output: (32, 5, 256)   -> 每个时间步、每个样本的隐藏状态
hidden: (1, 5, 256)    -> 第 32 个时间步、每个样本的隐藏状态
```

因为只有 1 层、单向，所以 `hidden` 第一维是 1。  
`hidden[0]` 就是最后一层（也是唯一一层）在最后一个时间步的隐藏状态。

---

## 6. 一句话总结

- **单层单向**：`num_layers=1`，`bidirectional=False`。
- **时间步**：序列上的位置，每个样本都要按时间步依次输入。batch 里的样本在每个时间步是并行处理的。
- **`hidden`**：所有时间步都走完后，最后一个时间步的隐藏状态。它包含了整个序列的信息。
- **`output`**：所有时间步的隐藏状态。
- **关系**：单层单向时，`output[-1] == hidden[0]`。


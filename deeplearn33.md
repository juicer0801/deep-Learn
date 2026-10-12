---
title: ai助教
date: 2026-10-10 15:29:12
tags:
---
# RNN 完整教程：从零开始彻底搞懂循环神经网络

> 目标读者：已学完深度学习基础（前馈神经网络、损失函数、反向传播），但对 RNN 细节不清楚的学习者。
> 
> 本教程包含：概念讲解、数学推导、图片教学、代码逐行注释、常见疑问解答

---

## 目录

1. 为什么需要 RNN
2. RNN 的核心思想
3. 词嵌入层
4. RNN 循环网络层
5. 数据形状变换
6. 滑动窗口采样
7. 案例代码逐行详解
8. 训练过程可视化
9. 预测过程详解
10. RNN 的局限与进阶方向
11. 常见疑问 FAQ

---

## 1. 为什么需要 RNN

### 1.1 前馈神经网络的局限

前馈神经网络（全连接网络、CNN）有一个共同假设：**输入和输出相互独立**。

```
前馈网络：x1 -> [隐藏层] -> y1    （每次输入互不影响）
          x2 -> [隐藏层] -> y2
```

这对图像分类没问题。但序列数据不行！

```
"我 爱 你"  和  "你 爱 我"  -> 完全不同的含义！
```

序列数据的特点是：**后面的数据和前面的数据有关**。

### 1.2 什么是序列数据

| 类型 | 例子 | 序列关系 |
|------|------|----------|
| 文本 | "今天天气真___" | 需要根据前面的词预测下一个词 |
| 时间序列 | 股票价格 | 明天的价格和今天的价格有关 |
| 语音 | 一段话 | 后面的音节依赖前面的音节 |
| 音乐 | 一段旋律 | 下一个音符依赖前面的音符 |

![RNN vs 前馈神经网络对比](artifacts/generated-images/83965c57269bedb7/image-01.png)

> 左图：前馈网络，数据单向流动，没有"记忆"。
> 右图：RNN，隐藏状态 h 会"循环"回来，作为下一步的输入之一。

---

## 2. RNN 的核心思想

### 2.1 一句话概括

**RNN = 一个带有"记忆"的神经元，在每个时间步都会更新记忆。**

### 2.2 类比理解

想象你在读一本书：
- 你读到第5章时，不会忘记第1章讲了什么
- 你的"理解"（隐藏状态）随着阅读不断积累
- 每读一个新句子，你结合"已有理解 + 新句子"来更新"理解"

RNN 就是这样工作的：

```
新理解(h_t) = f(旧理解(h_{t-1}) + 新输入(x_t))
```

### 2.3 RNN 的两种视角

**视角一：循环视角（Compact）**

```
        +--------------+
        |              |
  x_t ->|   RNN 神经元  |-> h_t -> y_t
        |              |
        ^    h_{t-1}   |
        +--------------+
```

看起来是一个神经元，但它被**反复使用**于每个时间步。

**视角二：展开视角（Unrolled）**

```
时间步1          时间步2          时间步3
x1="我"          x2="爱"          x3="你"
  |                |                |
  v                v                v
[RNN]--h1---> [RNN]--h2---> [RNN]--h3---> y3
  ^                ^                |
 h0=0            h1               h2
```

展开后更容易看清数据流动：每个时间步的 RNN 单元**共享同一组参数**（权重矩阵），但处理不同的输入。

![RNN 时序展开与计算流程](artifacts/generated-images/91c2d2aca72aa0a7/image-01.png)

---

## 3. 词嵌入层

### 3.1 为什么需要词嵌入

计算机不认识"猫"和"狗"，它只认识数字。

**方案一：One-Hot 编码（已淘汰）**

```
猫 -> [1, 0, 0, 0, ...]   （词汇量10000，向量就是10000维）
狗 -> [0, 1, 0, 0, ...]
```

问题：维度太高，任意两个词的距离都一样。

**方案二：词嵌入（Word Embedding）**

```
猫 -> [0.2, -0.5, 0.8, 0.1, ...]   （128维）
狗 -> [0.3, -0.4, 0.7, 0.2, ...]   （和猫很接近！）
飞机 -> [-0.9, 0.1, -0.3, 0.6, ...] （和猫狗很远）
```

![词嵌入 Word Embedding 原理](artifacts/generated-images/52c4ebf8d43870ae/image-01.png)

### 3.2 词嵌入层的工作流程

```
步骤1：分词
"我爱北京天安门" -> ["我", "爱", "北京", "天安门"]

步骤2：构建词表（词到索引映射）
{"我": 0, "爱": 1, "北京": 2, "天安门": 3, ...}

步骤3：查表得到词向量
"我" -> 索引0 -> 查嵌入矩阵第0行 -> [0.12, -0.34, ...]（128维向量）
```

### 3.3 PyTorch 代码

```python
# 创建词嵌入层
# num_embeddings=10000: 词汇表有10000个词
# embedding_dim=128: 每个词用128维向量表示
embed = nn.Embedding(num_embeddings=10000, embedding_dim=128)

# 输入：词的索引（整数）
# 输出：对应的词向量
word_indices = torch.tensor([0, 1, 2])  # 3个词
vectors = embed(word_indices)           # 形状: [3, 128]
```

> 关键理解：词嵌入层本质上就是一个**查找表（Lookup Table）**。
> 它的参数是一个 (词汇量 x 向量维度) 的矩阵，输入索引，输出对应行的向量。
> 训练过程中，这些向量会被不断更新，最终学到语义信息。

---

## 4. RNN 循环网络层

### 4.1 RNN 内部的数学计算

**RNN 神经元在每个时间步做两件事：**

#### 第一步：计算新的隐藏状态 h_t

```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

| 符号 | 含义 | 形状 |
|------|------|------|
| x_t | 当前时间步的输入（词向量） | [batch, input_size] |
| h_{t-1} | 上一个时间步的隐藏状态 | [batch, hidden_size] |
| W_ih | 输入到隐藏的权重矩阵 | [input_size, hidden_size] |
| b_ih | 输入到隐藏的偏置 | [hidden_size] |
| W_hh | 隐藏到隐藏的权重矩阵 | [hidden_size, hidden_size] |
| b_hh | 隐藏到隐藏的偏置 | [hidden_size] |
| h_t | 当前时间步的隐藏状态 | [batch, hidden_size] |

> 直觉理解：
> - W_ih * x_t：当前输入的信息
> - W_hh * h_{t-1}：过去记忆的信息
> - 两者相加 -> 融合"新信息"和"旧记忆"
> - tanh：把结果压缩到 (-1, 1) 之间

#### 第二步：计算输出 y_t

```
y_t = W_hy * h_t + b_y
```

| 符号 | 含义 |
|------|------|
| y_t | 当前时间步的输出 |
| W_hy | 隐藏到输出的权重矩阵 |
| b_y | 输出层偏置 |

> 在歌词生成任务中，y_t 的维度 = 词汇表大小（5703），表示对每个词的"得分"。

### 4.2 从得分到预测词

```
y_t = [2.1, 0.5, -0.3, 4.7, 1.2, ...]  （5703个得分）
                    | softmax
概率 = [0.01, 0.002, 0.001, 0.15, 0.03, ...]  （加起来=1）
                    | argmax
预测词 = 索引3对应的词 -> "你"
```

### 4.3 PyTorch 中的 RNN 层

```python
rnn = nn.RNN(
    input_size=128,    # 输入维度 = 词向量维度
    hidden_size=256,   # 隐藏状态维度
    num_layers=1       # RNN层数
)

output, h_n = rnn(x, h_0)
```

**输入输出的形状（最容易搞混的部分！）：**

```
输入 x:   [seq_len, batch, input_size]
输入 h_0: [num_layers, batch, hidden_size]

输出 output: [seq_len, batch, hidden_size]
输出 h_n:    [num_layers, batch, hidden_size]
```

> 注意：PyTorch 的 RNN 默认输入格式是 [seq_len, batch, feature]，
> 而不是常见的 [batch, seq_len, feature]。所以需要 transpose(0,1) 来交换维度！

---

## 5. 数据形状变换

![RNN 数据形状变换](artifacts/generated-images/af91772349d3d234/image-01.png)

用具体数字来追踪（以你的代码为例）：

```
假设：batch_size=5, seq_len=32, vocab_size=5703

1. 输入 x:           [5, 32]          （5个句子，每句32个词的索引）
2. 词嵌入后 embd:    [5, 32, 128]     （每个词索引变成128维向量）
3. 转置后:           [32, 5, 128]     （交换第0维和第1维）
4. RNN输出 output:   [32, 5, 256]     （每个时间步输出256维隐藏状态）
5. reshape后:        [160, 256]       （32x5=160，展平成二维）
6. 全连接层输出:      [160, 5703]      （每个位置对5703个词的得分）

对应的目标值 y:
原始 y:             [5, 32]
转置+reshape后:      [160]            （160个目标词索引）
```

> 为什么要 reshape？全连接层 nn.Linear(256, 5703) 只接受二维输入 [N, 256]，
> 所以要把 [32, 5, 256] 展平成 [160, 256]。

---

## 6. 滑动窗口采样

### 6.1 训练数据怎么构造

我们有一长串歌词索引：[0, 1, 2, 3, 40, 0, 4, 5, 6, 7, 8, 3, ...]

目标是：**给一段词，预测下一个词**。

![滑动窗口数据采样策略](artifacts/generated-images/6f7328a3985cbd81/image-01.png)

```
原始序列: [0, 1, 2, 3, 40, 0, 4, 5, 6, 7, ...]

窗口大小 = 5（即 num_chars=5）

样本1: x = [0, 1, 2, 3, 40]    y = [1, 2, 3, 40, 0]
样本2: x = [1, 2, 3, 40, 0]    y = [2, 3, 40, 0, 4]
样本3: x = [2, 3, 40, 0, 4]    y = [3, 40, 0, 4, 5]
```

> 关键：y 就是 x 向右移动一位！
> 给定位置 i 的词，模型要预测位置 i+1 的词。

### 6.2 代码实现

```python
class LyricsDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        start = idx
        end = start + self.num_chars
        
        x = self.corpus_idx[start:end]        # 输入：5个词
        y = self.corpus_idx[start+1:end+1]    # 目标：后移1位的5个词
        
        return torch.tensor(x), torch.tensor(y)
```

---

## 7. 案例代码逐行详解

### 7.1 整体结构

```
+-------------------------------------------------+
|                 完整流程                          |
|                                                   |
|  1. build_vocab()    -> 构建词表                   |
|  2. LyricsDataset    -> 构造训练数据                |
|  3. TextGenerator    -> 定义RNN模型                |
|  4. train()          -> 训练模型                   |
|  5. evaluate()       -> 用训练好的模型生成歌词       |
+-------------------------------------------------+
```

![RNN 歌词生成完整流程](artifacts/generated-images/19f0b156d006bf0e/image-01.png)

### 7.2 第一步：构建词表 build_vocab()

```python
def build_vocab():
    # unique_words: 去重后的所有词
    # all_words: 每行歌词的分词结果（不去重）
    unique_words, all_words = [], []
    
    # 逐行读取歌词文件
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        # jieba分词: "想要有直升机" -> ['想要', '有', '直升机']
        words = jieba.lcut(line)
        
        # 记录每行的分词结果
        all_words.append(words)
        
        # 去重：如果这个词还没见过，就加入unique_words
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    
    # 词汇表大小（去重后约5703个词）
    word_count = len(unique_words)
    
    # 构建词表字典: {'想要': 0, '有': 1, '直升机': 2, ...}
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    
    # 把整篇歌词转换成索引序列
    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])  # 词 -> 索引
        tmp.append(word_to_index[' '])       # 每行末尾加空格
        corpus_idx.extend(tmp)
    
    return unique_words, word_to_index, word_count, corpus_idx
```

> 为什么要加空格？每行歌词之间用空格隔开，让模型知道"这里是换行"。

**结果示意：**

```
原始歌词: "想要有直升机\n想要和你飞到宇宙去\n"
分词:     ['想要', '有', '直升机', '\n', '想要', '和', '你', ...]
索引:     [0,      1,   2,        3,    0,      4,   5,    ...]
```

### 7.3 第二步：数据集 LyricsDataset

```python
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx    # 整篇歌词的索引序列
        self.num_chars = num_chars      # 每个样本的长度（32个词）
        self.word_count = len(self.corpus_idx)
        self.number = self.word_count // self.num_chars
    
    def __len__(self):
        return self.number
    
    def __getitem__(self, idx):
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        
        x = self.corpus_idx[start:end]        # 输入序列
        y = self.corpus_idx[start + 1:end + 1] # 目标序列（右移一位）
        
        return torch.tensor(x), torch.tensor(y)
```

**具体例子（num_chars=5）：**

```
corpus_idx = [0, 1, 2, 3, 40, 0, 4, 5, 6, 7, 8, ...]

idx=0: x=[0,1,2,3,40],     y=[1,2,3,40,0]
idx=1: x=[1,2,3,40,0],     y=[2,3,40,0,4]
idx=2: x=[2,3,40,0,4],     y=[3,40,0,4,5]
```

> 核心思想：模型看到 x 中的词，要学会预测 y 中对应位置的下一个词。

### 7.4 第三步：模型 TextGenerator

```python
class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):  # = 5703
        super().__init__()
        
        # 1. 词嵌入层
        # 参数矩阵形状：[5703, 128]
        self.ebd = nn.Embedding(unique_word_count, 128)
        
        # 2. RNN层
        # 参数：W_ih[128,256], W_hh[256,256], b_ih[256], b_hh[256]
        self.rnn = nn.RNN(128, 256, 1)
        
        # 3. 全连接输出层
        # 参数：W[256, 5703], b[5703]
        self.out = nn.Linear(256, unique_word_count)
    
    def forward(self, inputs, hidden):
        # inputs形状: [batch=5, seq_len=32]
        
        # 1. 词嵌入 -> [5, 32, 128]
        embd = self.ebd(inputs)
        
        # 2. 转置 [5,32,128] -> [32,5,128]，然后RNN
        #    output: [32, 5, 256]
        #    hidden: [1, 5, 256]
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        
        # 3. reshape [32,5,256] -> [160,256]，然后全连接
        #    输出: [160, 5703]
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        
        return output, hidden
    
    def init_hidden(self, bs):
        # 初始化隐藏状态为全0: [1层, batch, 256维]
        return torch.zeros(1, bs, 256)
```

**模型结构图：**

```
输入 [5, 32]（词索引）
    |
    v
+------------------+
|  Embedding       |  [5703, 128] 的查找表
|  5703 -> 128     |
+------------------+
    | [5, 32, 128]
    v
+------------------+
|  transpose(0,1)  |  交换维度
+------------------+
    | [32, 5, 128]
    v
+------------------+
|  RNN             |  32个时间步
|  128 -> 256      |
+------------------+
    | [32, 5, 256]
    v
+------------------+
|  reshape         |  展平
+------------------+
    | [160, 256]
    v
+------------------+
|  Linear          |  全连接层
|  256 -> 5703     |
+------------------+
    | [160, 5703]
    v
输出：每个位置对5703个词的得分
```

### 7.5 第四步：训练 train()

```python
def train():
    # 1. 构建词表
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    
    # 2. 创建数据集（每个样本32个词）
    lyrics = LyricsDataset(corpus_idx, 32)
    
    # 3. 创建模型
    model = TextGenerator(unique_word_count)
    
    # 4. 数据加载器（每批5个样本，打乱顺序）
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    
    # 5. 损失函数：交叉熵损失（多分类问题）
    criterion = nn.CrossEntropyLoss()
    
    # 6. 优化器：Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 7. 训练50轮
    epochs = 50
    for epoch in range(epochs):
        start, iter_num, total_loss = time.time(), 0, 0.0
        
        for x, y in lyrics_dataloader:
            # x形状: [5, 32]  y形状: [5, 32]
            
            # 初始化隐藏状态
            hidden = model.init_hidden(5)  # [1, 5, 256]
            
            # 前向传播
            output, hidden = model(x, hidden)
            # output形状: [160, 5703]
            
            # 处理目标值y: [5,32] -> 转置[32,5] -> reshape[160]
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            
            # 计算损失
            loss = criterion(output, y)
            
            # 反向传播三步曲
            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数
            
            total_loss += loss.item()
            iter_num += 1
        
        print(f'epoch: {epoch+1}, loss: {total_loss/iter_num:.4f}')
    
    # 8. 保存模型
    torch.save(model.state_dict(), './model/text_generator.pth')
```

> 训练过程核心逻辑：
> 1. 给模型一段文本 x（32个词）
> 2. 模型对每个位置预测下一个词的概率分布（5703个概率）
> 3. 和真实答案 y 比较，计算损失
> 4. 反向传播，更新参数
> 5. 重复，直到损失足够小

### 7.6 第五步：预测 evaluate()

```python
def evaluate(start_word, sentence_length):
    # 1. 构建词表
    unique_words, word_to_index, unique_word_count, _ = build_vocab()
    
    # 2. 创建模型并加载训练好的参数
    model = TextGenerator(unique_word_count)
    model.load_state_dict(torch.load('./model/text_generator.pth'))
    
    # 3. 初始化隐藏状态（batch=1）
    hidden = model.init_hidden(1)
    
    # 4. 把起始词转为索引
    word_idx = word_to_index[start_word]
    
    # 5. 存放生成的词索引
    generate_sentence = [word_idx]
    
    # 6. 逐个生成
    for i in range(sentence_length):
        # 输入当前词（形状: [1, 1]）
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        
        # 从5703个得分中选最大的
        word_idx = torch.argmax(output)
        
        generate_sentence.append(word_idx)
    
    # 7. 把索引转回词并打印
    for idx in generate_sentence:
        print(unique_words[idx], end='')
```

**预测过程图解：**

```
输入: "星星"
       |
       v
  [词嵌入] -> [RNN] -> [全连接] -> 5703个得分 -> argmax -> "在"
                                    |
                                    v
                              [词嵌入] -> [RNN] -> [全连接] -> 5703个得分 -> argmax -> "天"
                                    |
                                    v
                              ... 继续生成 ...

最终输出: "星星在天..."（共50个词）
```

> 注意：预测时，每一步的输入是**上一步预测的词**，而不是真实数据。
> 这就是"自回归"生成。如果某一步预测错了，错误会累积。

---

## 8. 训练过程可视化

训练过程中 loss 的变化：

```
epoch: 1,  loss: 1.8442  <- 刚开始，模型几乎随机猜
epoch: 2,  loss: 0.2115  <- 快速下降！
epoch: 3,  loss: 0.1201
epoch: 5,  loss: 0.1023
epoch: 10, loss: 0.0957  <- 逐渐收敛
epoch: 50, loss: ~0.08   <- 模型已经很熟练了
```

**loss 含义解读：**

| loss 值 | 含义 |
|---------|------|
| ~6.36 (=ln5703) | 完全随机猜（均匀分布） |
| ~1.84 | 开始学到一些规律 |
| ~0.1 | 基本能预测对大部分词 |
| ~0.01 | 几乎完美预测（可能过拟合） |

---

## 9. 预测过程详解

### 9.1 温度参数（进阶技巧）

当前代码直接用 argmax 选概率最大的词，这会导致生成很"保守"。

可以加入**温度参数**来控制生成的随机性：

```python
def predict_with_temperature(output, temperature=0.8):
    # temperature < 1: 更确定，选高概率词
    # temperature = 1: 正常
    # temperature > 1: 更随机，更有创造性
    probs = torch.softmax(output / temperature, dim=-1)
    word_idx = torch.multinomial(probs, 1)[0]
    return word_idx
```

### 9.2 生成示例

```
输入: evaluate('分手', 50)

输出:
分手的话像语言暴力
我已无能为力再提起 决定中断熟悉
一步两步三步四步望著天 看星星
一颗两颗三颗四颗 连成线背著背默默许下心愿
看远方的星
```

> 模型确实学到了周杰伦歌词的风格和模式！

---

## 10. RNN 的局限与进阶方向

### 10.1 梯度消失问题

RNN 在反向传播时，梯度需要沿着时间步往回传：

```
h_T -> h_{T-1} -> h_{T-2} -> ... -> h_1
```

如果序列很长，梯度会**指数级衰减**，导致模型学不到长距离依赖。

### 10.2 解决方案

| 方案 | 核心思想 |
|------|----------|
| **LSTM** | 引入"门控"机制（遗忘门、输入门、输出门），选择性记忆和遗忘 |
| **GRU** | LSTM 的简化版，只有两个门，训练更快，效果相当 |
| **双向 RNN** | 同时从左到右和从右到左处理序列 |
| **Attention** | 让模型"看到"序列中所有位置 |
| **Transformer** | 完全基于 Attention，抛弃 RNN 结构（GPT、BERT 的基础） |

### 10.3 PyTorch 中使用 LSTM

```python
# 只需要把 nn.RNN 换成 nn.LSTM
self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1)

# LSTM 有两个隐藏状态: (h_n, c_n)
# h_n: 隐藏状态
# c_n: 细胞状态（LSTM 特有的"长期记忆"）
```

---

## 11. 常见疑问 FAQ

### Q1: 为什么隐藏状态初始化为全0

A: 因为没有先验知识。全0是一个中性的起点，让模型从数据中学习。

### Q2: 为什么每个 batch 都要重新初始化 hidden

A: 因为每个 batch 是独立的样本，不应该共享记忆。

### Q3: 为什么 RNN 的输入要 transpose

A: PyTorch 的 nn.RNN 要求输入格式是 [seq_len, batch, feature]，
而我们的数据是 [batch, seq_len, feature]。所以需要 transpose(0,1)。

### Q4: CrossEntropyLoss 和 NLLLoss 有什么区别

A: CrossEntropyLoss = LogSoftmax + NLLLoss。它内部自动做了 softmax，
所以模型的输出不需要手动加 softmax。

### Q5: 为什么预测时 batch_size=1

A: 因为预测时我们只生成一段文本，不需要批量处理。

### Q6: 模型参数量有多少

A: 粗略计算：
- Embedding: 5703 x 128 = 约73万
- RNN: 128x256 + 256x256 + 256 + 256 = 约10万
- Linear: 256x5703 + 5703 = 约146万
- 总计约 230万参数

### Q7: 训练多久能出效果

A: CPU 上大约 5-6 秒/epoch，50 个 epoch 约 5 分钟就能看到不错的效果。

### Q8: 生成的歌词为什么有时候不通顺

A: 几个原因：
1. 基础 RNN 的"记忆"有限，长距离依赖学不好
2. 训练数据有限
3. 词级别的处理粒度
4. 没有使用 beam search 等更好的解码策略

---

## 总结：RNN 核心知识点速查表

| 知识点 | 关键点 |
|--------|--------|
| 为什么用RNN | 处理序列数据，具有"记忆"能力 |
| 词嵌入 | 把词映射为低维稠密向量，捕捉语义 |
| 隐藏状态 h | RNN 的"记忆"，融合旧记忆和新输入 |
| 核心公式 | h_t = tanh(W * x_t + U * h_{t-1} + b) |
| 参数共享 | 所有时间步共用同一组权重 |
| 数据格式 | RNN输入: [seq_len, batch, feature] |
| 训练目标 | 给定前n个词，预测第n+1个词 |
| 损失函数 | CrossEntropyLoss（多分类） |
| 预测方式 | 自回归：上一步输出作为下一步输入 |
| 主要缺点 | 梯度消失，难以处理长序列 |
| 改进方向 | LSTM, GRU, Attention, Transformer |

---

> 学习建议：
> 1. 先理解概念，不要急着写代码
> 2. 用具体数字跟踪数据形状变化（最重要的技能！）
> 3. 动手跑一遍代码，打印每一步的形状
> 4. 尝试修改参数（hidden_size, num_chars, lr），观察效果
> 5. 学完 RNN 后，继续学 LSTM -> Attention -> Transformer

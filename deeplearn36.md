---
title: RNN新闻情感分类器（ai课题）
date: 2026-10-14 21:27:10
tags:
---
# 🎯 RNN 实践课题：中文新闻分类器

> **目标**：学完本教程后，你将独立完成一个完整的 RNN 项目
> **难度**：⭐⭐⭐（中等，需要理解前两个教程的内容）
> **预计时间**：2-3 小时

---

## 📋 课题说明

**任务**：给定一条新闻标题，判断它属于哪个类别
**数据集**：THUCNews（中文新闻文本分类数据集，简化版）
**类别**：体育、财经、娱乐、科技、家居（5分类）

```
输入: "梅西带领阿根廷队夺得世界杯冠军"     → 模型 → 输出: "体育" ✅
输入: "苹果公司发布新款iPhone 16"           → 模型 → 输出: "科技" ✅
输入: "央行宣布降准0.5个百分点"             → 模型 → 输出: "财经" ✅
```

---

## 🗺️ 完成步骤（按顺序执行）

### 步骤 1：环境准备（10分钟）

```python
# 安装必要的库
pip install torch datasets jieba scikit-learn
```

### 步骤 2：导入库（5分钟）

```python
"""
任务: 中文新闻分类 (体育/财经/娱乐/科技/家居)
数据集: THUCNews (简化版)
模型: Embedding + RNN + Linear
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import jieba
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
```

### 步骤 3：全局配置（10分钟）

```python
# ============================================================
# 0. 全局配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 50           # 新闻标题通常较短，50个词足够
BATCH_SIZE = 64        # 新闻分类可以用更大的batch
EMBED_DIM = 128        # 词向量维度
HIDDEN_SIZE = 256      # RNN隐藏层维度
NUM_LAYERS = 2         # 新闻分类可以用2层RNN，提取更复杂的特征
EPOCHS = 15            # 训练15轮
LR = 1e-3              # 学习率
MIN_FREQ = 3           # 过滤掉出现少于3次的词
MAX_VOCAB = 25000      # 词表大小

print(f"使用设备: {DEVICE}")
```

### 步骤 4：加载数据集（15分钟）

```python
# ============================================================
# 1. 加载数据集
# ============================================================
# 注意：这里使用一个简化的新闻分类数据集
# 实际项目中，你需要从 Hugging Face 或本地文件加载
dataset = load_dataset(
    path="thu-coai/THUCNews",  # 如果这个不可用，可以用其他新闻分类数据集
    cache_dir="./data",
    trust_remote_code=True
)

# 如果上面的数据集加载失败，可以使用以下代码创建模拟数据
# （实际项目中请替换为真实数据）
if dataset is None:
    # 创建模拟数据用于测试
    from datasets import Dataset
    import random
    
    categories = ['体育', '财经', '娱乐', '科技', '家居']
    sports_news = ["梅西进球", "世界杯决赛", "NBA季后赛", "奥运会开幕", "网球大满贯"]
    finance_news = ["股市上涨", "央行降准", "GDP增长", "房价下跌", "汇率变动"]
    entertainment_news = ["电影票房", "明星离婚", "音乐颁奖", "综艺节目", "电视剧热播"]
    tech_news = ["苹果发布", "AI突破", "5G网络", "芯片研发", "量子计算"]
    home_news = ["装修技巧", "家具选购", "家居设计", "厨房收纳", "花园布置"]
    
    train_data = []
    for _ in range(2000):
        cat = random.choice(categories)
        if cat == '体育':
            text = random.choice(sports_news) + random.choice(["冠军", "比赛", "胜利"])
        elif cat == '财经':
            text = random.choice(finance_news) + random.choice(["分析", "预测", "影响"])
        elif cat == '娱乐':
            text = random.choice(entertainment_news) + random.choice(["热门", "关注", "讨论"])
        elif cat == '科技':
            text = random.choice(tech_news) + random.choice(["突破", "进展", "应用"])
        else:
            text = random.choice(home_news) + random.choice(["推荐", "分享", "经验"])
        train_data.append({"text": text, "label": categories.index(cat)})
    
    dataset = Dataset.from_list(train_data)
    dataset = dataset.train_test_split(test_size=0.2)

print(f"训练集大小: {len(dataset['train'])}")
print(f"测试集大小: {len(dataset['test'])}")
print(f"类别: 体育(0), 财经(1), 娱乐(2), 科技(3), 家居(4)")
```

### 步骤 5：构建词表（20分钟）

```python
# ============================================================
# 2. 构建词表
# ============================================================
def build_vocab(texts, min_freq=3, max_size=25000):
    """
    构建词表：统计词频，过滤低频词，分配索引
    
    参数:
        texts: 文本列表
        min_freq: 最小词频，低于此值的词会被当作<unk>
        max_size: 词表最大大小
    
    返回:
        word_to_index: 词到索引的映射字典
    """
    counter = Counter()  # 创建计数器
    
    # 统计所有词的频率
    for text in texts:
        words = jieba.cut(text)  # 结巴分词
        counter.update(words)    # 累加词频
    
    # 初始化词表，包含两个特殊词
    word_to_index = {
        '<pad>': 0,  # 填充标记（用于对齐长度）
        '<unk>': 1,  # 未知词标记（用于未登录词）
    }
    
    # 按频率从高到低添加词
    for word, freq in counter.most_common():
        if freq < min_freq:  # 频率太低，停止
            break
        if len(word_to_index) >= max_size:  # 词表满了，停止
            break
        if word not in word_to_index:  # 避免重复
            word_to_index[word] = len(word_to_index)
    
    print(f"词表大小: {len(word_to_index)}")
    return word_to_index

# 用训练集构建词表
train_texts = [item['text'] for item in dataset['train']]
word_to_index = build_vocab(train_texts, MIN_FREQ, MAX_VOCAB)
```

### 步骤 6：自定义 Dataset（20分钟）

```python
# ============================================================
# 3. 自定义 Dataset
# ============================================================
class NewsDataset(Dataset):
    """
    新闻分类数据集
    
    功能:
        1. 将文本转换为词索引序列
        2. 截断或填充到固定长度
        3. 记录真实长度（用于跳过padding）
    """
    def __init__(self, texts, labels, word_to_index, max_len):
        """
        参数:
            texts: 文本列表
            labels: 标签列表
            word_to_index: 词表
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_len = max_len
        self.pad_idx = word_to_index['<pad>']
        self.unk_idx = word_to_index['<unk>']
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """获取第idx条数据"""
        # 3.1 取原始数据
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 3.2 分词并转换为索引
        words = jieba.cut(text)
        ids = [self.word_to_index.get(w, self.unk_idx) for w in words]
        
        # 3.3 截断（如果超过最大长度）
        ids = ids[:self.max_len]
        
        # 3.4 记录真实长度（至少为1）
        length = max(len(ids), 1)
        
        # 3.5 填充到固定长度
        ids = ids + [self.pad_idx] * (self.max_len - len(ids))
        
        # 3.6 转换为tensor
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        length_tensor = torch.tensor(length, dtype=torch.long)
        
        return ids_tensor, label_tensor, length_tensor

# 创建Dataset和DataLoader
train_texts = [item['text'] for item in dataset['train']]
train_labels = [item['label'] for item in dataset['train']]
test_texts = [item['text'] for item in dataset['test']]
test_labels = [item['label'] for item in dataset['test']]

train_dataset = NewsDataset(train_texts, train_labels, word_to_index, MAX_LEN)
test_dataset = NewsDataset(test_texts, test_labels, word_to_index, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练batch数: {len(train_loader)}")
print(f"测试batch数: {len(test_loader)}")
```

### 步骤 7：定义模型（30分钟）⭐ 核心！

```python
# ============================================================
# 4. 定义模型
# ============================================================
class NewsClassifierRNN(nn.Module):
    """
    新闻分类RNN模型
    
    结构:
        Embedding → RNN → 取最后有效输出 → Linear → 分类
    
    与情感分类的区别:
        1. 5分类（不是2分类）
        2. 可以使用2层RNN
        3. 序列长度更短（50 vs 100）
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes=5):
        """
        参数:
            vocab_size: 词表大小
            embed_dim: 词向量维度
            hidden_size: RNN隐藏层维度
            num_layers: RNN层数
            num_classes: 分类数（默认5）
        """
        super().__init__()
        
        # 4.1 词嵌入层
        # 作用: 将词索引转换为稠密向量
        # 输入: [batch, seq_len]
        # 输出: [batch, seq_len, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 4.2 RNN层
        # 作用: 提取序列特征
        # 输入: [batch, seq_len, embed_dim]
        # 输出: [batch, seq_len, hidden_size]
        self.rnn = nn.RNN(
            input_size=embed_dim,      # 输入维度 = 词向量维度
            hidden_size=hidden_size,   # 隐藏层维度
            num_layers=num_layers,     # RNN层数
            batch_first=True,          # 输入格式: [batch, seq, feature]
        )
        
        # 4.3 全连接层（分类头）
        # 作用: 将RNN输出映射到类别数
        # 输入: [batch, hidden_size]
        # 输出: [batch, num_classes]
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, lengths):
        """
        前向传播
        
        参数:
            x: 输入词索引 [batch, seq_len]
            lengths: 每个样本的真实长度 [batch]
        
        返回:
            logits: 分类分数 [batch, num_classes]
        """
        # 步骤1: 词嵌入
        # [batch, seq_len] → [batch, seq_len, embed_dim]
        emb = self.embedding(x)
        
        # 步骤2: RNN处理
        # [batch, seq_len, embed_dim] → [batch, seq_len, hidden_size]
        output, hidden = self.rnn(emb)
        
        # 步骤3: 取最后一个有效时间步的输出
        # 为什么不用hidden[-1]？因为hidden包含了padding的信息
        # 我们需要手动取每个样本的真实最后一个时间步
        batch_size = x.size(0)
        idx = (lengths - 1).to(x.device)  # 每个样本的真实最后位置
        
        # 高级索引: 取output[i, idx[i]]
        last_output = output[
            torch.arange(batch_size, device=x.device), 
            idx
        ]  # [batch, hidden_size]
        
        # 步骤4: 分类
        # [batch, hidden_size] → [batch, num_classes]
        logits = self.fc(last_output)
        
        return logits

# 创建模型
vocab_size = len(word_to_index)
model = NewsClassifierRNN(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=5
).to(DEVICE)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 步骤 8：训练与评估（30分钟）
```python
# ============================================================
# 5. 训练与评估
# ============================================================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（5分类）
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    """
    评估模型
    
    返回:
        avg_loss: 平均损失
        acc: 准确率
        f1: F1分数（macro）
        all_preds: 所有预测
        all_labels: 所有真实标签
    """
    model.eval()  # 评估模式
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():  # 不计算梯度
        for x, y, lengths in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # 前向传播
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            # 预测
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1, all_preds, all_labels

# 训练循环
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()  # 训练模式
    total_loss = 0.0
    
    for x, y, lengths in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 前向传播
        logits = model(x, lengths)
        loss = criterion(logits, y)
        
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        total_loss += loss.item() * x.size(0)
    
    # 计算训练指标
    train_loss = total_loss / len(train_loader.dataset)
    
    # 评估
    val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"train_loss: {train_loss:.4f} | "
          f"val_acc: {val_acc:.4f} | "
          f"val_f1: {val_f1:.4f}")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_news_classifier.pth')
        print(f"  ✓ 保存最佳模型 (acc: {best_acc:.4f})")

print(f"\n训练完成！最佳准确率: {best_acc:.4f}")
```

### 步骤 9：测试与可视化（20分钟）

```python
# ============================================================
# 6. 测试与可视化
# ============================================================
# 加载最佳模型
model.load_state_dict(torch.load('best_news_classifier.pth'))

# 在测试集上评估
test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(model, test_loader)
print(f"\n测试集结果:")
print(f"  准确率: {test_acc:.4f}")
print(f"  F1分数: {test_f1:.4f}")

# 混淆矩阵
categories = ['体育', '财经', '娱乐', '科技', '家居']
cm = confusion_matrix(all_labels, all_preds)
print(f"\n混淆矩阵:")
print(f"          ", "  ".join([f"{cat:>4}" for cat in categories]))
for i, row in enumerate(cm):
    print(f"{categories[i]:>8}", "  ".join([f"{val:4d}" for val in row]))

# 预测示例
def predict(text):
    """预测单条新闻的类别"""
    model.eval()
    
    # 分词
    words = jieba.cut(text)
    
    # 转索引
    ids = [word_to_index.get(w, word_to_index['<unk>']) for w in words]
    
    # 截断 + 记录长度 + 填充
    ids = ids[:MAX_LEN]
    length = max(len(ids), 1)
    ids = ids + [word_to_index['<pad>']] * (MAX_LEN - len(ids))
    
    # 转tensor
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long)
    
    # 预测
    with torch.no_grad():
        logits = model(x, lengths)
        pred = torch.argmax(logits, dim=1).item()
    
    return categories[pred]

# 测试几个例子
examples = [
    "梅西带领阿根廷队夺得世界杯冠军",
    "苹果公司发布新款iPhone 16",
    "央行宣布降准0.5个百分点",
    "某明星宣布离婚，粉丝震惊",
    "家居装修小技巧分享",
]

print("\n预测示例:")
for text in examples:
    pred = predict(text)
    print(f"  {text} → {pred}")
```

---

## 🎓 进阶挑战（可选）

完成基础版本后，尝试以下改进：

### 挑战 1：使用 LSTM 替换 RNN

```python
# 将 nn.RNN 替换为 nn.LSTM
self.lstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
)

# forward 中修改
output, (hidden, cell) = self.lstm(emb)
```

### 挑战 2：添加 Dropout 防止过拟合

```python
# 在 __init__ 中添加
self.dropout = nn.Dropout(0.5)

# 在 forward 中使用
emb = self.dropout(emb)
last_output = self.dropout(last_output)
```

### 挑战 3：使用双向 RNN

```python
# 在 __init__ 中设置 bidirectional=True
self.rnn = nn.RNN(
    input_size=embed_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,  # 双向RNN
)

# 修改全连接层的输入维度
self.fc = nn.Linear(hidden_size * 2, num_classes)  # 乘以2
```

### 挑战 4：使用预训练词向量

```python
# 加载预训练的词向量（如 Word2Vec）
# 初始化 Embedding 层的权重
embedding_weights = torch.tensor(pretrained_vectors)
self.embedding = nn.Embedding.from_pretrained(embedding_weights)
```

---

## 📊 预期结果

完成课题后，你应该达到以下指标：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 训练准确率 | > 95% | 模型在训练集上的表现 |
| 测试准确率 | > 85% | 模型在测试集上的表现 |
| F1 分数 | > 0.80 | 综合考虑精确率和召回率 |
| 训练时间 | 5-10分钟 | 取决于硬件 |

---

## 💡 常见问题

**Q1: 为什么新闻分类用5分类，情感分类用2分类？**
A: 因为任务不同。情感分类是二分类（好/差），新闻分类是多分类（5个类别）。

**Q2: 为什么新闻标题用 MAX_LEN=50，评论用 MAX_LEN=100？**
A: 新闻标题通常较短，50个词足够。评论可能更长，需要100个词。

**Q3: 为什么用2层RNN而不是1层？**
A: 多层RNN可以提取更复杂的特征。新闻分类比情感分类更复杂，需要更强的表达能力。

**Q4: 如果准确率很低怎么办？**
A: 检查以下几点：
1. 数据是否正确加载和预处理
2. 词表是否合理（过滤低频词）
3. 模型是否过拟合（添加dropout）
4. 学习率是否合适（尝试调整LR）
5. 训练轮数是否足够（增加EPOCHS）

---

## 🎉 完成标志

当你能够：
1. ✅ 独立运行完整代码
2. ✅ 理解每一行代码的作用
3. ✅ 修改参数并解释影响
4. ✅ 达到预期的准确率指标
5. ✅ 完成至少一个进阶挑战

恭喜你！你已经掌握了 RNN 的基础应用，可以开始学习更高级的模型了！

---

## 📚 下一步学习路线

```
RNN（你已完成）
  ↓
LSTM/GRU（解决长期依赖问题）
  ↓
Attention（注意力机制）
  ↓
Transformer（当前主流架构）
  ↓
BERT/GPT（预训练模型）
```

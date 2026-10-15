---
title: 完整rnn代码
date: 2026-10-10 15:29:07
tags:
---

```python
# ============================================================
# 中文情感分类 (好评/差评) —— 教学版
# 数据集: ChnSentiCorp (酒店评论)
# 模型:   Embedding → RNN → Linear
# 作者提示: 从上到下顺序阅读, 每一段都有"为什么"的注释
# ============================================================

# ---------- 0. 导入依赖 ----------
import time                                              # 计时, 看每个 epoch 多快
import torch                                             # PyTorch 主库
import torch.nn as nn                                    # 网络层 / 损失函数
import torch.optim as optim                              # 优化器
from torch.utils.data import Dataset, DataLoader         # 数据集抽象 + 批量加载
from datasets import load_dataset                        # HuggingFace 数据集库
import jieba                                             # 中文分词
from collections import Counter                          # 词频统计
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # 评估指标


# ============================================================
# 1. 全局配置: 所有超参数集中放这里, 方便调参
# ============================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN     = 100      # 一条评论最多保留 100 个词, 超了截断, 短了补齐
BATCH_SIZE  = 32       # 每批 32 条样本
EMBED_DIM   = 128      # 每个词映射成 128 维向量
HIDDEN_SIZE = 256      # RNN 隐藏状态 256 维
NUM_LAYERS  = 1        # RNN 只堆 1 层 (多了容易过拟合 + 慢)
EPOCHS      = 10       # 训练 10 轮
LR          = 1e-3     # 学习率 0.001
MIN_FREQ    = 2        # 词频 <2 的一律当 <unk>, 过滤噪声
MAX_VOCAB   = 20000    # 词表最多 20000 个词, 防止太大

print(f"Using device: {DEVICE}")   # 打印用的 GPU 还是 CPU


# ============================================================
# 2. 加载数据集
#    ChnSentiCorp 结构:
#      train      : 训练集 (约 9600 条)
#      validation : 验证集 (约 1200 条)
#      test       : 测试集 (约 1200 条)
#    每条样本是 {'text': '...', 'label': 1 或 0}  1=好评 0=差评
# ============================================================
print("Loading ChnSentiCorp ...")
dataset = load_dataset(
    path="lansinuote/ChnSentiCorp",   # 数据集名字, 会自动下载
    cache_dir="./data",               # 下载后缓存在本地 ./data 目录
    trust_remote_code=True            # 允许执行数据集自带的加载脚本
)
print(dataset)                         # 打印数据集结构, 方便检查


# ============================================================
# 3. 构建词表
#    核心思想: 把"词"变成"整数 ID", 模型只认数字
#    流程: 分词 → 统计词频 → 按频率从高到低编号
# ============================================================
def build_vocab(texts, min_freq=2, max_size=20000):
    """
    texts    : 训练集所有文本 (只用训练集! 用别的会数据泄露)
    min_freq : 词频低于该值就不收录, 直接当 <unk>
    max_size : 词表上限, 防止词表爆炸
    返回     : {词: ID}
    """
    counter = Counter()                    # 空计数器
    for text in texts:                     # 遍历每条评论
        words = jieba.lcut(text)           # jieba 分词, lcut 返回 list
        counter.update(words)              # 把这一句的词加到计数器里

    # 两个特殊符号: <pad> 补齐, <unk> 未知词
    word_to_index = {'<pad>': 0, '<unk>': 1}

    # most_common(): 按词频从高到低遍历 (词, 次数)
    for word, freq in counter.most_common():
        if freq < min_freq:                # 频率不达标
            break                          # 因为已经排好序, 后面只会更低 → 直接 break
        if len(word_to_index) >= max_size: # 词表满了
            break
        if word not in word_to_index:      # 还没收录过
            word_to_index[word] = len(word_to_index)  # ID = 当前词表长度

    return word_to_index


train_texts = dataset['train']['text']                             # 只取训练集文本
word_to_index = build_vocab(train_texts, MIN_FREQ, MAX_VOCAB)      # 建词表
vocab_size = len(word_to_index)                                    # 词表大小
print(f"Vocabulary size: {vocab_size}")


# ============================================================
# 4. 自定义 Dataset
#    PyTorch 要求实现 __len__ 和 __getitem__ 两个方法
#    作用: 把 (文本, 标签) 转成 (ID张量, 标签张量, 长度)
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word_to_index, max_len):
        self.texts = texts                             # 原始文本列表
        self.labels = labels                           # 原始标签列表
        self.word_to_index = word_to_index             # 词表
        self.max_len = max_len                         # 最大长度
        self.pad_idx = word_to_index['<pad>']          # 0
        self.unk_idx = word_to_index['<unk>']          # 1

    def __len__(self):
        # DataLoader 需要知道总共有多少条样本
        return len(self.texts)

    def __getitem__(self, idx):
        # ---- 4.1 取一条样本 ----
        text = self.texts[idx]
        label = self.labels[idx]

        # ---- 4.2 分词 → 转 ID (没见过的词用 <unk>) ----
        words = jieba.lcut(text)
        ids = [self.word_to_index.get(w, self.unk_idx) for w in words]

        # ---- 4.3 截断 (太长会浪费算力) ----
        ids = ids[:self.max_len]

        # ---- 4.4 记录真实长度 (至少为 1, 防止空句子) ----
        # 后面 RNN 取"最后一个有效时间步"要用它
        length = max(len(ids), 1)

        # ---- 4.5 补齐 (太短要 pad 到 max_len, 才能凑成矩阵) ----
        ids = ids + [self.pad_idx] * (self.max_len - len(ids))

        return (
            torch.tensor(ids, dtype=torch.long),       # 输入 ID
            torch.tensor(label, dtype=torch.long),     # 标签
            torch.tensor(length, dtype=torch.long),    # 真实长度
        )


# 构造三个 Dataset
train_dataset = SentimentDataset(dataset['train']['text'],      dataset['train']['label'],      word_to_index, MAX_LEN)
val_dataset   = SentimentDataset(dataset['validation']['text'], dataset['validation']['label'], word_to_index, MAX_LEN)
test_dataset  = SentimentDataset(dataset['test']['text'],       dataset['test']['label'],       word_to_index, MAX_LEN)

# 用 DataLoader 包起来: 自动批量 + 打乱 + 多进程 (这里单进程)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)   # 训练集打乱
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)  # 验证集不打乱
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)  # 测试集不打乱


# ============================================================
# 5. 模型定义
#    数据流:  (B, L) → Embedding → (B, L, E) → RNN → (B, L, H) → 取最后有效步 → (B, H) → Linear → (B, 2)
# ============================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes=2):
        super().__init__()

        # 5.1 词嵌入: 每个 ID 换成 embed_dim 维向量
        #     padding_idx=0 表示 <pad> 的向量恒为 0, 且不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 5.2 RNN 层
        #     batch_first=True → 输入形状是 (batch, seq_len, embed_dim)
        #     默认非线性激活是 tanh
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 5.3 全连接: 把 hidden_size 维特征映射成 2 类得分
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len) 的整数张量
        emb = self.embedding(x)                 # (B, L)    → (B, L, E)

        # RNN 前向
        # output: (B, L, H)  每个时间步的隐状态
        # hidden: (num_layers, B, H)  最后一个时间步的隐状态
        output, hidden = self.rnn(emb)

        # 5.4 关键技巧: 取"最后一个有效时间步"的 output
        #     为什么不用 hidden[-1]? 因为 padding 会污染 hidden
        #     (RNN 会把 <pad> 也读进去), 所以要精确定位每条样本的真实末尾
        batch_size = x.size(0)
        idx = (lengths - 1).to(x.device)        # 真实末尾的下标 (B,)
        last_output = output[
            torch.arange(batch_size, device=x.device),   # 行索引: 0..B-1
            idx                                          # 列索引: 每条的真实末尾
        ]                                        # (B, H)

        # 5.5 过全连接得到 2 类 logits (未经过 softmax)
        logits = self.fc(last_output)            # (B, 2)
        return logits


# 实例化模型并搬到设备上
model = SentimentRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)


# ============================================================
# 6. 损失函数 & 优化器
# ============================================================
criterion = nn.CrossEntropyLoss()                       # 内部自带 softmax, 输入是 logits
optimizer = optim.Adam(model.parameters(), lr=LR)       # Adam 优化器, 自适应学习率


# ============================================================
# 7. 评估函数 (训练中 & 测试时都会用)
# ============================================================
def evaluate(model, loader):
    """返回: 平均 loss, 准确率, F1, 所有预测, 所有真实标签"""
    model.eval()                                # 切评估模式: 关闭 Dropout/BN
    total_loss = 0.0
    all_preds, all_labels = [], []              # 收集所有预测 / 标签, 最后算指标

    with torch.no_grad():                       # 关闭梯度: 省显存 + 提速
        for x, y, lengths in loader:            # 逐批取
            x, y = x.to(DEVICE), y.to(DEVICE)   # 搬到设备
            logits = model(x, lengths)          # 前向
            loss = criterion(logits, y)         # 算损失
            total_loss += loss.item()           # 累加

            preds = torch.argmax(logits, dim=1) # 取分高的类别
            all_preds.extend(preds.cpu().numpy())   # 搬到 CPU 再转 numpy
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)         # 平均损失
    acc = accuracy_score(all_labels, all_preds) # 准确率
    f1  = f1_score(all_labels, all_preds)       # F1 (类别不均衡时更可信)
    return avg_loss, acc, f1, all_preds, all_labels


# ============================================================
# 8. 训练循环
# ============================================================
for epoch in range(EPOCHS):
    model.train()                              # 切训练模式: 开启 Dropout
    start = time.time()                        # 记录开始时间
    total_loss = 0.0

    for x, y, lengths in train_loader:         # 逐批
        x, y = x.to(DEVICE), y.to(DEVICE)      # 搬设备

        logits = model(x, lengths)             # 前向: (B, 2)
        loss = criterion(logits, y)            # 算损失

        optimizer.zero_grad()                  # 清空上一轮梯度 (必须!)
        loss.backward()                        # 反向传播
        optimizer.step()                       # 更新参数

        total_loss += loss.item()              # 累加

    train_loss = total_loss / len(train_loader)                     # 平均训练损失
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)   # 验证

    # 打印本轮结果
    print(
        f"Epoch {epoch + 1:>2}/{EPOCHS} | "
        f"time: {time.time() - start:.2f}s | "
        f"train_loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"val_acc: {val_acc:.4f} | "
        f"val_f1: {val_f1:.4f}"
    )


# ============================================================
# 9. 测试集最终评估
# ============================================================
test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader)

print("\n========== Test Results ==========")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1:       {test_f1:.4f}")
print("Confusion Matrix:")                       # 混淆矩阵: 看哪类容易错
print(confusion_matrix(test_labels, test_preds))  # 行=真实, 列=预测


# ============================================================
# 10. 单句预测函数 (推理接口)
# ============================================================
def predict(text):
    """输入一句话, 返回 '好评' 或 '差评'"""
    model.eval()                                    # 评估模式

    # ---- 复刻 Dataset 里的预处理, 保证训练/推理一致 ----
    words = jieba.lcut(text)                        # 分词
    ids = [word_to_index.get(w, word_to_index['<unk>']) for w in words]   # 转 ID
    ids = ids[:MAX_LEN]                             # 截断
    length = max(len(ids), 1)                       # 真实长度
    ids = ids + [word_to_index['<pad>']] * (MAX_LEN - len(ids))          # 补齐

    # ---- 组 batch, 形状要 (1, MAX_LEN) ----
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():                           # 推理不用梯度
        logits = model(x, lengths)                  # (1, 2)
        pred = torch.argmax(logits, dim=1).item()   # 取类别

    return "好评" if pred == 1 else "差评"          # 1=好评 0=差评


# 测试几个句子
examples = [
    "这家餐厅味道很好，服务也很周到，下次还会再来！",
    "菜太咸了，服务员态度也差，再也不会来了。",
    "东西还行吧，没什么特别的感觉。",
    "质量很好，物流很快，非常满意！",
    "包装破损了，里面的东西也坏了，差评。",
]

print("\n========== 预测示例 ==========")
for text in examples:
    print(f"  {text}  ->  {predict(text)}")
```
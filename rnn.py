"""
任务: 中文情感分类 (好评 / 差评)
数据集: ChnSentiCorp
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


# ============================================================
# 0. 全局配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 100          # 每条评论最多保留多少个词
BATCH_SIZE = 32
EMBED_DIM = 128        # 词向量维度
HIDDEN_SIZE = 256      # RNN 隐藏层维度
NUM_LAYERS = 1         # RNN 层数
EPOCHS = 10
LR = 1e-3
MIN_FREQ = 2           # 词频低于 2 的词, 统一当作 <unk>
MAX_VOCAB = 20000      # 词表最大容量

print(f"Using device: {DEVICE}")


# ============================================================
# 1. 加载数据集
# ============================================================
print("Loading ChnSentiCorp ...")
dataset = load_dataset(
    path="lansinuote/ChnSentiCorp",
    cache_dir="./data",
    trust_remote_code=True
)
print(dataset)


# ============================================================
# 2. 构建词表
# ============================================================
def build_vocab(texts, min_freq=2, max_size=20000):
    """
    遍历所有训练文本, 用 jieba 分词, 统计词频,
    返回: {词: 索引}
    """
    counter = Counter()
    for text in texts:
        words = jieba.cut(text)
        counter.update(words)

    # 两个特殊 token
    word_to_index = {'<pad>': 0, '<unk>': 1}

    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(word_to_index) >= max_size:
            break
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    return word_to_index


train_texts = dataset['train']['text']
word_to_index = build_vocab(train_texts, MIN_FREQ, MAX_VOCAB)
vocab_size = len(word_to_index)
print(f"Vocabulary size: {vocab_size}")


# ============================================================
# 3. 自定义 Dataset
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word_to_index, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_len = max_len
        self.pad_idx = word_to_index['<pad>']
        self.unk_idx = word_to_index['<unk>']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 3.1 取一条文本和标签
        text = self.texts[idx]
        label = self.labels[idx]

        # 3.2 分词 -> 转索引
        words = jieba.cut(text)
        ids = [self.word_to_index.get(w, self.unk_idx) for w in words]

        # 3.3 截断到 max_len
        ids = ids[:self.max_len]

        # 3.4 记录真实长度 (至少为 1, 防止空句子)
        length = max(len(ids), 1)

        # 3.5 padding 到 max_len
        ids = ids + [self.pad_idx] * (self.max_len - len(ids))

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


train_dataset = SentimentDataset(
    dataset['train']['text'], dataset['train']['label'],
    word_to_index, MAX_LEN
)
val_dataset = SentimentDataset(
    dataset['validation']['text'], dataset['validation']['label'],
    word_to_index, MAX_LEN
)
test_dataset = SentimentDataset(
    dataset['test']['text'], dataset['test']['label'],
    word_to_index, MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 4. 模型定义
# ============================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes=2):
        super().__init__()
        # 4.1 词嵌入层: 词索引 -> 词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 4.2 RNN 层: batch_first=True, 输入 (batch, seq_len, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # 4.3 输出层: 隐藏状态 -> 2 类分数
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len)
        emb = self.embedding(x)                 # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(emb)          # output: (batch, seq_len, hidden)
                                                # hidden: (num_layers, batch, hidden)

        # 4.4 取每个样本 "最后一个有效时间步" 的 output
        #     不能直接取 hidden[-1], 因为 padding 会让 hidden 变脏
        batch_size = x.size(0)
        idx = (lengths - 1).to(x.device)        # (batch,)
        last_output = output[
            torch.arange(batch_size, device=x.device), idx
        ]                                        # (batch, hidden)

        # 4.5 过全连接, 得到 2 类 logits
        logits = self.fc(last_output)            # (batch, 2)
        return logits


model = SentimentRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)


# ============================================================
# 5. 训练 & 评估函数
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def evaluate(model, loader):
    """返回: 平均 loss, 准确率, F1, 所有预测, 所有真实标签"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1, all_preds, all_labels


# ============================================================
# 6. 训练循环
# ============================================================
for epoch in range(EPOCHS):
    model.train()
    start = time.time()
    total_loss = 0.0

    for x, y, lengths in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x, lengths)          # (batch, 2)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)

    print(
        f"Epoch {epoch + 1:>2}/{EPOCHS} | "
        f"time: {time.time() - start:.2f}s | "
        f"train_loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"val_acc: {val_acc:.4f} | "
        f"val_f1: {val_f1:.4f}"
    )


# ============================================================
# 7. 测试集评估
# ============================================================
test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader)

print("\n========== Test Results ==========")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1:       {test_f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))


# ============================================================
# 8. 预测示例
# ============================================================
def predict(text):
    """输入一句话, 返回 '好评' 或 '差评'"""
    model.eval()

    words = jieba.cut(text)
    ids = [word_to_index.get(w, word_to_index['<unk>']) for w in words]
    ids = ids[:MAX_LEN]
    length = max(len(ids), 1)
    ids = ids + [word_to_index['<pad>']] * (MAX_LEN - len(ids))

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        pred = torch.argmax(logits, dim=1).item()

    return "好评" if pred == 1 else "差评"


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



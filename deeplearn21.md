---
title: 动手实践(自己制作数据)
date: 2026-10-05 20:14:37
tags:
---
```python

# 导包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split 

# 种子
torch.manual_seed(42)
np.random.seed(42)

# 模拟数据
centers = [[2, 2, 2], [-2, -2, -2], [2, -2, 2], [-2, 2, -2]]
X = []
y = []

for i, center in enumerate(centers):
    # 每个类别生成200个样本，高斯噪声
    X_class = np.random.randn(200, 3) * 0.5 + np.array(center)
    y_class = np.full(200, i)  # 标签为 0, 1, 2，3
    X.append(X_class)
    y.append(y_class)

# 合并所有类别
X = np.vstack(X)  # 形状 (800, 3)
y = np.concatenate(y)  # 形状 (800,)

# 转换为 PyTorch 张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 划分训练集和测试集（80% 训练，20% 测试）
indices = torch.randperm(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

#自动切分
# 划分训练集和测试集（80% 训练，20% 测试），分层抽样保持类别比例
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, train_size=0.8, random_state=88, stratify=y)


# 3. 自定义 Dataset 和 DataLoader
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据集对象
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

# 创建 DataLoader，用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 4. 定义模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # 第1层:输入维度：3，输出维度：64
        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 一维批量归一化
        self.act = nn.LeakyReLU(0.01)  # 激活函数
        self.dropout = nn.Dropout(0.5)  # 随机失活

        # 第二层，输入维度：64，输出维度：64
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)  # 一维批量归一化

        # 第三层，输入维度：64，输出维度：32
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        # 输出层：32 -> 4（4个类别）
        self.fc4 = nn.Linear(32, 4)

        # kaiming初始化
        # 对每个线性层的权重使用 Kaiming 正态初始化，适用于 LeakyReLU
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)  # 偏置初始化为 0

        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc2.bias)

        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc3.bias)

        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc4.bias)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x


# 实例化模型
model = MyNet()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 优化器Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ai要求的
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 训练模型
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        # 前向传播
        outputs = model(x_batch)

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 反向传播及优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        running_loss += loss.item()

        # 学习率衰减
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

# 模型评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        # 获取预测类别（最大 logit 对应的索引）
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    acc = correct / total
    print(
        f'Epoch {epoch + 1:2d} | Loss: {running_loss / len(train_loader):.4f} | LR: {current_lr:.6f} | Test Acc: {acc:.4f}')

print('\n训练结束')

# 7. 保存模型参数
torch.save(model.state_dict(), 'mynet_params.pth')
print('模型参数已保存至 mynet_params.pth')

# 8. 加载模型参数并在测试集上评估

# 新建一个结构相同的模型，加载刚才保存的参数
loaded_model = MyNet()
loaded_model.load_state_dict(torch.load('mynet_params.pth'))
loaded_model.eval()  # 关闭 Dropout，BN 用运行统计量

correct = 0
total = 0
num_classes = 4
class_correct = [0] * num_classes
class_total = [0] * num_classes

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = loaded_model(x_batch)
        _, predicted = torch.max(outputs, dim=1)

        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        # 逐样本统计每个类别
        for label, pred in zip(y_batch, predicted):
            label = label.item()
            class_total[label] += 1
            if pred.item() == label:
                class_correct[label] += 1

overall_acc = correct / total
print('\n========== 测试集评估结果 ==========')
print(f'样本总数  : {total}')
print(f'正确个数  : {correct}')
print(f'错误个数  : {total - correct}')
print(f'整体正确率: {overall_acc:.4f} ({overall_acc * 100:.2f}%)')

print('\n---------- 各类别准确率 ----------')
for c in range(num_classes):
    if class_total[c] > 0:
        acc_c = class_correct[c] / class_total[c]
        print(f'类别 {c}: 正确 {class_correct[c]:3d} / 总数 {class_total[c]:3d} '
              f'→ 准确率 {acc_c:.4f} ({acc_c * 100:.2f}%)')
    else:
        print(f'类别 {c}: 测试集中无样本')
```

真实项目里是不用生成数据集这个流程，但必须转张量。

切分数据集必须做，而且很重要。
通常用 sklearn 的工具或者 PyTorch 自带的：
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

定义Dataset、DataLoader和创建对象完全是套路，甚至是固定模板。
1. 自定义 Dataset 类

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):  # 或者传入文件路径、DataFrame等
        self.X = X
        self.y = y

    def __len__(self):         # 必须返回数据总量
        return len(self.X)

    def __getitem__(self, idx): # 必须返回 (特征, 标签)
        # 真实项目里，这里往往是：读取图片、做数据增强、转为Tensor
        return self.X[idx], self.y[idx]
```

2. 实例化 Dataset

```python
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
```

3. 实例化 DataLoader（设好 batch_size 和 shuffle）

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集要打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试集不用打乱
```

4. 在训练循环里遍历
```python
for x_batch, y_batch in train_loader:
    # 把数据送入模型...
```
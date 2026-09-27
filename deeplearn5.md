---
title: 深度学习的小demo展示（激励一下）
date: 2026-09-26 19:35:00
tags:
---

基于 ResNet18 迁移学习的图像分类系统

流程介绍
数据集：Oxford Flowers102（torchvision 自带，102 类）或 Kaggle Dogs vs Cats
模型：ResNet18 / ResNet50 预训练模型
框架：PyTorch + torchvision
技术点：数据增强、迁移学习、冻结-解冻、余弦退火、早停、混淆矩阵
结果：验证集准确率 90% 左右
部署：Gradio / Streamlit，上传图片输出 Top-3 类别和置信度
开源：GitHub README + 结果截图 + Demo 链接

加载数据集
Flowers102 是 torchvision 内置数据集，包含 102 类花卉。数据划分如下
train	1,020（每类 10 张）
val	1,020（每类 10 张）
test	6,149（每类至少 20 张）

注意：训练集只有 1020 张，这就是为什么要用迁移学习——从零训练在这么少的数据上很难收敛。

数据增强与预处理（训练集用增强，验证/测试集不用）：
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 训练集增强
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证/测试集仅 Resize + Normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_set = datasets.Flowers102(root='./data', split='train',
                                 download=True, transform=train_transform)
val_set = datasets.Flowers102(root='./data', split='val',
                               download=True, transform=val_transform)
test_set = datasets.Flowers102(root='./data', split='test',
                                download=True, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
```

num_workers 设为 4，可根据 CPU 核心数调整。

第三步：加载 ResNet18 并改造模型
```python
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #兼容cpu

# 加载 ImageNet 预训练 ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结所有预训练层
for param in model.parameters():
    param.requires_grad = False

# 替换最后的全连接层：512 -> 102
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 102)

model = model.to(device)
```

冻结所有层后，仅 fc 层的参数可训练，可训练参数量从约 1100 万降到约 5.2 万，可以降低了在小数据集上过拟合的风险。

第四步：阶段一训练（只训练分类头）
```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=5)

best_val_acc = 0.0
patience = 3
no_improve = 0

for epoch in range(5):
    # 训练
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    # 验证
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    scheduler.step()
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 提前叫停
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), 'best_model_stage1.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```
余弦退火调度器让学习率按余弦曲线从初始值逐渐降到 0，帮助模型在后期更稳定地收敛。


第五步：阶段二训练——解冻部分层微调
# 加载阶段一最优权重
```python
model.load_state_dict(torch.load('best_model_stage1.pth'))

# 解冻 layer4（ResNet18 最后一个残差块）
for param in model.layer4.parameters():
    param.requires_grad = True

# 用更小的学习率微调
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

best_val_acc = 0.0
no_improve = 0

for epoch in range(10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    scheduler.step()
    print(f"[Fine-tune] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

layer4 包含最高层的语义特征，与花卉分类最相关，故而解冻 layer4 而不是全部解冻；底层卷积层提取的是通用边缘/纹理特征，保留冻结即可。全解冻反而容易在小数据集上过拟合。


第六步：评估
测试集准确率：
```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy: {test_correct / test_total:.4f}")

混淆矩阵：
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
print(classification_report(all_labels, all_preds, digits=4))

plt.figure(figsize=(20, 20))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)
```

预期结果：测试集准确率约 90%-95%。有研究显示，用 ImageNet 预训练的 ResNet18 微调花卉分类，20 分钟内可达到 90%+ 的准确率。另有一个 GitHub 项目对 Flowers102 做 ResNet18 微调，验证集准确率达到 100%，训练集 96.45%——说明这个任务的上限很高。（抄的，其实不用这么精确）


第七步：Gradio 部署
创建 app.py：
```python
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 模型定义（必须和训练时一致）
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 102)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    top3_prob, top3_idx = torch.topk(probs, 3)
    return {f"Class {idx.item()}": prob.item()
            for prob, idx in zip(top3_prob, top3_idx)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="花卉分类器",
    description="上传一张花卉图片，返回最可能的 3 个类别及置信度"
)

demo.launch()
```
本地运行 python app.py，浏览器打开 http://localhost:7860 测试。



第八步：部署到 HuggingFace Spaces
注册 HuggingFace 账号，访问 huggingface.co/new-space
SDK 选 Gradio，硬件选 CPU (free)，命名如 flower-classifier

上传以下文件：
app.py（上面的 Gradio 代码）
best_model.pth（训练好的权重）
requirements.txt
README.md

上传后在 Space 页面即可获得公开 Demo 链接


第九步：GitHub 仓库整理
flower-classification/
├── train.py              # 完整训练脚本
├── evaluate.py           # 评估脚本
├── app.py                # Gradio 推理
├── best_model.pth        # 训练好的权重
├── requirements.txt
├── README.md             # 项目说明
├── confusion_matrix.png  # 混淆矩阵截图
└── results/
    ├── training_curve.png
    └── sample_predictions.png


推荐README包含：任务描述、数据集信息、模型架构、训练策略（冻结-解冻两阶段）、最终指标（测试集准确率）、Demo 链接、运行方式。


常见问题（ai生成的可能的问题）
验证集准确率高于训练集	
训练集用了数据增强（更难），验证集没有；BatchNorm/Dropout 在 eval 模式下行为不同	

loss 不下降
冻结后只训练 fc，学习率太大（如 0.01），用 1e-3 或更小

解冻后准确率崩了
微调学习率太大，微调阶段用 1e-4 或 1e-5

CPU 训练太慢	
num_workers 没设或太大，可以设成4，并确保 pin_memory=True

Gradio 加载模型报错
加载权重时没定义相同的模型结构，要先定义 resnet18 和 fc 层，再 load_state_dict。
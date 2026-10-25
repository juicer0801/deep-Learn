---
title: 数据预处理
date: 2026-09-26 19:35:03
tags:
---
## 第 4 节：数据预处理

数据就是一个张量，但你拿到手的数据，十有八九不是干净的、可以直接喂给模型的张量。

真实的数据可能是这样的：CSV 里有一堆缺失值，某一列是字符串，图像大小参差不齐，类别分布极不均匀。模型本身再牛逼，数据没处理好，训出来的效果就是不行。

预备工作，使用过python的估计都对pandas这个包有印象，一个用于数据分析的包，而它与张量兼容。

## 4.1 表格数据的预处理：pandas + torch

### 造一个带脏数据的 CSV


```python
import os
import pandas as pd
import torch

os.makedirs("data", exist_ok=True)
data_file = os.path.join("data", "house_tiny.csv")

with open(data_file, "w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

data = pd.read_csv(data_file)
print(data)
```

输出：

```
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```

CSV 里的 `NA` 被 pandas 自动识别成了 `NaN`，也就是缺失值。

### 分离特征和标签

最后一列 `Price` 是我们要预测的目标，其他列是特征。

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)
```

`inputs` 包含 `NumRooms` 和 `Alley` 两列，`outputs` 是 `Price`。

### 处理缺失值

对于

一些常用的 baseline 策略

**数值特征**，比如 `NumRooms`，最常用的做法是用均值填充：

```python
inputs_num = inputs.iloc[:, 0]
inputs.iloc[:, 0] = inputs_num.fillna(inputs_num.mean())
```

为什么用均值？因为均值不改变该列的整体分布中心，是最保守的填充方式。缺点是会缩小方差，而且如果缺失值本身有信息（比如“收入未知”可能意味着“收入很低”或“很高”），均值填充会把这个信息抹掉。

**其他填充策略**：

- **中位数填充**：数据有极端值时比均值稳健。
- **众数填充**：类别特征常用。
- **前向填充 / 后向填充**：时间序列数据常用，用上一个或下一个有效值填。
- **常数填充**：比如填 0 或者 -1，如果缺失本身有含义就合适。
- **模型预测填充**：用其他特征训练一个小模型来预测缺失值，成本高但可能更准。

```python
# 中位数填充
inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].median())

# 常数填充
inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(0)
```

**注意**：填充的统计量（均值、中位数）只能用**训练集**算，不能把验证集和测试集混进来一起算。否则就发生了数据泄漏——模型在训练时“偷看”了测试集的信息，评估结果会虚高。(关于这个其实可以展开来讲，但这里篇幅有限，以后在专门来讲)

正确做法：

```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 用训练集的均值
train_mean = train_data['NumRooms'].mean()

train_data['NumRooms'] = train_data['NumRooms'].fillna(train_mean)
test_data['NumRooms'] = test_data['NumRooms'].fillna(train_mean)  # 用同一个均值
```

### 类别特征：one-hot 编码（独热编码）

`Alley` 列的值是 `Pave` 或 `NaN`，这是字符串，不能直接喂模型。

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

`get_dummies` 会把每个类别变成一个 0/1 列。`dummy_na=True` 表示把缺失值也当成一个单独的类别。所以 `Alley` 列会变成 `Alley_Pave` 和 `Alley_nan` 两列。（pandas自动转换），其中Alley列pave的行会将Alley_Pave的值设置为1，Alley_nan的值设置为0缺失Alley的行(Alley那一列中为NAN的)会将Alley_Pave设置为0，将Alley_nan设置为1.


**为什么要保留 NaN 作为一个类别？** 因为“这条数据里 Alley 信息缺失”本身可能是有信息量的。

**但有一种情况必须去掉一列**：如果类别特征有 K 个取值，one-hot 后产生 K 列，这 K 列加起来恒等于 1，存在完全共线性。对于线性模型来说这会出问题。解决办法是 `drop_first=True`，丢掉第一列。

```python
inputs = pd.get_dummies(inputs, dummy_na=True, drop_first=True)
```

对于神经网络，共线性问题没那么严重，但 `drop_first=True` 仍然是个好习惯，能减少参数量。

### 转成 Tensor

pandas 处理完，最后一步是转成 PyTorch 张量。

```python
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float)).reshape(-1, 1)

print(X.shape, y.shape)   # torch.Size([4, 3]) torch.Size([4, 1])
```

**注意** `dtype=float` 这个参数。如果不指定，one-hot 之后的 0/1 列可能被识别成整数，混合列的类型可能变成 object，转 tensor 会报错。显式指定 float 最安全。

`outputs` 加了 `.reshape(-1, 1)`，把它从一维变成二维。很多损失函数（比如 `MSELoss`）要求预测值和标签的形状一致，都是 `(batch_size, 1)`。

---

## 4.2 Dataset 和 DataLoader：PyTorch 的数据管道

上面处理的是数据内容本身。但**怎么把数据高效地喂给模型？**

如果数据集只有 4 条，直接循环就行。但实际数据集可能有几万、几十万条，甚至装不进内存。你不可能每次都手动切片、打乱、组装 batch。PyTorch 提供了方法解决这个问题：`Dataset` 和 `DataLoader`。

> **Dataset 回答“一条数据长什么样”，DataLoader 回答“怎么一batch一batch地获取”。**

### Dataset：定义一条数据

`Dataset` 是一个抽象类，你继承它，然后实现两个方法：

- `__len__`：数据集有多少条。
- `__getitem__(idx)`：给定索引，返回一条数据（具备：特征 + 标签）。

```python
from torch.utils.data import Dataset

class HouseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

`__getitem__` 的返回就是DataLoader的获得。

实际的 `__getitem__` 里通常会做更多事：读图像文件、做变换、tokenize 文本等等。
本质都是**索引变样本**

### DataLoader：批量加载

`DataLoader` 把 `Dataset` 包起来，提供批量加载、打乱、多进程读取等功能。

```python
from torch.utils.data import DataLoader

dataset = HouseDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_X, batch_y in dataloader:
    print(batch_X.shape, batch_y.shape)
```

`batch_size=2` 表示每次拿 2 条。`shuffle=True` 表示每个 epoch 开始前打乱顺序。
注意：**训练集必须 shuffle，验证集和测试集不要 shuffle。**

DataLoader作用：

- **自动 batching**：把多个单条样本拼成一个 batch 张量。
- **shuffle**：打乱顺序。
- **多进程加载**：`num_workers` 参数控制用几个进程读取数据。
- **自动拼接**：`__getitem__` 返回的多个样本，DataLoader 会用默认的 collate 函数把它们拼成 batch。

`num_workers` 的设置有个经验值：如果数据加载是瓶颈（比如要读大量小文件），设成 CPU 核心数左右。如果数据已经在内存里，设 0 就行。在 Windows 上多进程有时候会出问题，先用 `num_workers=0` 调试，没问题了再加。


### 一个完整的图像 Dataset 示例

图像分类是 Dataset 用得最多的场景。假设你有一个文件夹，里面是图片，还有一个 CSV 记录了每张图片的文件名和标签。

```python
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
```

`transform` ：**Dataset 负责“找到并读取一条数据”，transform 负责“把这条数据变成适合喂模型的样子”。**

---

## 4.3 图像变换：torchvision.transforms

表格数据可以用 pandas 处理，图像数据不行。图像的预处理包括：统一尺寸、转成张量、归一化、以及训练时的数据增强。PyTorch的操作都在`torchvision.transforms` 里。

### 最基本的变换

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 统一尺寸
    transforms.ToTensor(),                   # PIL Image → Tensor，像素值缩放到 [0, 1]
    transforms.Normalize(                    # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

`Compose` 把多个变换串起来，按顺序执行。
`ToTensor()` 把 PIL 图像或 NumPy 数组转成张量，同时把像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`。
`Normalize` 的 mean 和 std 是 ImageNet 数据集的统计值。如果用的是预训练模型，**必须用和预训练时相同的 mean/std**，否则输入分布不对，模型表现会下降。如果是从零训练，可以用自己数据集的 mean/std。

### 数据增强

上面那个 transform 是“确定性”的，每次对同一张图做的变换都一样。训练时我们还需要“随机性”的变换，也就是数据增强。目的是让模型看到更多样的数据，减少过拟合。

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),        # 随机裁剪并缩放到 224
    transforms.RandomHorizontalFlip(p=0.5),   # 50% 概率水平翻转
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),                                         # 随机调整颜色
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                # 验证时用中心裁剪，不用随机
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

注意：**训练和验证/测试的 transform 不一样**。训练时用随机增强，验证和测试时用确定性的变换。因为验证和测试需要的是稳定的、可复现的结果，不能每次评估结果都不一样。

常用的增强操作：

| 变换 | 作用 |
|---|---|
| `RandomHorizontalFlip` | 随机水平翻转 |
| `RandomCrop` / `RandomResizedCrop` | 随机裁剪 |
| `ColorJitter` | 随机调整亮度、对比度、饱和度、色相 |
| `RandomRotation` | 随机旋转 |
| `RandomGrayscale` | 随机转灰度 |

**这个变换会不会改变图片的语义？** 比如做猫狗分类，水平翻转没问题，猫翻过来还是猫。但做数字识别，6 翻转一下变成 9，就不行了。

### 把 transform 接进 Dataset

```python
train_dataset = ImageDataset("train.csv", "images/", transform=train_transform)
val_dataset = ImageDataset("val.csv", "images/", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

`Dataset` 读取数据，`transform` 处理数据，`DataLoader` 批量输出，模型 `DataLoader` 即可。
---

## 4.4 数据标准化

标准化可能是最容易被忽略，但影响很大的预处理步骤。

神经网络本质上是做梯度下降。如果不同特征的尺度差异很大——比如一个特征范围是 0 到 1，另一个是 0 到 10000——梯度在不同方向上的尺度差异巨大，训练会非常不稳定。大的那个特征会主导梯度，小的那个特征几乎学不到东西。

**标准化的目标：让每个特征都变成均值 0、标准差 1。**

```text
x_standardized = (x - mean) / std
```

用 sklearn 的 `StandardScaler` 是最方便的做法：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 只在训练集上 fit
X_train_scaled = scaler.fit_transform(X_train)

# 用同一个 scaler 变换验证集和测试集
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**注意**：`fit_transform` 只在训练集上用，验证集和测试集只用 `transform`。如果对测试集也用 `fit_transform`，就是用测试集的统计量重新标准化了一遍，造成了数据泄漏。

图像数据的标准化在 `Normalize` 变换里已经做了，原理一样：`(pixel/255 - mean) / std`。

**另一种常见的缩放是 Min-Max 归一化**，把数据缩放到 `[0, 1]`：

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

Min-Max 适合数据分布不是正态的情况，或者你需要严格保证输出在某个范围内（比如图像像素）。缺点是受极端值影响很大——一个极大的 outlier 会把所有其他值压缩到很小的范围里。

**选哪个？** 大多数情况下用 `StandardScaler`，因为它对 outlier 更稳健，而且假设数据近似正态分布时效果最好。如果数据有明确的边界（比如像素值），用 Min-Max。

---

## 4.5 数据集的划分

原始数据并不能直接全部拿去训练。你需要把数据分成三份：

- **训练集**：模型学习用。通常占 60%–80%。
- **验证集**：调超参数、选模型、早停用。通常占 10%–20%。
- **测试集**：最终评估用，只在最后跑一次。通常占 10%–20%。

常见的划分比例是 70/15/15 或 80/10/10。数据量大的时候，验证集和测试集的比例可以更小。

用 sklearn 的 `train_test_split` 可以分两次完成：

```python
from sklearn.model_selection import train_test_split

# 先分训练集和临时集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 再从临时集里分验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

`stratify=y` 这个参数很重要。它保证划分后每一类的比例和原始数据一致。对于类别不平衡的数据集，不加 `stratify` 可能导致某一类在验证集里一条都没有。

**注意**：划分之前就做了标准化或者填充缺失值。这是常见数据泄漏。
正确顺序是：**先划分，再在训练集上 fit，然后用训练集的统计量去 transform 验证集和测试集。**

---

## 4.6 类别不平衡

类别不平衡的处理方法主要有三类：

**第一类：重采样。** 让各类样本数量接近。

- **过采样**：复制少数类样本，或者用 SMOTE 生成新的少数类样本。SMOTE 的思路是在少数类样本之间做插值，生成“合成”的新样本。缺点是有时生成的样本不真实，可能引入噪声。
- **欠采样**：随机丢弃多数类样本。缺点是丢掉了大量信息，数据量本来就少的时候慎用。
- **混合**：过采样和欠采样结合。

**第二类：改损失函数。** 不改变数据，改变模型对错误的惩罚。

- **类别权重**：给少数类更高的权重。PyTorch 的 `CrossEntropyLoss` 支持 `weight` 参数。

```python
import torch.nn as nn

# 假设类别 0 有 9900 条，类别 1 有 100 条
# 权重可以设为数量的反比
class_weights = torch.tensor([1.0/9900, 1.0/100])
class_weights = class_weights / class_weights.sum() * 2   # 归一化

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

- **Focal Loss**：专门为极度不平衡设计的损失函数，降低容易分类样本的权重，让模型聚焦在难分类的样本上。

**第三类：换评估指标。**使用 Precision、Recall、F1 或者 AUC。

一个全预测多数的模型准确率 99%，但 F1 可能是 0。评估的时候一定要看每个类别的 Precision 和 Recall，或者看混淆矩阵。

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

---

## 4.7 一个完整的预处理流程
一个典型的有监督学习项目的预处理流程大概长这样：

**第一步：读数据。**

```python
data = pd.read_csv("data.csv")
```

**第二步：清洗。**

- 删除重复行：`data.drop_duplicates()`
- 处理异常值：根据业务逻辑或统计方法（比如 3σ 法则、IQR）
- 统一格式：日期、单位、编码

**第三步：划分数据。**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

**第四步：在训练集上 fit 预处理器。**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**第五步：处理缺失值、编码类别特征。** 同样的原则：fit 在训练集，transform 在所有集。

**第六步：封装成 Dataset 和 DataLoader。**

```python
train_dataset = MyDataset(X_train_scaled, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**第七步：训模型。** 这一步是下一节的内容。

---

## 本节小结

- **表格数据用 pandas**：读 CSV、填缺失、one-hot 编码、转张量。
- **Dataset 定义一条数据长什么样**，`__len__` 和 `__getitem__` 是两个必须实现的方法。
- **DataLoader 负责批量加载**：batching、shuffle、多进程。
- **图像用 torchvision.transforms**：训练用随机增强，验证/测试用确定性变换。
- **标准化让训练更稳**：fit 只在训练集，transform 用在所有集。
- **先划分再预处理**：任何 fit 操作都只能用训练集，否则数据泄漏。
- **类别不平衡要看 F1 和 Recall**：准确率不一定准确。

不同数据集、不同任务、不同模型，适合的处理方式都不一样。但有一条原则是通用的：**训练集和测试集的处理方式必须完全一致，而且测试集不能参与任何 fit 操作。**
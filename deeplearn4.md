---
title: pytorch基本操作
date: 2026-09-26 19:34:56
tags:
---

打开 Jupyter Notebook

import torch

# 创建张量
## 1.
```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x)
print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.int64

torch.tensor() 会复制数据。如果你有个 NumPy 数组，不想复制，用 torch.from_numpy()。
```

## 2.
```python
a = torch.zeros((2, 3, 4))      # 全零，形状 (2, 3, 4)
b = torch.ones((5, 3))          # 全一，形状 (5, 3)
c = torch.randn((2, 2))         # 标准正态分布随机数
d = torch.arange(12)            # 0 到 11 的整数，形状 (12,)
e = torch.linspace(0, 1, 5)     # 0 到 1 之间均匀取 5 个点
```

torch.arange(12) 得到的是 tensor([0, 1, 2, ..., 11])，左闭右开，跟 Python 的 range 一个逻辑
默认整数，可指定，这个张量有12个元素

## 3.
```python
i = torch.ones((5, 3), dtype=torch.int16)
gpu_tensor = torch.randn(3, 3, device='cuda:0')  # 直接在 GPU 上造
```

默认的浮点类型是 torch.float32，这也是大多数情况下的最佳选择。如果你想用 64 位浮点数，得手动指定 dtype=torch.float64

## 4.从另一个张量造

```python
x = torch.randn(3, 4)
y = torch.zeros_like(x)    # 跟 x 同形状的全零张量
z = torch.rand_like(x)     # 跟 x 同形状的随机张量
```

*_like 系列在写模型的时候特别常见，比如你想初始化一个跟某个参数同形状的零张量，直接用 torch.zeros_like(param) 就行。


# 算数运算
## 加减乘除
张量的运算跟 NumPy 几乎一模一样。两个同形状的张量，加减乘除都是按元素算的：
```python
x = torch.tensor([1.0, 2.0, 4.0, 8.0])
y = torch.tensor([2.0, 2.0, 2.0, 2.0])

print(x + y)    # tensor([ 3.,  4.,  6., 10.])
print(x - y)    # tensor([-1.,  0.,  2.,  6.])
print(x * y)    # tensor([ 2.,  4.,  8., 16.])
print(x / y)    # tensor([0.5000, 1.0000, 2.0000, 4.0000])
print(x ** y)   # tensor([ 1.,  4., 16., 64.])
```
**注意** * 是按元素乘，不是矩阵乘法。
## 矩阵乘法
用 @ 或者 torch.matmul()：
```python
A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])
print(A @ B)   # 矩阵乘法
```

## 指数、对数、绝对值这些也有：
```python
print(torch.exp(x))
print(torch.log(x))
print(torch.abs(x))
print(torch.sum(x))          # 所有元素求和
print(torch.mean(x))         # 所有元素求平均
```

还有一个常用的：torch.max() 和 torch.min()，可以取整个张量的最大最小值，也可以沿某个轴取：
```python
x = torch.tensor([[1, 5, 3],
                  [4, 2, 6]])
print(x.max())              # tensor(6)  全局最大
print(x.max(dim=1))         # 沿第 1 轴取最大，返回 (values, indices)
max(dim=1) 返回的是一个 namedtuple，包含 values 和 indices 两个张量。如果你只想取值，写 values, indices = x.max(dim=1)。
```

对张量中的所有元素求和，会产生一个单元素张量。 x.sum() tensor(66.)


# 张量的连接
单独把连接（拼接）拎出来讲，因为它在实际写模型的时候太常见了。

所谓链接其实是把两个张量端对端地叠起来形成一个更大的值的张量。我们只需要提供张量列表，并给出沿哪个轴连接。

## torch.cat
沿已有的轴拼
cat 是 concatenate 的缩写。它把多个张量沿已经存在的某个轴接起来，不新建维度。
```python
a = torch.tensor([[1, 2],
                  [3, 4]])   # shape (2, 2)

b = torch.tensor([[5, 6],
                  [7, 8]])   # shape (2, 2)


# 沿第 0 轴拼（矩阵中就是竖着接）
c = torch.cat([a, b], dim=0)
print(c)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
print(c.shape)   # torch.Size([4, 2])

# 沿第 1 轴拼（矩阵中就是横着接）
d = torch.cat([a, b], dim=1)
print(d)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
print(d.shape)   # torch.Size([2, 4])
```
注意：除了你指定的那个轴，其他所有轴的长度必须完全一样。
```python
a = torch.randn(2, 3)
b = torch.randn(2, 5)

torch.cat([a, b], dim=1)   # OK，第 0 轴都是 2，第 1 轴 3+5
torch.cat([a, b], dim=0)   # 报错！第 1 轴 3 和 5 不相等
```

cat 可以一次拼多个，不限于两个：
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)

torch.cat([a, b, c], dim=0).shape   # torch.Size([6, 3])
```

## torch.stack
新建一个轴来叠
stack 跟 cat 不一样。它要求所有张量形状完全一致，然后新建一个维度，把它们叠起来。
```python
a = torch.tensor([1, 2, 3])   # shape (3,)
b = torch.tensor([4, 5, 6])   # shape (3,)

c = torch.stack([a, b], dim=0)
print(c)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(c.shape)   # torch.Size([2, 3])

d = torch.stack([a, b], dim=1)
print(d)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
print(d.shape)   # torch.Size([3, 2])
```

区别：
cat 是在已有的轴上接，维度数不变。
stack 是新建一个轴，维度数加一。
cat 是“接”，stack 是“叠”。


## 什么时候用哪个？
用 cat 的场景：特征拼接。
```python
# 两个不同来源的特征，拼在一起送进下一层
feat_a = torch.randn(32, 128)   # batch=32, 特征 128
feat_b = torch.randn(32, 64)    # batch=32, 特征 64

combined = torch.cat([feat_a, feat_b], dim=1)   # (32, 192)
```

用 stack 的场景：把多个样本组成一个 batch。

```python
# 有三张单张图片，形状都是 (C, H, W)
img1 = torch.randn(3, 224, 224)
img2 = torch.randn(3, 224, 224)
img3 = torch.randn(3, 224, 224)


# 叠成一个 batch
batch = torch.stack([img1, img2, img3], dim=0)   # (3, 3, 224, 224)
```

stack 出来的 (3, 3, 224, 224)，第一个 3 是 batch 大小，第二个 3 是通道数。这就是为什么 stack 常用来做 batch 组装。


## 逆操作：split 和 chunk
torch.split 按指定大小拆，torch.chunk 按份数拆。
```python
x = torch.arange(12).reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 沿第 0 轴拆成 3 份，每份大小 1
parts = torch.split(x, 1, dim=0)
print(len(parts))   # 3
print(parts[0])     # tensor([[0, 1, 2, 3]])


# 沿第 1 轴拆成 2 份，每份大小 2
parts = torch.split(x, 2, dim=1)
print(parts[0].shape)   # torch.Size([3, 2])
print(parts[1].shape)   # torch.Size([3, 2])
```

torch.chunk 是按份数拆：
```python
x = torch.arange(12).reshape(3, 4)


# 沿第 1 轴拆成 2 份
parts = torch.chunk(x, 2, dim=1)
print(parts[0].shape)   # torch.Size([3, 2])
print(parts[1].shape)   # torch.Size([3, 2])
```

split 和 chunk 的区别： 
split(x, size, dim)：你告诉它每份多大。
chunk(x, chunks, dim)：你告诉它拆成几份。

当然，拆的时候如果除不尽，最后一份会小一点。

注意：torch.cat 和 torch.stack 的第一个参数都是一个张量的列表或元组，不是把张量一个个传进去。

```python
# 示范
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)
 
# 错的
torch.cat(a, b, dim=0)      # 报错
torch.stack(a, b, dim=0)    # 报错
```

这个跟 torch.add(a, b) 那种可以分开传参的函数不一样。


## 李沐大神说
有时，我们想通过逻样运算符构建二元张量。以X==Y为例:对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为True，这意味着逻辑语句X==Y在该位置处为True，否则为False.
X==Y
tensor([[False,True, False，True]，
	    [False,False,False,False],
        [False,False,False,False]])

建议：
写模型的时候，遇到“把两个特征合到一起”就用 cat，遇到“把多个样本组成一批”就用 stack。用错了一般会直接报形状不匹配的错，调试的时候先看 shape，多半能发现问题。


# 索引和切片（去看 老汤圆 老大的blog）
## 张量的索引
格式：
张量对象[行，列]

跟 Python 列表基本一样，只是多了维度。
```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(x[0])        # 第一行: tensor([1, 2, 3])
print(x[0, 1])     # 第一行第二列: tensor(2)
print(x[:, 1])     # 所有行的第二列: tensor([2, 5, 8])
print(x[0:2, 1:3]) # 前两行的第二到第三列
print(x[[1,2],[0,2]]) #(1,0)和(2,2)两个位置的元素，即4和9
print(x[[[0],[1]]],[1,2]) #这里要注意[0]代表的是索引第一行一整行，也就是说这行代码会索引  tensor([2,3],
#          [5,6])
print(x[1::2,::2]) #打印所有奇数行偶数列

```

图例：
{% asset_img expamle.jpg "自己看" %}
个人理解方法：
{% asset_img expamle2.jpg "自己看" %}


易错点：
x[0:2, 1:3] 得到的是：
```python
tensor([[2, 3],
        [5, 6]])
```

左闭右开，跟 Python 切片一样。





## 布尔索引
也很常用：
```python
print(x[torch.tensor([True,False,True]),:])  #tensor([1,2,3],
                                             #         [7,8,9])
```

下面的例子x自定义了，以后的x依然同上
```python
x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
mask = x > 5
print(mask)          # tensor([False, False, ..., True, True, ...])
print(x[mask])       # tensor([6, 7, 8, 9])
print(x[x > 5])      # 效果一样
```

```python
print(x[:,x[2] > 5]) # tensor([[1, 2, 3],  没有删掉任何列，x[:, [True, True, True]]，mask控制列
                     #         [4, 5, 6],
                     #         [7, 8, 9]])

print(x[x[2] > 5]) #tensor([[1, 2, 3], mask 长度 = 3，它来自 x[2]（一行，长度=列数）。
                   #但这里被用来索引行，所以要求 行数也必须 = 3。 mask 用在行维度上，长度 3 == 行数 3，合法，保留所有行：（纯属巧合：本例 N == M == 3 才没报错；换个形状就会 IndexError，x[[True, True, True]]
                   #         [4, 5, 6],
                   #         [7, 8, 9]])
                    
                    
print(x[x[:,2] > 5]) #x[:，2] = [3,6,9] 打印第3列，大于5的行数据 tensor([4,5,6],   只保留第 3 列 > 5 的行（即第 2、3 行）：x[[False, True, True]]#                               [7,8,9])

print(x[:,x[1,:] > 5])
print(x[:，x[1] > 5]) #效果同上
 
print(x[1,x[1,:] > 5]) #索引大于5的元素
                     
```

  


mask写在唯一位置（没有，），则该语句控制**行**：

布尔 mask 索引的是"它出现位置对应的那个维度"
写在第一个位置 → mask 作用在行上，长度必须 = 行数
写在第二个位置（冒号后面）→ mask 作用在列上，长度必须 = 列数

正确：
第一步：算 x[:, 2]
text
x[:, 2] = [3, 6, 9]    ← 第 2 列，长度 = 行数 = 3

第二步：算 x[:, 2] > 5
text
[3,6,9] > 5  →  [False, True, True]
长度 = 3 = 行数。这个 mask 写在唯一位置，管的是行，长度刚好匹配。

第三步：套用索引
text
x[  [False, True, True]  ]
          ↑
       管"行"，逐行判断
行0: False → 丢弃

行1: True → 保留 [4,5,6]

行2: True → 保留 [7,8,9]

结果：

text
[[4, 5, 6],
 [7, 8, 9]]




这个在数据处理里用得非常多，比如你想把某个条件之外的元素全部选出来。

## 多维索引(降维)
```python
y = troch.randint(1,10,(2,3,4)) #2，3，4分别指0轴，1轴，2轴上的原素个数
print(f'y:{y}')

#获取0轴上的第一个数据
print(y[0,:,:])

#tensor([[3, 4, 6, 5],
#       [8, 8, 8, 3],
#       [4, 9, 6, 7]])


#获取（所有）1轴上的第一个元素
print(y[:,0,:])

#tensor([[3, 4, 6, 5],
#        [2, 8, 8, 5]])


#获取2轴上的第一个元素
print(y[:,:,0])

#tensor([[3, 8, 4],
#        [2, 6, 2]])

```

图像：
{% asset_img Snipaste_2026-10-03_11-47-16.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-03_12-47-19.png "Hexo 博客封面示例" %}
{% asset_ing Snipaste_2026-10-03_12-47-25.png "hexo 博客封面示例" %}



## 修改 ：
```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
x[0, 1] = 100
print(x)   # tensor([[  1, 100,   3], [  4,   5,   6]])
```
注意：索引和切片操作会直接改原张量，不会返回新的。如果你想保留原张量，先 clone()。



# 形状操作
API：
```python
reshape()
unsqueeze()
squeeze()
transpose()
permute()
view()
contiguous()
is_contiguous()

#主要掌握：
reshape()
unqueeze()
permute()
view()
```


可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状,我的理解是形状本身是一个存储着各个轴长度的数组，故而可以通过计算形状的所有元素的乘积来得到这个张量的总元素数量，即Size

print(x.shape)
输出为:torch.Size([12])

print(x.numel())
输出为：12（这里因为x是一个向量，故而有点巧合）

## 输出行和列
```python
torch.manual_seed(24) #设置种子
y = torch.randint(1,10,size=(2,3))
print(f'y:{y},shape:{y.shape},row:{y.shape{0}},columns:{y.shape[1]},{y.shape[-1]}')
```



## reshape 和 view
这两个都是用来改形状的。
reshape改变一个张量的形状，但是不改变它的大小。


```python
x = torch.arange(12)
print(x.shape)        # torch.Size([12])

y = x.reshape(3, 4)
print(y.shape)        # torch.Size([3, 4])

z = x.reshape(-1, 4)  # -1 表示自动算
print(z.shape)        # torch.Size([3, 4])
```

小技巧：
我们不需要通过手动指定每个维度来改变形状。，例如二维矩阵中，在知道宽度后，高度会被自动计算得，可以通过-1来调用Py的自动计算出形状，即我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。（动手学深度学习中摘录）

view() 跟 reshape() 功能一样，但有个限制：view() 只能用在内存连续（contiguous）的张量上，不连续的时候会报错。reshape() 不连续的时候会自动复制一份新的，所以更安全。平时用 reshape() 就行，不用纠结。

### 什么叫不该改变它的大小和内容
实例：
```python
import torch

a = torch.arange(12)
print("a =", a)
print("a.shape =", a.shape)

b = a.reshape(3, 4)
print("b =\n", b)
print("b.shape =", b.shape)

c = b.reshape(2, 6)
print("c =\n", c)
print("c.shape =", c.shape)

d = b.reshape(2, 2, 3)
print("d =\n", d)
print("d.shape =", d.shape)

print("a.flatten():", a.flatten())
print("b.flatten():", b.flatten())
print("c.flatten():", c.flatten())
print("d.flatten():", d.flatten())
```

输出：

```text
a = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
a.shape = torch.Size([12])

b =
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
b.shape = torch.Size([3, 4])

c =
 tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
c.shape = torch.Size([2, 6])

d =
 tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
d.shape = torch.Size([2, 2, 3])

a.flatten(): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
b.flatten(): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
c.flatten(): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
d.flatten(): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
可以看到，a、b、c、d 的形状分别是：
```

```text
(12,)
(3, 4)
(2, 6)
(2, 2, 3)
但 flatten() 之后元素顺序完全一样。一旦改变这个顺序就不是reshape
```

2. 二维 reshape 成二维，不是转置
```python
e = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

f = e.reshape(3, 2)

print("e =\n", e)
print("f =\n", f)

print("e.flatten():", e.flatten())
print("f.flatten():", f.flatten())
print("e.t().flatten():", e.t().flatten())
```
输出：
```text
e =
 tensor([[1, 2, 3],
        [4, 5, 6]])

f =
 tensor([[1, 2],
        [3, 4],
        [5, 6]])

e.flatten(): tensor([1, 2, 3, 4, 5, 6])
f.flatten(): tensor([1, 2, 3, 4, 5, 6])
e.t().flatten(): tensor([1, 4, 2, 5, 3, 6])
```
说明：
e.reshape(3, 2) 得到的是：

```text
[[1, 2],
 [3, 4],
 [5, 6]]
```
它不是转置,e.t() 才是转置：

```text
[[1, 4],
 [2, 5],
 [3, 6]]
```

所以 reshape 只改变形状，不改变按**行**优先读取的元素顺序。






{% asset_ing Snipaste_2026-10-03_20-26-42.png "hexo 博客封面示例" %}

## 转置和拼接
```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.T)            # 转置，形状从 (2,3) 变 (3,2)

a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[5, 6],
                  [7, 8]])

print(torch.cat([a, b], dim=0))   # 沿行拼，形状 (4, 2)
print(torch.cat([a, b], dim=1))   # 沿列拼，形状 (2, 4)
```

torch.cat 不改变维度数，只是在指定轴上把张量接起来。还有一个 torch.stack，它会新建一个维度，把张量叠起来，维度数加一。

## squeeze 和 unsqueeze

```python
x = torch.zeros(1, 3, 1, 5)
print(x.squeeze().shape)      # torch.Size([3, 5])，去掉所有长度为 1 的轴
print(x.squeeze(0).shape)     # torch.Size([3, 1, 5])，只去掉第 0 轴

y = torch.tensor([1, 2, 3])
print(y.unsqueeze(0).shape)   # torch.Size([1, 3])，在0维上，增加一个维度
print(y.unsqueeze(1).shape)   # torch.Size([3, 1])
```

squeeze 去掉长度为 1 的轴（降维），unsqueeze 在指定位置插入一个长度为 1 的轴（升维）。这两个在写模型的时候会反复用到，尤其是处理 batch 维度的时候。

### 1. 原始二维张量

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("x.shape =", x.shape)
print(x)
```

输出：

```text
x.shape = torch.Size([2, 3])
tensor([[1, 2, 3],
        [4, 5, 6]])
```

原始形状：

```text
(2, 3)
```

---

### 2. `unsqueeze`：增加一个大小为 1 的维度

#### 2.1 `unsqueeze(0)`：在最前面加一维

```python
y0 = x.unsqueeze(0)

print("y0.shape =", y0.shape)
print(y0)
```

输出：

```text
y0.shape = torch.Size([1, 2, 3])
tensor([[[1, 2, 3],
         [4, 5, 6]]])
```

形状变化：

```text
(2, 3) -> (1, 2, 3)
```

---

#### 2.2 `unsqueeze(1)`：在中间加一维

```python
y1 = x.unsqueeze(1)

print("y1.shape =", y1.shape)
print(y1)
```

输出：

```text
y1.shape = torch.Size([2, 1, 3])
tensor([[[1, 2, 3]],

        [[4, 5, 6]]])
```

形状变化：

```text
(2, 3) -> (2, 1, 3)
```

---

#### 2.3 `unsqueeze(2)`：在最后加一维

```python
y2 = x.unsqueeze(2)

print("y2.shape =", y2.shape)
print(y2)
```

输出：

```text
y2.shape = torch.Size([2, 3, 1])
tensor([[[1],
         [2],
         [3]],

        [[4],
         [5],
         [6]]])
```

形状变化：

```text
(2, 3) -> (2, 3, 1)
```

---

#### 2.4 负数维度写法

```python
print(x.unsqueeze(-1).shape)  # torch.Size([2, 3, 1])
print(x.unsqueeze(-2).shape)  # torch.Size([2, 1, 3])
print(x.unsqueeze(-3).shape)  # torch.Size([1, 2, 3])
```

对应关系：

```text
unsqueeze(-1) 等价于 unsqueeze(2)
unsqueeze(-2) 等价于 unsqueeze(1)
unsqueeze(-3) 等价于 unsqueeze(0)
```

---

**注意**：unsqueeze不能跨维度，也就是说形状为（2，3）的张量，使用最多可以unsqueeze[2],如果[3]，则会报错

### 3. `squeeze`：删除大小为 1 的维度

原始 `x` 的形状是 `(2, 3)`，没有大小为 1 的维度，所以 `squeeze` 不会改变它。

```python
print(x.squeeze().shape)   # torch.Size([2, 3])
print(x.squeeze(0).shape)  # torch.Size([2, 3])
print(x.squeeze(1).shape)  # torch.Size([2, 3])
print(x.squeeze(-1).shape) # torch.Size([2, 3])
```

也就是说：

```text
x.squeeze() 仍然是 (2, 3)
x.squeeze(0) 仍然是 (2, 3)
x.squeeze(1) 仍然是 (2, 3)
```

因为维度 0 大小是 2，维度 1 大小是 3，都不是 1。

---

### 4. 先 `unsqueeze` 再 `squeeze`，可以还原

#### 4.1 从 `(1, 2, 3)` 还原

```python
y0 = x.unsqueeze(0)   # (1, 2, 3)

print(y0.shape)               # torch.Size([1, 2, 3])
print(y0.squeeze(0).shape)    # torch.Size([2, 3])
print(y0.squeeze().shape)     # torch.Size([2, 3])
```

#### 4.2 从 `(2, 1, 3)` 还原

```python
y1 = x.unsqueeze(1)   # (2, 1, 3)

print(y1.shape)               # torch.Size([2, 1, 3])
print(y1.squeeze(1).shape)    # torch.Size([2, 3])
print(y1.squeeze().shape)     # torch.Size([2, 3])
```

#### 4.3 从 `(2, 3, 1)` 还原

```python
y2 = x.unsqueeze(2)   # (2, 3, 1)

print(y2.shape)               # torch.Size([2, 3, 1])
print(y2.squeeze(2).shape)    # torch.Size([2, 3])
print(y2.squeeze().shape)     # torch.Size([2, 3])
```

---

### 5. 多个大小为 1 的维度

```python
w = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 3)

print(w.shape)               # torch.Size([1, 1, 2, 3])
print(w.squeeze(0).shape)    # torch.Size([1, 2, 3])
print(w.squeeze().shape)     # torch.Size([2, 3])
```

说明：

- `squeeze(0)` 只删除第 0 维；
- `squeeze()` 不带参数，会删除所有大小为 1 的维度。

---

### 6. 总结表

| 表达式 | 形状 | 说明 |
|---|---|---|
| `x` | `(2, 3)` | 原始二维张量 |
| `x.unsqueeze(0)` | `(1, 2, 3)` | 最前面加一维 |
| `x.unsqueeze(1)` | `(2, 1, 3)` | 中间加一维 |
| `x.unsqueeze(2)` | `(2, 3, 1)` | 最后加一维 |
| `x.squeeze()` | `(2, 3)` | 没有大小为 1 的维度，不变 |
| `x.unsqueeze(0).squeeze(0)` | `(2, 3)` | 还原 |
| `x.unsqueeze(1).squeeze(1)` | `(2, 3)` | 还原 |
| `x.unsqueeze(2).squeeze(2)` | `(2, 3)` | 还原 |

核心结论：

```text
unsqueeze：增加一个大小为 1 的维度，不改变元素值和顺序。
squeeze：删除大小为 1 的维度，不改变元素值和顺序。
指定 squeeze(dim) 时，如果该维度大小不是 1，则张量保持不变。
```

## tranpose和permute
下面用 **PyTorch** 举例说明 `transpose` 和 `permute`，重点以二维张量为主，再补充三维时的区别。

### 1. 原始二维张量

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("x.shape =", x.shape)
print(x)
```

输出：

```text
x.shape = torch.Size([2, 3])
tensor([[1, 2, 3],
        [4, 5, 6]])
```

形状是：

```text
(2, 3)
```

---

### 2. `transpose`：交换两个维度

#### 2.1 `x.transpose(0, 1)`

```python
xt = x.transpose(0, 1)

print("xt.shape =", xt.shape)
print(xt)
```

输出：

```text
xt.shape = torch.Size([3, 2])
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

形状变化：

```text
(2, 3) -> (3, 2)
```

也可以写成函数形式：

```python
xt2 = torch.transpose(x, 0, 1)
print(xt2)
```

结果一样。
**注意**：不会改变原始张量的顺序（内容），而是返回新的张量
**技巧**：transpose(0,-1)依然可以使用，三维张量中效果与(0,2)相同
          链式编程其实也可以实现多维交换

---

### 3. `permute`：重排所有维度
补充知识：
{% asset_ing Snipaste_2026-10-03_21-34-57.png "hexo 博客封面示例" %}

1. 原始三维张量
```python
import torch

x = torch.arange(1, 13).reshape(2, 2, 3)

print("x.shape =", x.shape)
print(x)
输出：

text
x.shape = torch.Size([2, 2, 3])

tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]])

```
原始形状：
```text
(2, 2, 3)
```

```python
y4 = x.permute(2, 1, 0)

print("y4.shape =", y4.shape)
print(y4)
```

```text
y4.shape = torch.Size([3, 2, 2])

tensor([[[ 1,  7],
         [ 4, 10]],

        [[ 2,  8],
         [ 5, 11]],

        [[ 3,  9],
         [ 6, 12]]])
```

#### 3.1 `x.permute(1, 0)`
预览：
```python
xp = x.permute(1, 0)

print("xp.shape =", xp.shape)
print(xp)
```


输出：

```text
xp.shape = torch.Size([3, 2])
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

形状变化：

```text
(2, 3) -> (3, 2)
```

也可以写成：

```python
xp2 = torch.permute(x, (1, 0))
print(xp2)
```

结果一样。

---

### 4. 二维时，`transpose(0,1)` 和 `permute(1,0)` 等价

对于二维张量：

```python
x.transpose(0, 1)
x.permute(1, 0)
x.t()
x.T
```

这四个结果相同：

```python
print(x.transpose(0, 1))
print(x.permute(1, 0))
print(x.t())
print(x.T)
```

输出都是：

```text
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

所以：

```text
二维张量：
transpose(0, 1) == permute(1, 0) == t() == T
```

---

### 5. 它们改变的是维度顺序，不是元素值

原始张量按行优先展平：

```python
print("x.flatten():", x.flatten())
```

输出：

```text
x.flatten(): tensor([1, 2, 3, 4, 5, 6])
```

转置后按行优先展平：

```python
print("xt.flatten():", xt.flatten())
```

输出：

```text
xt.flatten(): tensor([1, 4, 2, 5, 3, 6])
```

也就是说：

```text
原张量逻辑顺序：1, 2, 3, 4, 5, 6
转置后逻辑顺序：1, 4, 2, 5, 3, 6
```

元素值没有变，但维度顺序变了，所以展平顺序也变了。

---

### 6. `transpose` / `permute` 后通常是非连续的

```python
xt = x.transpose(0, 1)

print("xt.is_contiguous():", xt.is_contiguous())
```

输出：

```text
xt.is_contiguous(): False
```

此时不能直接 `view`：

```python
# xt.view(6)  # 会报错
```

但 `reshape` 可以自动处理：

```python
print(xt.reshape(6))
```

输出：

```text
tensor([1, 4, 2, 5, 3, 6])
```

如果一定要 `view`，需要先 `contiguous()`：

```python
print(xt.contiguous().view(6))
```

输出：

```text
tensor([1, 4, 2, 5, 3, 6])
```

---

### 7. 三维张量时，`transpose` 和 `permute` 的区别

`transpose` 一次只能交换两个维度，`permute` 可以一次性重排所有维度。

```python
a = torch.arange(24).reshape(2, 3, 4)

print("a.shape =", a.shape)
```

输出：

```text
a.shape = torch.Size([2, 3, 4])
```

#### 7.1 `transpose(0, 1)`：交换第 0 维和第 1 维

```python
b = a.transpose(0, 1)
print("b.shape =", b.shape)
```

输出：

```text
b.shape = torch.Size([3, 2, 4])
```

#### 7.2 `permute(1, 0, 2)`：等价于交换第 0 维和第 1 维

```python
c = a.permute(1, 0, 2)
print("c.shape =", c.shape)
```

输出：

```text
c.shape = torch.Size([3, 2, 4])
```

#### 7.3 `permute(2, 0, 1)`：一次性变成全新维度顺序

```python
d = a.permute(2, 0, 1)
print("d.shape =", d.shape)
```

输出：

```text
d.shape = torch.Size([4, 2, 3])
```

这种重排，`transpose` 一次做不到，需要多次交换。

---

### 8. 总结表

| 表达式 | 形状 | 说明 |
|---|---|---|
| `x` | `(2, 3)` | 原始二维张量 |
| `x.transpose(0, 1)` | `(3, 2)` | 交换第 0 维和第 1 维 |
| `x.permute(1, 0)` | `(3, 2)` | 新维度顺序为原第 1 维、原第 0 维 |
| `x.t()` | `(3, 2)` | 二维转置 |
| `x.T` | `(3, 2)` | 二维转置 |
| `a.transpose(0, 1)` | `(3, 2, 4)` | 三维中只交换两个维度 |
| `a.permute(2, 0, 1)` | `(4, 2, 3)` | 三维中任意重排所有维度 |


```text
transpose：交换两个维度，一次只能交换两个。
permute：按指定顺序重排所有维度。

二维张量中：
x.transpose(0, 1) == x.permute(1, 0) == x.t() == x.T

它们都不改变元素值，只改变维度顺序。
转置或重排后通常变成非连续张量，view 可能失败，reshape 会自动处理。
```

## view和contiguous




# 广播

规则：从最后一个维度开始往前比，对应的维度要么相等，要么其中一个为 1，要么其中一个不存在。
可以解决上面提到的张量相加时其他轴长度不相等的问题

例子：
```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # shape (2, 3)

b = torch.tensor([10, 20, 30])  # shape (3,)

print(a + b)
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

b 的形状是 (3,)，a 是 (2, 3)。从最后一个维度比：3 == 3，OK。然后 b 没有第 0 维，相当于一个长度为 1 的维度，可以扩展到 2。所以 b 被“广播”成了：
```python
[[10, 20, 30],
 [10, 20, 30]]
```
 
再跟 a 相加。

广播不会真的复制数据，它只是在计算的时候假装那个维度存在。所以可以节省内存。

例子：
```python
x = torch.ones(5, 3, 4, 1)
y = torch.ones(   3, 1, 1)
# x 和 y 可以广播，因为：
# 从最后一维比：1 vs 1，OK
# 倒数第二维：4 vs 1，y 为 1 可以扩展
# 倒数第三维：3 vs 3，OK
# 倒数第四维：5 vs y 没有，OK
print((x + y).shape)   # torch.Size([5, 3, 4, 1])
```

反过来，如果某个维度上两个数都不为 1 且不相等，就会报错：
```python
x = torch.ones(5, 2, 4, 1)
y = torch.ones(   3, 1, 1)
# 倒数第三维：2 vs 3，不相等且都不为 1，报错
```

技巧：大多数情况下我们选择数组中长度为1的轴进行广播

# 内存：view 和 clone

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
y = x.reshape(3, 2)   # 或者 y = x.view(3, 2)
y[0, 0] = 100
print(x)
# tensor([[100,   2,   3],
#         [  4,   5,   6]])
```

你改了 y，x 也跟着变了。因为 reshape 和 view 返回的是视图，跟原张量共享同一块内存。

想真正复制一份独立的，用 clone()：

```python
y = x.clone()
y[0, 0] = 999
print(x)   # 不变
print(y)   # 变了
```

clone() 会复制数据，而且会被记录在计算图里，梯度回传时会传到源张量。如果你只是想把数据拿出来不参与梯度，用 .detach()。

李沐大神说：
执行某些操作可能导致内存重新分配
在用Python的id函数可以检验。
执行Y=Y+X后，我们会发现id(Y)指向另一个位置。这是因为Python首先计算Y+X，为结果分配新的内存，然后使Y指向内存中的这个新位置。
```python
before==id(Y) 
Y == Y+X
id(Y)== before 
```
```python
False
```

这可能是不可取的，原因有以下两个。
(1)分配内存可能导致不便。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新。
(2)如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

执行原地操作非常简单。
使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:]=<expression>。
为了说明这一点，我们先创建一个新的矩阵Z，其形状与Y相同，使用zeros_like来分配一个全0的块。
```python
Z=torch.zeros_like(Y)	
print('id(Z):',id(Z))
Z[:]=X+Y
print('id(Z):',id(Z)) 
```
输出为：
```python
id(Z):140470599776960 id(Z)::140470599776960
```

如果在后续计算中没有重复使用X，我们也可以使用X[:]=X+Y或X+=Y来减少操作的内存开销。
```python
before=id(X) X+=Y
id(x)== before
```


输出为：
```python
True
```

# GPU
PyTorch 最简单的 GPU 操作就是 .to(device)：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 3).to(device)
print(x.device)   # cuda:0 或 cpu
```

也可以造的时候直接指定：
```python
x = torch.randn(3, 3, device=device)
```

两个张量做运算，必须在同一个设备上。一个在 CPU 一个在 GPU，会报错。所以写训练代码的时候，通常会在开头定一个 device 变量，然后所有张量和模型都 .to(device)。

把 GPU 上的张量拿回 CPU 用 .cpu()：
```python
x_cpu = x.cpu()
```

但要注意，.cpu() 会复制数据，有开销。如果你只是想在 GPU 上打印一下值，用 x.item() 拿标量值，或者 x.cpu().numpy() 转成 NumPy。


# 自动微分
这是 PyTorch 最核心的功能，也是它区别于 NumPy 的根本原因。

只有标了 requires_grad=True 的张量，PyTorch 才会跟踪它的计算历史，才能算梯度。
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

z.backward()      # 反向传播，算梯度
print(x.grad)     # tensor([2., 4., 6.])
```

z = x² 的和，z 对 x 的导数就是 2x。x 是 [1, 2, 3]，所以梯度是 [2, 4, 6]。z.backward() 之后，梯度存在 x.grad 里。

requires_grad 默认是 False，只有你明确指定了才会跟踪。这是有意设计的，因为跟踪计算历史有开销，不是所有张量都需要梯度。

torch.no_grad() 临时关闭跟踪：
```python
with torch.no_grad():
    y = x * 2
    print(y.requires_grad)   # False
```

在推理或者更新参数的时候会用到，因为那些操作不需要算梯度。

梯度会累积，要手动清零：
```python
x = torch.tensor([1.0], requires_grad=True)

y = x * 2
y.backward()
print(x.grad)   # tensor([2.])

y = x * 3
y.backward()
print(x.grad)   # tensor([5.])，不是 3，是 2+3
```

第二次 backward() 的时候，梯度是累加的，不是覆盖的。所以训练循环里每一步都要写 optimizer.zero_grad() 或者 x.grad.zero_() 来清零。
detach() 把张量从计算图里摘出来：
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.detach()
print(z.requires_grad)   # False
```

detach() 返回一个跟原张量共享数据的张量，但不跟踪梯度。在你想用某个张量的值但不想让它参与梯度计算的时候用。


# 转化为python对象

取单个值：item()
张量里只有一个元素的时候，用 .item() 拿 Python 标量：
```python
x = torch.tensor([3.14])
print(x.item())        # 3.14
print(type(x.item()))  # <class 'float'>

y = torch.tensor(42)
print(y.item())        # 42
print(type(y.item()))  # <class 'int'>
```

最常见的用途：训练循环里打印 loss。
```python
loss = criterion(output, target)
print(loss.item())   # 打印一个 Python 浮点数，不是张量
```

**注意**直接 print(loss) 会打印 tensor(0.5234, grad_fn=<...>)，带一堆计算图信息。而且 loss 如果还挂在计算图上，一直持有它会导致显存不释放。

注意：.item() 只能用在只有一个元素的张量上。哪怕形状是 (1,) 或 (1, 1) 也行，元素数只要是 1 就可以。如果有多个元素，会报错：

```python
x = torch.tensor([1, 2, 3])
x.item()   # 报错：only one element tensors can be converted to Python scalars
```

多元素的张量想要 Python 数，得先选一个：

```python
x = torch.tensor([1, 2, 3])
print(x[0].item())   # 1
```

## 转成列表：tolist()
.tolist() 把张量转成嵌套的 Python 列表，形状保持不变：
```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.tolist())
# [[1, 2, 3], [4, 5, 6]]
print(type(x.tolist()))   # <class 'list'>
```

标量张量转出来是单个数字：
```python
x = torch.tensor(5)
print(x.tolist())   # 5
```

注意是 tolist()，不是 to_list()，没有下划线。
```python
.tolist() 会复制数据，而且如果在 GPU 上，会自动搬到 CPU，不用先 .cpu()：
x = torch.randn(3, 3, device='cuda')
print(x.tolist())   # 直接可以，自动回 CPU
```

## 和NumPy互转
这是最常用的互转。深度学习里经常需要用 NumPy 做数据处理，或者用 matplotlib 画图，都得先转成 NumPy 数组。

张量 → NumPy：
```python
x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
n = x.numpy()
print(type(n))       # <class 'numpy.ndarray'>
print(n)
# [[1. 2.]
#  [3. 4.]]
```


NumPy → 张量：

```python
import numpy as np

n = np.array([[1.0, 2.0],
              [3.0, 4.0]])
x = torch.from_numpy(n)
print(type(x))       # <class 'torch.Tensor'>
```

互转时的易错点：共享内存
.numpy() 和 torch.from_numpy() 默认共享内存，改一个另一个也变。

```python
x = torch.tensor([1.0, 2.0, 3.0])
n = x.numpy()
n[0] = 100
print(x)   # tensor([100.,   2.,   3.])  张量也变了
```

反过来也一样：
```python
n = np.array([1.0, 2.0, 3.0])
x = torch.from_numpy(n)
x[0] = 100
print(n)   # [100.   2.   3.]  NumPy 也变了
```

想避免这个，用 .clone() 或者 np.copy() 复制一份：
```python
n = x.numpy().copy()
# 或者
n = x.clone().numpy()
```

GPU 上的张量不能直接 .numpy()，必须先 .cpu()：
```python
x = torch.randn(3, 3, device='cuda')
x.numpy()          # 报错
x.cpu().numpy()    # OK
```


## 跟自动微分的关系
带梯度的张量（requires_grad=True）不能直接转 NumPy：
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
x.numpy()   # 报错：Can't call numpy() on Tensor that requires grad
```

先 .detach() 摘出来：
```python
x.detach().numpy()          # OK
x.detach().cpu().numpy()    # GPU 上的完整写法
```

.detach() 返回一个共享数据但不跟踪梯度的张量。如果你还要改数据，再加 .clone()：
```python
x.detach().cpu().clone().numpy()
```

# 反向转换：从 Python 对象造张量
从 Python 数、列表、元组造：
```python
torch.tensor(3.14)              # 标量
torch.tensor([1, 2, 3])         # 一维
torch.tensor([[1, 2], [3, 4]])  # 二维
torch.tensor((1, 2, 3))         # 元组也行
torch.tensor() 永远复制数据，跟原来的 Python 对象脱钩。
```

## 从 NumPy 造：
```python
torch.from_numpy(n)   # 共享内存
torch.tensor(n)       # 复制数据
torch.as_tensor(n)    # 能共享就共享，不能才复制
```

as_tensor 是个折中选项。如果输入已经是张量，它直接返回；如果是 NumPy 数组，它尽量共享内存。不确定用哪个的时候，torch.tensor() 最安全，因为它总是复制，不会有意外修改。

## 使用场景：
画图：plt.plot(x.numpy(), y.numpy())，matplotlib 只吃 NumPy 数组。
打印日志：loss.item() 拿一个干净的浮点数写进日志。
保存结果：pred.tolist() 转成列表再存 JSON。
和别的库对接：sklearn、pandas 这些库都吃 NumPy，不吃张量。
调试：看某个中间值到底是多少，.item() 或 .tolist() 都比直接 print 张量清楚。







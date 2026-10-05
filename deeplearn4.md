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

思维：索引一般是根据某个条件，将符合该条件的特征，的其他维度的特征一起打包索引，例如二维张量中以行设条件，则符合条件的行特征，及其列特征会被一起索引



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

## 转置



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


```text
view：改变形状，不复制数据，但通常要求张量在内存中连续。
contiguous：把张量变成连续内存布局；如果已经连续，直接返回自身；否则复制一份。
```

---

### 1. 什么是连续张量

```python
import torch

x = torch.arange(12).reshape(3, 4)

print("x =\n", x)
print("x.shape =", x.shape)
print("x.stride() =", x.stride())
print("x.is_contiguous() =", x.is_contiguous())
```

输出：

```text
x =
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
x.shape = torch.Size([3, 4])
x.stride() = (4, 1)
x.is_contiguous() = True
```

`stride = (4, 1)` 表示：

- 第 0 维走一步，内存跳过 4 个元素；
- 第 1 维走一步，内存跳过 1 个元素。

这就是标准的行优先连续布局。
{% asset_img Snipaste_2026-10-04_11-19-25.png "Hexo 博客封面示例" %}
---

### 2. `view`：共享内存，不复制

```python
y = x.view(2, 6)

print("y =\n", y)
print("y.shape =", y.shape)
print("x.data_ptr() == y.data_ptr():", x.data_ptr() == y.data_ptr())
```

输出：

```text
y =
 tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
y.shape = torch.Size([2, 6])
x.data_ptr() == y.data_ptr(): True
```

说明：

- `view` 不复制数据；
- `x` 和 `y` 共享同一块内存；
- 元素按行优先顺序不变。

---

### 3. 转置后通常非连续，不能直接 `view`

```python
z = x.t()

print("z =\n", z)
print("z.shape =", z.shape)
print("z.stride() =", z.stride())
print("z.is_contiguous() =", z.is_contiguous())
```

输出：

```text
z =
 tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
z.shape = torch.Size([4, 3])
z.stride() = (1, 4)
z.is_contiguous() = False
```

此时 `z` 是非连续张量。  
如果直接 `z.view(12)`，通常会报错：

```python
# z.view(12)  # RuntimeError: view size is not compatible with input tensor's size and stride
```

原因是：`z` 的逻辑顺序是：

```text
0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11
```

但它在内存中的实际排列并不是这个顺序连续存放的，所以不能直接 `view` 成一维。

同理，transpose和permute后的张量也不行

---

### 4. `contiguous()`：变成连续张量

```python
z_cont = z.contiguous()

print("z_cont =\n", z_cont)
print("z_cont.shape =", z_cont.shape)
print("z_cont.stride() =", z_cont.stride())
print("z_cont.is_contiguous() =", z_cont.is_contiguous())
```

输出：

```text
z_cont =
 tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
z_cont.shape = torch.Size([4, 3])
z_cont.stride() = (3, 1)
z_cont.is_contiguous() = True
```

现在 `z_cont` 是连续张量，可以正常 `view`：

```python
print("z_cont.view(12):", z_cont.view(12))
```

输出：

```text
z_cont.view(12): tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
```

注意：`z_cont` 和 `z` 不再共享内存，因为 `contiguous()` 复制了数据。

```python
print("z.data_ptr() == z_cont.data_ptr():", z.data_ptr() == z_cont.data_ptr())
```

输出：

```text
z.data_ptr() == z_cont.data_ptr(): False
```

---

### 5. 如果张量本来就连续，`contiguous()` 不复制

```python
a = torch.arange(6).reshape(2, 3)
b = a.contiguous()

print("a.is_contiguous():", a.is_contiguous())
print("a.data_ptr() == b.data_ptr():", a.data_ptr() == b.data_ptr())
```

输出：

```text
a.is_contiguous(): True
a.data_ptr() == b.data_ptr(): True
```

如果已经连续，`contiguous()` 直接返回自身，不复制。

如果非连续：

```python
c = a.t()
d = c.contiguous()

print("c.is_contiguous():", c.is_contiguous())
print("c.data_ptr() == d.data_ptr():", c.data_ptr() == d.data_ptr())
```

输出：

```text
c.is_contiguous(): False
c.data_ptr() == d.data_ptr(): False
```

此时 `contiguous()` 会复制一份新的连续张量。

---

### 6. `reshape` 和 `view` 的区别

```python
print("z.reshape(12):", z.reshape(12))
```

输出：

```text
z.reshape(12): tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
```

`reshape` 更智能：

- 如果张量连续，`reshape` 通常等价于 `view`，不复制；
- 如果张量非连续，`reshape` 会自动复制成连续张量再改变形状；
- `view` 不会自动复制，所以非连续时可能报错。

可以理解为：

```text
reshape ≈ view + 必要时 contiguous
```

---

### 7. 对比总结

| 操作 | 是否复制数据 | 对连续性的要求 | 结果 |
|---|---|---|---|
| `view` | 不复制 | 通常要求连续 | 共享内存，改变形状 |
| `contiguous()` | 已连续不复制，否则复制 | 无要求 | 返回连续张量 |
| `reshape` | 能 `view` 就不复制，否则复制 | 无要求 | 改变形状 |
| `transpose` / `permute` | 不复制 | 无要求 | 通常得到非连续张量 |

---

### 8. 核心结论

```text
view：
    只改变形状，不复制数据。
    要求张量连续，否则可能报错。

contiguous：
    把张量变成连续内存布局。
    如果已经连续，返回自身；
    如果不连续，复制一份新的连续张量。

典型流程：
    z = x.t()                  # 非连续
    z.view(...)                # 可能报错
    z.contiguous().view(...)   # 正确
    z.reshape(...)             # 也可以，自动处理
```
结论：

```text
view 是“看”成新形状，不复制；
contiguous 是“整理”内存布局，必要时复制；
reshape 是“看”得成就看，看不成先整理再看。
```

## 拼接
API:
```python
torch.cut() #不改变维度数，拼接张量，除了拼接指定的那个维度外其余维度必须一致,例：二维+二维=二维

torch.stack() #会在新的维度上链接一系列的张量，会增加一个新维度，并且所有输入张量形状必须完全相同,例：二维+二维=三维
```

---

### 1. 准备两个二维张量

```python
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

print("a.shape =", a.shape)
print("b.shape =", b.shape)
print("a =\n", a)
print("b =\n", b)
```

输出：

```text
a.shape = torch.Size([2, 3])
b.shape = torch.Size([2, 3])
a =
 tensor([[1, 2, 3],
        [4, 5, 6]])
b =
 tensor([[ 7,  8,  9],
        [10, 11, 12]])
```

两个张量形状都是 `(2, 3)`。

---

### 2. `torch.cat`：沿已有维度拼接，不增加新维度

#### 2.1 沿 `dim=0` 拼接

```python
c0 = torch.cat([a, b], dim=0)

print("c0.shape =", c0.shape)
print(c0)
```

输出：

```text
c0.shape = torch.Size([4, 3])
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
```

形状变化：

```text
(2, 3) + (2, 3)  --dim=0-->  (4, 3)
```

说明：在行方向拼接，行数相加，列数不变。

---

### 2.2 沿 `dim=1` 拼接

```python
c1 = torch.cat([a, b], dim=1)

print("c1.shape =", c1.shape)
print(c1)
```

输出：

```text
c1.shape = torch.Size([2, 6])
tensor([[ 1,  2,  3,  7,  8,  9],
        [ 4,  5,  6, 10, 11, 12]])
```

形状变化：

```text
(2, 3) + (2, 3)  --dim=1-->  (2, 6)
```

说明：在列方向拼接，列数相加，行数不变。

---

### 3. `torch.stack`：沿新维度堆叠，增加一个新维度

`stack` 要求所有张量形状完全相同。它会创建一个新维度，并把张量逐个放进去。

#### 3.1 沿 `dim=0` 堆叠

```python
s0 = torch.stack([a, b], dim=0)

print("s0.shape =", s0.shape)
print(s0)
```

输出：

```text
s0.shape = torch.Size([2, 2, 3])
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]])
```

形状变化：

```text
(2, 3) 和 (2, 3)  --stack dim=0-->  (2, 2, 3)
```

新维度大小 = 张量个数 = 2。

---

#### 3.2 沿 `dim=1` 堆叠

```python
s1 = torch.stack([a, b], dim=1)

print("s1.shape =", s1.shape)
print(s1)
```

输出：

```text
s1.shape = torch.Size([2, 2, 3])
tensor([[[ 1,  2,  3],
         [ 7,  8,  9]],

        [[ 4,  5,  6],
         [10, 11, 12]]])
```

形状变化：

```text
(2, 3) 和 (2, 3)  --stack dim=1-->  (2, 2, 3)
```

注意：形状虽然也是 `(2, 2, 3)`，但元素排列和 `dim=0` 不同。

---

### 3.3 沿 `dim=2` 堆叠

```python
s2 = torch.stack([a, b], dim=2)

print("s2.shape =", s2.shape)
print(s2)
```

输出：

```text
s2.shape = torch.Size([2, 3, 2])
tensor([[[ 1,  7],
         [ 2,  8],
         [ 3,  9]],

        [[ 4, 10],
         [ 5, 11],
         [ 6, 12]]])
```

形状变化：

```text
(2, 3) 和 (2, 3)  --stack dim=2-->  (2, 3, 2)
```

---

## 4. 对比总结

| 操作 | 作用 | 是否增加新维度 | 示例形状变化 |
|---|---|---|---|
| `torch.cat([a,b], dim=0)` | 沿第 0 维拼接 | 否 | `(2,3)+(2,3) -> (4,3)` |
| `torch.cat([a,b], dim=1)` | 沿第 1 维拼接 | 否 | `(2,3)+(2,3) -> (2,6)` |
| `torch.stack([a,b], dim=0)` | 沿新第 0 维堆叠 | 是 | `(2,3),(2,3) -> (2,2,3)` |
| `torch.stack([a,b], dim=1)` | 沿新第 1 维堆叠 | 是 | `(2,3),(2,3) -> (2,2,3)` |
| `torch.stack([a,b], dim=2)` | 沿新第 2 维堆叠 | 是 | `(2,3),(2,3) -> (2,3,2)` |

区别：

```text
cat：在已有维度上拼接，维度数不变。
stack：在新的维度上堆叠，维度数加 1。
```

要求：

```text
cat：除了拼接的那个维度，其他维度必须相同。
stack：所有张量的形状必须完全相同。
```

---

## 5. 完整代码汇总

```python
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

print("a.shape =", a.shape)
print("b.shape =", b.shape)

print("\n--- torch.cat ---")
c0 = torch.cat([a, b], dim=0)
c1 = torch.cat([a, b], dim=1)
print("c0.shape =", c0.shape)
print(c0)
print("c1.shape =", c1.shape)
print(c1)

print("\n--- torch.stack ---")
s0 = torch.stack([a, b], dim=0)
s1 = torch.stack([a, b], dim=1)
s2 = torch.stack([a, b], dim=2)
print("s0.shape =", s0.shape)
print(s0)
print("s1.shape =", s1.shape)
print(s1)
print("s2.shape =", s2.shape)
print(s2)
```

**注意**
拼接不可越界，stack时可以产生新的维度，但是依旧不能跨纬度

{% asset_img Snipaste_2026-10-04_13-57-21.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-04_14-00-49.png "Hexo 博客封面示例" %}

-1索引最后一个维度依然可以使用


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

{% asset_img Snipaste_2026-10-04_14-07-43.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-04_14-30-54.png "Hexo 博客封面示例" %}

**注意**：pytorch不支持向量张量对向量张量求导，只支持标量张量对向量张量求导
          大多数底层操作都是浮点型，要转型

```python
          loss.sum().backward() #自动执行反向传播
          forward() #前向传播

          y.backward() #y是一个标量
          x.grad #获取x点的梯度值，会累加上一次的梯度值
```

## 权重更新公式
W新 = W旧 - 学习率 * 梯度

权重 = 损失函数的导数


只有标了 **requires_grad=True** 的张量，PyTorch 才会跟踪它的计算历史，才能算梯度。
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

## 案例
```python
import torch

torch.manual_seed(42)

# =========================
# 1. 标量自动微分
# =========================
x = torch.tensor(3.0, requires_grad=True)

# y = x^2 + 2x + 1
y = x**2 + 2*x + 1

# 反向传播，自动求导
y.backward()

print("1. 标量自动微分")
print("x =", x.item())
print("y =", y.item())
print("dy/dx = x.grad =", x.grad.item())  # 2x + 2 = 8
print()


# =========================
# 2. 多变量偏导
# =========================
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# z = x^2 + y^3
z = x**2 + y**3

z.backward()

print("2. 多变量偏导")
print("x =", x.item(), "y =", y.item())
print("z =", z.item())
print("∂z/∂x =", x.grad.item())  # 2x = 4
print("∂z/∂y =", y.grad.item())  # 3y^2 = 27
print()


# =========================
# 3. 向量输出的自动微分
# =========================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# y = x^2，输出是向量
y = x**2

# 向量不能直接 backward()，需要传入与 y 同形状的梯度权重
# 这里等价于对 y.sum() 求导
y.backward(torch.ones_like(y))

print("3. 向量自动微分")
print("x =", x)
print("y =", y)
print("x.grad =", x.grad)  # 2x = [2, 4, 6]
print()


# =========================
# 4. 线性回归训练：自动微分 + 优化器
# =========================
# 数据：y = 2x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 优化器
optimizer = torch.optim.SGD([w, b], lr=0.03)

print("4. 线性回归训练")
for epoch in range(1000):
    # 前向传播
    y_pred = x @ w + b

    # 计算损失
    loss = ((y_pred - y) ** 2).mean()

    # 清空上一轮梯度
    optimizer.zero_grad()

    # 反向传播，自动计算梯度
    loss.backward()

    # 更新参数
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1:4d}, loss={loss.item():.6f}, w={w.item():.4f}, b={b.item():.4f}")

print("训练后：")
print("w =", w.item())
print("b =", b.item())
print("目标：w ≈ 2, b ≈ 1")
```

运行后你会看到类似输出：

```text
1. 标量自动微分
x = 3.0
y = 16.0
dy/dx = x.grad = 8.0

2. 多变量偏导
x = 2.0 y = 3.0
z = 31.0
∂z/∂x = 4.0
∂z/∂y = 27.0

3. 向量自动微分
x = tensor([1., 2., 3.], requires_grad=True)
y = tensor([1., 4., 9.], grad_fn=<PowBackward0>)
x.grad = tensor([2., 4., 6.])

4. 线性回归训练
epoch  200, loss=0.000000, w=2.0000, b=1.0000
...
训练后：
w = 2.0000...
b = 1.0000...
目标：w ≈ 2, b ≈ 1
```

关键点：
requires_grad=True：告诉 PyTorch 需要跟踪这个张量的计算历史。
backward()：从输出反向传播，自动计算所有需要梯度的张量的梯度。
.grad：保存计算得到的梯度。
optimizer.zero_grad()：清空上一轮梯度，否则梯度会累加。
loss.backward()：自动求导。
optimizer.step()：根据梯度更新参数。
向量输出不能直接 backward()，需要传入 gradient 参数，或先 sum() 变成标量再 backward()。
推理时可用 with torch.no_grad(): 关闭梯度跟踪，节省内存和计算。

## 案例2
{% asset_img Snipaste_2026-10-04_15-17-48.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-04_15-23-23.png "Hexo 博客封面示例" %}





## 梯度计算

```python
"""
梯度: 求导,求微分 上山下山最快的方向
梯度下降法: W1=W0-lr*梯度   lr是可调整已知参数  W0:初始模型的权重,已知  计算出W0的梯度后更新到W1权重
pytorch中如何自动计算梯度 自动微分模块
注意点: ①loss标量和w向量进行微分  ②梯度默认累加,计算当前的梯度, 梯度值是上次和当前次求和  ③梯度存储.grad属性中
"""
import torch


def dm01():
	# 创建标量张量 w权重
	# requires_grad: 是否自动微分,默认False
	# dtype: 自动微分的张量元素类型必须是浮点类型
	# w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	# 创建向量张量 w权重
	w = torch.tensor(data=[10, 20], requires_grad=True, dtype=torch.float32)
	# 定义损失函数, 计算损失值
	loss = 2 * w ** 2
	print('loss->', loss)
	print('loss.sum()->', loss.sum())
	# 计算梯度 反向传播  loss必须是标量张量,否则无法计算梯度
	loss.sum().backward()
	# 获取w权重的梯度值
	print('w.grad->', w.grad)
	w.data = w.data - 0.01 * w.grad
	print('w->', w)


if __name__ == '__main__':
	dm01()
```

## 梯度下降法求最优解

需求：
{% asset_img Snipaste_2026-10-04_20-12-03.png "Hexo 博客封面示例" %}

翻译：
{% asset_img Snipaste_2026-10-04_20-13-33.png "Hexo 博客封面示例" %}


```python
"""
① 创建自动微分w权重张量
② 自定义损失函数 loss=w**2+20  后续无需自定义,导入不同问题损失函数模块
③ 前向传播 -> 先根据上一版模型计算预测y值, 根据损失函数计算出损失值
④ 反向传播 -> 计算梯度
⑤ 梯度更新 -> 梯度下降法更新w权重
"""
import torch


def dm01():
	# ① 创建自动微分w权重张量
	w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	print('w->', w)
	# ② 自定义损失函数 后续无需自定义, 导入不同问题损失函数模块
	loss = w ** 2 + 20
	print('loss->', loss)
	# 0.01 -> 学习率
	print('开始 权重x初始值:%.6f (0.01 * w.grad):无 loss:%.6f' % (w, loss))

        #迭代(更新)1000次
	for i in range(1, 1001):
		# ③ 前向传播 -> 先根据上一版模型计算预测y值, 根据损失函数计算出损失值
		loss = w ** 2 + 20

		# 梯度清零 -> 梯度累加, 没有梯度默认None，而第一次运行w.grand.zero_()时，如果梯度为None，则会报错，且梯度会逐渐累加，故而要先进行判断
		if w.grad is not None:
			w.grad.zero_()

		# ④ 反向传播 -> 计算梯度
		loss.sum().backward()

		# ⑤ 梯度更新 -> 梯度下降法更新w权重，如果不更新，则梯度没变，权重W也就没变
		# W = W - lr * W.grad
		# w.data -> 更新w张量对象的数据, 不能直接使用w(将结果重新保存到一个新的变量中)
		w.data = w.data - 0.01 * w.grad
		print('w.grad->', w.grad)
		print('次数:%d 权重w: %.6f, (0.01 * w.grad):%.6f loss:%.6f' % (i, w, 0.01 * w.grad, loss))

        #打印最终结果
	print(f'最终结果 权重：{w},梯度：{w.grad:.5f},loss:{loss:.5f}')


if __name__ == '__main__':
	dm01()
```

**计算过程**
{% asset_img Snipaste_2026-10-04_21-25-35.png "Hexo 博客封面示例" %}

## 梯度计算注意点
- 不能将自动微分的张量转换成numpy数组，会发生报错，可以通过detach()方法实现

  ```python
  # 定义一个张量
  x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
  
  # 将x张量转换成numpy数组
  # 发生报错,RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
  # 不能将自动微分的张量转换成numpy数组
  # print(x1.numpy())
  
  # 通过detach()方法产生一个新的张量,作为叶子结点
  x2 = x1.detach()
  # x1和x2张量共享数据,但是x2不会自动微分
  print(x1.requires_grad)
  print(x2.requires_grad)
  # x1和x2张量的值一样,共用一份内存空间的数据
  print(x1.data)
  print(x2.data)
  print(id(x1.data))
  print(id(x2.data))
  
  # 将x2张量转换成numpy数组
  print(x2.numpy())
  ```
# detach()

## 案例
```python
import torch

# =========================
# 1. detach 基本作用：从计算图中分离
# =========================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x ** 2
print("y =", y)
print("y.requires_grad =", y.requires_grad)          # True
print("y.grad_fn =", y.grad_fn)                      # PowBackward0

y_detach = y.detach()
print("\ny_detach =", y_detach)
print("y_detach.requires_grad =", y_detach.requires_grad)  # False
print("y_detach.grad_fn =", y_detach.grad_fn)              # None

# y_detach 不能反向传播
try:
    y_detach.sum().backward()
except RuntimeError as e:
    print("\ny_detach.backward() 报错：")
    print(e)

# y 可以正常反向传播
y.sum().backward()
print("\nx.grad =", x.grad)  # 2x = [2, 4, 6]


# =========================
# 2. detach 共享内存：修改 detach 后的张量会影响原张量
# =========================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x_detach = x.detach()

print("\n修改前 x =", x)
x_detach[0] = 100.0
print("修改后 x =", x)          # x 也被修改，因为共享内存
print("x_detach =", x_detach)

# 如果不想共享内存，需要 clone
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x_indep = x.detach().clone()
x_indep[0] = 999.0
print("\n使用 detach().clone() 后：")
print("x =", x)                 # 不受影响
print("x_indep =", x_indep)


# =========================
# 3. 训练循环中记录 loss：避免保留计算图
# =========================
# 简单数据：y = 2x + 1
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.05)

losses = []

for epoch in range(200):
    y_pred = x_data @ w + b
    loss = ((y_pred - y_data) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 关键：loss.detach().item() 只取数值，不保留计算图
    losses.append(loss.detach().item())

print("\n训练完成")
print("w =", w.item(), "b =", b.item())
print("前 5 个 loss：", losses[:5])
print("后 5 个 loss：", losses[-5:])

# 如果直接 losses.append(loss)，loss 会一直持有计算图，可能造成内存泄漏
# 正确做法：loss.detach().item() 或 loss.item()


# =========================
# 4. detach 与 torch.no_grad() 的区别
# =========================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# detach：只针对某个张量，原张量仍然在计算图中
y = x * 2
y_det = y.detach()
print("\n--- detach ---")
print("y.requires_grad =", y.requires_grad)          # True
print("y_det.requires_grad =", y_det.requires_grad)  # False

# no_grad：上下文内所有计算都不构建计算图
with torch.no_grad():
    z = x * 2
print("--- no_grad ---")
print("z.requires_grad =", z.requires_grad)          # False
print("z.grad_fn =", z.grad_fn)                      # None


# =========================
# 5. 常见用途：将 tensor 转为 numpy
# =========================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# y.numpy() 会报错，因为 requires_grad=True
try:
    y.numpy()
except RuntimeError as e:
    print("\ny.numpy() 报错：")
    print(e)

# 正确做法：detach().numpy()
y_np = y.detach().numpy()
print("y.detach().numpy() =", y_np)
```

关键点总结：

| 操作 | 作用 |
|---|---|
| `y.detach()` | 返回一个新张量，从计算图中分离，`requires_grad=False`，但与原张量共享内存 |
| `y.detach().clone()` | 分离且复制数据，不共享内存 |
| `loss.detach().item()` | 训练中记录 loss，不保留计算图，避免内存泄漏 |
| `with torch.no_grad():` | 上下文内所有操作都不构建计算图 |
| `y.detach().numpy()` | 将需要梯度的张量转为 NumPy 数组 |

```text
detach()：把张量从计算图中“摘”出来，不再参与梯度计算，但数据仍与原张量共享。
```

注意：

- `detach()` 后不能反向传播；
- 修改 `detach()` 得到的张量，会影响原张量的数据；
- 如果只想取值，用 `detach().item()`；
- 如果想独立复制，用 `detach().clone()`。



## 共享内存
张量一旦设置了自动微分(requires_grad=True,)则无法转换成numpy的ndarray对象了，需要借助detach()

y = x.detach()后，x与y共用同一块内存
测试：
```python
x = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
print(f'x:{x},type:{type(x)}')
y = x.detach()
print(f'y:{y},type:{type(y)}')

#测试
x.data[0] = 100
print(f'x:{x},type:{type(x)}')
print(f'y:{y},type:{type(y)}')

#查看x和y谁可以自动微分
print(x.rquires_grad) #True
print(y.requires_grad) #False

```

## 转numpy
```python
# 自动微分的张量不能转换成numpy数组, 可以借助detach()方法生成新的不自动微分张量
import torch


def dm01():
	x1 = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	print('x1->', x1)
	# 判断张量是否自动微分 返回True/False
	print(x1.requires_grad)
	# 调用detach()方法对x1进行剥离, 得到新的张量,不能自动微分,数据和原张量共享
	x2 = x1.detach()
	print(x2.requires_grad)
	print(x1.data)
	print(x2.data)
	print(id(x1.data))
	print(id(x2.data))
	# 自动微分张量转换成numpy数组
	n1 = x2.numpy()
	print('n1->', n1)


if __name__ == '__main__':
	dm01()
```

**最终代码**
```python
y = x.detach().numpy()
```

## 自动微分模块应用（二维张量）

流程：
1.前向传播，计算出预测值
2.基于损失函数，预测值，真实值，计算梯度
3.利用权重更新公式W新 = W旧 - 学习率 * 梯度 来更新权重

{% asset_img Snipaste_2026-10-05_10-43-38.png "Hexo 博客封面示例" %}

```python
import torch

# 输入张量 2*5，表示：特征(输入数据)
x = torch.ones(2, 5)

# 目标值是 2*3  表示：标签（真实值）  
y = torch.zeros(2, 3)

# 设置要更新的权重和偏置的初始值（初始化）
# 根据公式x @ w(权重) + b(偏置)     (矩阵乘法),也就是前向传播求出预测值
# (5, 3,  是矩阵（x和w）乘法得出的，前提是要有5w（根据x定）
w = torch.randn(5, 3, requires_grad=True)

#3，意思是要3个b，根据公式做完矩阵乘法的结果tensor（5行3列）需要加上3个b（偏置）
b = torch.randn(3, requires_grad=True)

# 设置网络的输出值
z = torch.matmul(x, w) + b  # 矩阵乘法
#相当于z = x @ w(权重) + b(偏置)

# 设置损失函数，并进行损失的计算
loss = torch.nn.MSELoss() #nn ；是neural network:神经网络，包含许多损失函数

#loss = 损失
loss = loss(z, y) #实际只有一个值，backwork()无需sum()

# 自动微分
loss.backward()

# 打印 w,b 变量用来更新 的梯度
# backward 函数计算的梯度值会存储在张量的 grad 变量中
print("W的梯度:", w.grad)
print("b的梯度", b.grad)

#后续W新 = W旧 - 学习率 * 梯度 来更新权重
```

{% asset_img Snipaste_2026-10-05_10-55-36.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_11-09-51.png "Hexo 博客封面示例" %}

**更新流程**：
初始的w,1.6535被作为w旧，其梯度-0.6179与固定的学习率0.01参与公式，从而得出w新，如次w（矩阵）的第一个参数就更新完成了，之后就是重复
同理b也是如此： b新 = b旧 - 学习率 * 梯度 来更新权重

```python
import torch
import torch.nn as nn  # 损失函数,优化器函数,模型函数


def dm01():
	# todo:1-定义样本的x和y
	x = torch.ones(size=(2, 5))
	y = torch.zeros(size=(2, 3))
	print('x->', x)
	print('y->', y)
	# todo:2-初始模型权重 w b 自动微分张量
	w = torch.randn(size=(5, 3), requires_grad=True)
	b = torch.randn(size=(3,), requires_grad=True)
	print('w->', w)
	print('b->', b)
	# todo:3-初始模型,计算预测y值
	y_pred = torch.matmul(x, w) + b
	print('y_pred->', y_pred)
	# todo:4-根据MSE损失函数计算损失值
	# 创建MSE对象, 类创建对象
	criterion = nn.MSELoss()
	loss = criterion(y_pred, y)
	print('loss->', loss)
	# todo:5-反向传播,计算w和b梯度
	loss.sum().backward()
	print('w.grad->', w.grad)
	print('b.grad->', b.grad)


if __name__ == '__main__':
	dm01()
```

## 案例线性回归

我们使用 PyTorch 的各个组件来构建线性回归模型。在pytorch中进行模型构建的整个流程一般分为四个步骤：

- 准备训练集数据
- 构建要使用的模型
- 设置损失函数和优化器
- 模型训练

![1733908449747](assets/1733908449747.png)


**注意**
张量不能直接使用，要转换为DataLoader，具体转换流程如下：
{% asset_img Snipaste_2026-10-05_16-31-17.png "Hexo 博客封面示例" %}

要使用的API：

- 使用 PyTorch 的 nn.MSELoss() 代替平方损失函数
- 使用 PyTorch 的 data.DataLoader 代替数据加载器
- 使用 PyTorch 的 optim.SGD 代替优化器（优化器是帮助更新权重的）
- 使用 PyTorch 的 nn.Linear 代替假设函数

图例：
{% asset_img Snipaste_2026-10-05_16-50-04.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_16-51-03.png "Hexo 博客封面示例" %}

```python
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集，以后就不用这个包了，以后数据集都是已有的
import matplotlib.pyplot as plt #可视化


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 构造数据集
def create_dataset():
    x, y, coef = make_regression(n_samples=100,#100条样本（100个样本点）
                                 n_features=1, #1个特征（1个特征点）
                                 noise=10, #噪声，使图像更接近真实情况，噪声越大，样本点越散，反之，越集中
                                 coef=True, #是否返回系数，默认False，返回值为None
                                 bias=14.5, #偏置
                                 random_state=0) #随机种子，确保输出数据相同

    #print(type(x)),  x的类型为nadrray

    # 将构建数据转换为张量类型
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)

    return x, y, coef


# 训练模型
def train():
    # 构造数据集
    x, y, coef = create_dataset()

    # 构造数据集对象
    dataset = TensorDataset(x, y)

    # 构造数据加载器
    # dataset=:数据集对象
    # batch_size=:批量训练样本数据
    # shuffle=:样本数据是否进行乱序
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    # 构造模型
    # in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size
    # out_features指的是输出的二维张量的大小，即输出的[batch_size，size]中的size
    model = nn.Linear(in_features=1, out_features=1)

    # 构造平方损失函数
    criterion = nn.MSELoss()

    # 构造优化函数
    # params=model.parameters():训练的参数,w和b
    # lr=1e-2:学习率, 1e-2为10的负二次方
    print("w和b-->", list(model.parameters()))
    print("w-->", model.weight)
    print("b-->", model.bias)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-2)，#1e-2 科学计数法，等价于0.01

    # 初始化训练次数
    epochs = 100

    # 损失的变化
    epoch_loss = [] #记录每轮的损失，训练结束后通过这个（储存着100个损失）来绘制图像
    total_loss=0.0 #总损失
    train_sample=0.0 #训练的样本

    #具体的训练动作
    for _ in range(epochs): #_是填充物，不起作用，使用i也可以。

        for train_x, train_y in dataloader: #每批16个，共7批，每轮都读这7批，但是由于 shuffle=True ，每轮的数据都不一样

            # 将一个batch的训练数据送入模型
            y_pred = model(train_x.type(torch.float32)) #上面的代码已近转换过了，这里可以不用了

            # 计算损失值,均方误差,当前批次所有样本的平均误差 
            loss = criterion(y_pred, train_y.reshape(-1, 1).type(torch.float32))
            #y与x的格式不匹配要进行转换

            #计算总损失
            total_loss += loss.item() 
            # loss是平均误差,所以样本数+1

            #计算样本（批次）数
            train_sample += 1

            # 梯度清零
            optimizer.zero_grad()

            # 自动微分(反向传播)，专业可以加上sum，但这里无需
            loss.backward()

            # 更新参数
            optimizer.step()

        # 计算所有batch的平均误差作为当前epoch的误差,即把本轮的平均损失值，添加到列表中 
        epoch_loss.append(total_loss/train_sample)
        print(f'轮数：{epoch+1},平均损失：{total_loss/train_sample}')

    #打印最终的训练结果
    print(f'{epochs}轮的平均损失分别为{epoch_loss}')

    # 打印回归模型的w
    print(model.weight)

    # 打印回归模型的b
    print(model.bias)
    
    # 绘制损失变化曲线 
    plt.plot(range(epochs), epoch_loss) 
    plt.title('损失变化曲线') 
    plt.grid() 
    plt.show() #两个pit.show，意味着两张图

    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)

    #x是100个样本点特征，v是x中的每个值，计算预测值，经典公式y = kx + b
    y1 = torch.tensor([v * model.weight + model.bias for v in x])

    #计算真实值
    y2 = torch.tensor([v * coef + 14.5 for v in x])

    #绘制预测值和真实值的折线图
    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
```

补图：
{% asset_img Snipaste_2026-10-05_17-21-04.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-05_17-47-57.png "Hexo 博客封面示例" %}
{% asset_img 1733908907916.png "Hexo 博客封面示例" %}
{% asset_img 1733908912932.png "Hexo 博客封面示例" %}
{% asset_img 1733909608918.png "Hexo 博客封面示例" %}
{% asset_img 1733909623818.png "Hexo 博客封面示例" %}

![1733909608918](assets/1733909608918.png)

![1733909623818](assets/1733909623818.png)

### 案例解释2

```python
# todo: 2-模型训练
def train(x, y, coef):
	# 创建张量数据集对象
	datasets = TensorDataset(x, y)
	print('datasets->', datasets)
	# 创建数据加载器对象
	# dataset: 张量数据集对象
	# batch_size: 每个batch的样本数
	# shuffle: 是否打乱样本
	dataloader = DataLoader(dataset=datasets, batch_size=16, shuffle=True)
	print('dataloader->', dataloader)
	# for batch in dataloader:  # 每次遍历取每个batch样本
	# 	print('batch->', batch)  # [x张量对象, y张量对象]
	# 	break
	# 创建初始回归模型对象, 随机生成w和b, 元素类型为float32
	# in_features: 输入特征数 1个
	# out_features: 输出特征数 1个
	model = nn.Linear(in_features=1, out_features=1)
	print('model->', model)
	# 获取模型对象的w和b参数
	print('model.weight->', model.weight)
	print('model.bias->', model.bias)
	print('model.parameters()->', list(model.parameters()))
	# 创建损失函数对象, 计算损失值
	criterion = nn.MSELoss()
	# 创建SGD优化器对象, 更新w和b
	optimizer = SGD(params=model.parameters(), lr=0.01)
	# 定义变量, 接收训练次数, 损失值, 训练样本数
	epochs = 100
	loss_list = []  # 存储每次训练的平均损失值
	total_loss = 0.0
	train_samples = 0
	for epoch in range(epochs):  # 训练100次
        # 借助循环实现 mini-batch SGD 模型训练
		for train_x, train_y in dataloader:
			# 模型预测
			# train_x->float64
			# w->float32
			y_pred = model(train_x.type(dtype=torch.float32))  # y=w*x+b
			print('y_pred->', y_pred)
			# 计算损失值, 调用损失函数对象
			# print('train_y->', train_y)
			# y_pred: 二维张量
			# train_y: 一维张量, 修改成二维张量, n行1列
			# 可能发生报错, 修改形状
			# 修改train_y元素类型, 和y_pred类型一致, 否则发生报错
			loss = criterion(y_pred, train_y.reshape(shape=(-1, 1)).type(dtype=torch.float32))
			print('loss->', loss)
			# 获取loss标量张量的数值 item()
			# 统计n次batch的总MSE值
			total_loss += loss.item()
			# 统计batch次数
			train_samples += 1
			# 梯度清零
			optimizer.zero_grad()
			# 计算梯度值
			loss.backward()
			# 梯度更新 w和b更新
			# step()等同 w=w-lr*grad
			optimizer.step()
		# 每次训练的平均损失值保存到loss列表中
		loss_list.append(total_loss / train_samples)
		print('每次训练的平均损失值->', total_loss / train_samples)
	print('loss_list->', loss_list)
	print('w->', model.weight)
	print('b->', model.bias)
    
    # 绘制每次训练损失值曲线变化图
	plt.plot(range(epochs), loss_list)
	plt.title('损失值曲线变化图')
	plt.grid()
	plt.show()

	# 绘制预测值和真实值对比图
	# 绘制样本点分布
	plt.scatter(x, y)
	# 获取1000个样本点
	# x = torch.linspace(start=x.min(), end=x.max(), steps=1000)
	# 计算训练模型的预测值
	y1 = torch.tensor(data=[v * model.weight + model.bias for v in x])
	# 计算真实值
	y2 = torch.tensor(data=[v * coef + 14.5 for v in x])
	plt.plot(x, y1, label='训练')
	plt.plot(x, y2, label='真实')
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == '__main__':
	x, y, coef = create_datasets()
	train(x, y, coef)
```



## 固定套路
{% asset_img Snipaste_2026-10-05_17-27-59.png "Hexo 博客封面示例" %}

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







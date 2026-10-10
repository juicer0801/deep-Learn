---
title: 卷积神经网络
date: 2026-10-05 20:14:27
tags:
---

# 卷积神经网络CNN


回顾：

{% asset_img Snipaste_2026-10-09_17-10-50.png "Hexo 博客封面示例" %}


## 图像基础知识

###  图像基本概念

图像是人类视觉的基础，是自然景物的客观反映，是人类**认识世界和人类本身**的重要源泉。**“图”是物体反射或透射光的分布，“像“是人的视觉系统所接受的图在人脑中所形成的印象或认识**，[照片](https://baike.baidu.com/item/%E7%85%A7%E7%89%87/1465692?fromModule=lemma_inlink)、绘画、剪贴画、地图、书法作品、手写汉字、传真、卫星云图、影视画面、X光片、脑电图、[心电图](https://baike.baidu.com/item/%E5%BF%83%E7%94%B5%E5%9B%BE/399200?fromModule=lemma_inlink)等都是图像。

在计算机中，按照颜色和灰度的多少可以将图像分为四种基本类型。

- **二值图像**

  一幅二值图像的二维[矩阵](https://so.csdn.net/so/search?q=%E7%9F%A9%E9%98%B5&spm=1001.2101.3001.7020)仅由0、1两个值构成，**“0”代表黑色，“1”代白色**。由于每一像素（矩阵中每一元素）取值仅有0、1两种可能，所以计算机中二值图像的数据类型通常为1个二进制位。二值图像通常用于文字、线条图的扫描识别（OCR）和掩膜图像的存储。

{% asset_img 1737263836325.png "Hexo 博客封面示例" %}

- **灰度图像**

  **灰度图像矩阵元素的取值范围通常为[0，255]**。因此其数据类型一般为**8位无符号整数的（int8）**，这就是人们经常提到的256灰度图像。**“0”表示纯黑色，“255”表示纯白色，中间的数字从小到大表示由黑到白的过渡色。**二值图像可以看成是灰度图像的一个特例。

- **索引图像**

  索引图像的文件结构比较复杂，除了**存放图像的二维矩阵**外，还包括一个称之为**颜色索引矩阵MAP的二维数组**。MAP的大小由存放图像的矩阵元素值域决定，如矩阵元素值域为[0，255]，则MAP矩阵的大小为256Ⅹ3，**用MAP=[RGB]表示**。**MAP中每一行的三个元素分别指定该行对应颜色的红、绿、蓝单色值，MAP中每一行对应图像矩阵像素的一个灰度值**，如某一像素的灰度值为64，则该像素就与MAP中的第64行建立了映射关系，该像素在屏幕上的实际颜色由第64行的[RGB]组合决定。也就是说，图像在屏幕上显示时，每一像素的颜色由存放在矩阵中该像素的灰度值作为索引通过检索颜色索引矩阵MAP得到。

{% asset_img 1737263684063.png "Hexo 博客封面示例" %}

- **真彩色RGB图像**（重要）

  RGB图像与索引图像一样都可以用来表示彩色图像。与索引图像一样，它分别用红（R）、绿（G）、蓝（B）三原色的组合来表示每个像素的颜色。但与索引图像不同的是，**RGB图像每一个像素的颜色值（由RGB三原色表示）直接存放在图像矩阵中**，由于每一像素的颜色需由R、G、B三个分量来表示，**M、N分别表示图像的行列数，三个M x N的二维矩阵分别表示各个像素的R、G、B三个颜色分量。**RGB图像的数据类型一般为8位无符号整形。**注意：通道的顺序是 BGR 而不是 RGB。**

{% asset_img 562c11dfa9ec8a132677b97cfd03918fa0ecc039.jpg "Hexo 博客封面示例" %}


  | 图像类型     | 通道数           | 像素值范围       | 主要特点                                 | 常见用途                         |
  | ------------ | ---------------- | ---------------- | ---------------------------------------- | -------------------------------- |
  | **二值图像** | 1通道            | 0 或 1           | 每个像素只有黑与白两种值                 | 形态学操作、二值化、轮廓检测     |
  | **灰度图像** | 1通道            | 0 到 255         | 每个像素表示灰度（亮度）                 | 图像预处理、物体检测、人脸识别   |
  | **索引图像** | 1通道            | 0 到 255（索引） | 像素值为颜色表的索引，颜色表决定实际颜色 | 存储压缩、较少颜色的图像表示     |
  | **RGB图像**  | 3通道（R、G、B） | 0 到 255         | 每个像素由红、绿、蓝三个通道组成         | 普通彩色图像显示、图像处理与分析 |

  简单的讲：**图像是由像素点组成的，每个像素点的取值范围为: [0, 255] 。像素值越接近于0，颜色越暗，接近于黑色；像素值越接近于255，颜色越亮，接近于白色。**

  在深度学习中，我们使用的图像大多是彩色图，彩色图由RGB3个通道组成，如下图所示：

{% asset_img image-20220518101738866-2840262.png "Hexo 博客封面示例" %}

绘制图像课堂代码：
```python
"""
案例:
    演示基础的图像操作.

图像分类:
    二值图:        1通道, 每个像素点由0, 1组成
    灰度图:        1通道, 每个像素点的范围: [0, 255]
    索引图:        1通道, 每个像素点的范围: [0, 255], 像素点表示颜色表的索引
    RGB真彩图:     3通道, Red, Green, Blue, 红绿蓝.

涉及到的API:
    imshow()    基于HWC, 展示图像
    imread()    读取图像, 获取HWC
    imsave()    基于HWC, 保存图片.
"""

# 导包
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. 定义函数, 绘制: 全黑, 全白图.
def dm01():
    # 1. 定义全黑图片: 像素点越接近0越黑, 越接近255越白.
    # HWC:  H: 高度, W: 宽度, C: 通道.
    img1 = np.zeros((200, 200, 3))

    # print(f'img1: {img1}') #200行，3列的全零矩阵

    # 2. 绘制图片.
    plt.imshow(img1)

    # plt.axis('off') #关闭坐标系，黑图画出来了
    plt.show() #全黑图像


    # 2. 定义全白图片.
    img2 = torch.full(size=(200, 200, 3), fill_value=255)

    # print(f'img2: {img2}')
    plt.imshow(img2)

    # plt.axis('off') #关闭坐标系，白图画不出来了
    plt.show() #全白图像

# 2. 定义函数, 加载图片.
def dm02():
    # 1. 加载图片.
    img1 = plt.imread('./data/img.jpg') #图片转成数值
    # print(f'img1: {img1}')
    # print(f'img1.shape: {img1.shape}')  # (640, 640, 3), HWC

    # 2. 保存图像.
    plt.imsave('./data/img_copy.png', img1)

    # 3. 展示图像.
    plt.imshow(img1)
    plt.show()


# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()
```


###  图像加载

{% asset_img Snipaste_2026-10-09_17-15-55.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-09_17-18-17.png "Hexo 博客封面示例" %}


使用 matplotlib 库来实际理解下上面讲解的图像知识。

```python
import numpy as np
import matplotlib.pyplot as plt


# 像素值的理解
def test01():
    # 全0数组是黑色的图像
    # H, W, C -> 高, 宽, 通道
    img = np.zeros(shape=[200, 200, 3])
    # 展示图像
    plt.imshow(img)
    # 对坐标轴进行设置
    # off:关闭坐标轴
    plt.axis("off")
    plt.show()

    # 全255数组是白色的图像
    img = np.full(shape=[200, 200, 3], fill_value=255)
    # 展示图像
    plt.imshow(img)
    plt.show()


# 图像的加载
def test02():
    # 读取图像
    img = plt.imread("data/img.jpg")
    # 保存图像
    plt.imsave("data/img1.jpg", img)
    # 打印图像形状 高,宽,通道
    print("图像的形状(H, W, C):\n", img.shape)
    # 展示图像
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    test01()
    test02()
```

**输出结果:**

全黑和全白图像：

{% asset_img image-20220518104734551.png "Hexo 博客封面示例" %}

图像的形状为：

```python
图像的形状（H，W，C）:
 (640, 640, 3)
```

{% asset_img image-20220518104441645-2841883.png "Hexo 博客封面示例" %}

## 卷积神经网络（CNN）概述

###  什么是卷积神经网络

**卷积神经网络是深度学习在计算机视觉领域的突破性成果，专门用于处理图像、视频、语音等数据的神经网络** 

在计算机视觉领域, 往往我们输入的图像都很大，使用全连接网络的话，计算的代价较高。另外图像也很难保留原有的特征，导致图像处理的准确率不高。

卷积神经网络（Convolutional Neural Network）是**含有卷积层的神经网络**。卷积层的**作用就是用来自动学习、提取图像的特征**。

CNN网络主要由三部分构成：**卷积层、池化层和全连接层**构成：

（1）卷积层负责提取图像中的局部特征

（2）池化层用来大幅降低参数量级(降维)

（3）全连接层类似人工神经网络的部分，用来输出想要的结果

{% asset_img image-20220518095605390.png "Hexo 博客封面示例" %}

上图中CNN要做的事情是：给定一张图片，是车还是马未知，是什么车也未知，现在需要模型判断这张图片里具体是一个什么东西，总之输出一个结果：如果是车，那是什么车

- 最左边是
  - 数据输入层：对数据做一些处理，比如去均值（各维度都减对应维度的均值，使得输入数据各个维度都中心化为0，避免数据过多偏差，影响训练效果）、归一化（把所有的数据都归一到同样的范围）、PCA等等。CNN只对训练集做“去均值”这一步。
- 中间是
  - 卷积层(CONV)：线性乘积求和，提取图像中的局部特征
  - 激励层(RELU)：ReLU激活函数,输入数据转换成输出数据
  - 池化层(POOL)：取区域平均值或最大值，大幅降低参数量级(降维)
- 最右边是
  - 全连接层(FC)：接收二维数据集，输出CNN模型预测结果

### 卷积神经网络应用

#### 卷积计算

{% asset_img Snipaste_2026-10-09_17-39-37.png "Hexo 博客封面示例" %}
滤波器是算法自动生成的

{% asset_img 02.png "Hexo 博客封面示例" %}
{% asset_img 04.png "Hexo 博客封面示例" %}


#### 应用
**图像分类**：最常见的应用，例如识别图片中的物体类别

**目标检测**：检测图像中物体的位置和类别

**图像分割**：将图像分成多个区域，用于语义分割

**人脸识别**：识别图像中的人脸

**医学图像分析**：用于检测医学图像中的异常（如癌症检测、骨折检测等）

**自动驾驶**：用于识别交通标志、车辆、行人

###  **CNN中的经典算法/网络架构**

**LeNet-5:**作为最早的CNN架构之一，证明了CNN在图像识别任务上的有效性，为后续的CNN发展奠定了基础

- **卷积层:** 提取图像的边缘、角点等基本特征
- **池化层 (子采样层):** 降低特征图的维度，减少计算量，并提高模型对输入图像微小变化的鲁棒性
- **全连接层:** 将卷积层和池化层提取的特征进行组合，用于最终的分类

**AlexNet:**显著提升了ImageNet图像分类的准确率，证明了深度学习在计算机视觉领域的潜力，并推动了深度学习的快速发展

- **卷积层:** 使用更大的卷积核和更多的卷积核，提取更丰富的图像特征
- **ReLU激活函数:** 加速训练过程，并提高模型的性能
- **最大池化层:** 降低特征图的维度
- **Dropout层:** 防止过拟合
- **全连接层:** 用于最终的分类

**VGGNet:**探索了网络深度对性能的影响，证明了更深的网络可以提取更抽象和更具表达力的特征

- **卷积层:** 使用更小的卷积核 (3x3)，并堆叠多个卷积层，增加了网络的深度，提取更复杂的特征
- **最大池化层:** 降低特征图的维度
- **全连接层:** 用于最终的分类

**GoogLeNet (Inception):**提出了 Inception 模块，在提高性能的同时减少了计算量，为后续的网络架构设计提供了新的思路

- **Inception 模块:** 并行使用不同大小的卷积核和池化操作，然后将它们的输出连接起来，增加了网络的宽度，提高了网络的效率

**ResNet:**解决了深度网络训练困难的问题，使得可以训练更深的网络，从而显著提高了模型的性能

- **残差块 (Residual Block):** 引入跳跃连接 (Shortcut Connection)，允许梯度直接反向传播到浅层，解决了深度网络的梯度消失问题，使得训练非常深的网络成为可能

**DenseNet:**

- **密集块 (Dense Block):** 将每一层都与之前的所有层连接，特征重用更加充分，进一步提高了网络的性能和参数效率
- DenseNet通过密集连接（Dense Connectivity）在网络中各层之间建立了直接的连接，即每一层都接收前面所有层的输出作为输入。这种设计增强了特征传递和梯度流动，避免了梯度消失问题，并提高了信息的利用率

##  卷积层

> 卷积层（Convolutional Layer）通过卷积操作提取输入数据中的特征（例如图像中的边缘、纹理、形状等）。
>
> 卷积层利用卷积核（滤波器）对输入进行处理，从而生成特征图（feature map），并且每个卷积层能够提取不同层次的特征，从低级特征（如边缘）到高级特征（如物体的形状）。
>
> **卷积层的主要作用如下：**（重要）
>
> - **特征提取**：卷积层的主要作用是从输入图像中提取低级特征（如边缘、角点、纹理等）。通过多个卷积层的堆叠，网络能够逐渐从低级特征到高级特征（如物体的形状、区域等）进行学习。
>
> - **权重共享**：在卷积层中，同一个卷积核在整个输入图像上共享权重，这使得卷积层的参数数量大大减少，减少了计算量并提高了训练效率。
>
> - **局部连接**：卷积层中的每个神经元仅与输入图像的一个小局部区域（与卷积核大小相同的区域）相连，这称为**局部感受野**，这种局部连接方式更符合图像的空间结构，有助于捕捉图像中的局部特征。
>
> - **空间不变性**：由于卷积操作是局部的并且采用权重共享，卷积层在处理图像时具有**平移不变性**。也就是说，不论物体出现在图像的哪个位置，卷积层都能有效地检测到这些物体的特征。

### 卷积计算

{% asset_img 01.png "Hexo 博客封面示例" %}

1. input 表示输入的图像

2. filter 表示卷积核, 也叫做滤波器(滤波矩阵)

   - 一组固定的权重，因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter
   - 非严格意义上来讲，下图中红框框起来的部分便可以理解为一个滤波器，即带着**一组固定权重的神经元**。多个滤波器叠加便成了卷积层
   - 一个卷积核就是一个神经元

   缺图

3. input 经过 filter 得到输出为最右侧的图像，该图叫做特征图

那么, 它是如何进行计算的呢？**卷积运算本质上就是在滤波器和输入数据的局部区域间做点积。**

{% asset_img 02.png "Hexo 博客封面示例" %}

左上角的点计算方法：

{% asset_img 03.png "Hexo 博客封面示例" %}

按照上面的计算方法可以得到最终的特征图为:

{% asset_img 04.png "Hexo 博客封面示例" %}

图像上的卷积:

在下图对应的计算过程中，输入是一定区域大小(width*height)的数据，和滤波器filter（带着一组固定权重的神经元）做内积后得到新的二维数据。

{% asset_img 20160702214116669.png "Hexo 博客封面示例" %}

具体来说，左边是图像输入，中间部分就是滤波器filter（带着一组固定权重的神经元），不同的滤波器filter会得到不同的输出数据，比如颜色深浅、轮廓。相当于如果想提取图像的不同特征，则用不同的滤波器filter，提取想要的关于图像的特定信息：颜色深浅或轮廓。

### Padding（填充）

通过上面的卷积计算过程，最终的特征图比原始图像小很多，如果想要保持经过卷积后的图像大小不变, 可以在**原图周围**添加 Padding 来实现。

Padding（填充）操作是一种用于==在输入特征图的边界周围添加额外像素（通常是零）==。

**Padding的主要作用：**

- **保持空间维度：**如果不使用 padding，每次卷积操作后，特征图的尺寸都会缩小。多次卷积后，特征图会变得非常小，可能会丢失重要的边缘信息。Padding可以帮助维持输出特征图的尺寸与输入相同或接近相同。
- **保留边缘信息：**图像边缘的像素在卷积过程中参与的计算次数较少，这意味着边缘信息在特征提取过程中容易丢失。Padding通过在边缘添加额外的像素，增加了边缘像素的参与度，从而更好地保留了边缘信息。
- **提高性能：**Padding有助于避免由于特征图尺寸快速缩小而导致的信息丢失，从而提高模型的性能，尤其是在处理较小的图像或需要进行多层卷积时。

**Padding的类型：**

- **Valid Padding (No Padding):** 不进行任何填充。卷积核只在输入图像的有效区域内滑动。输出尺寸会缩小。 
- **Same Padding:** 添加足够的填充，使得输出特征图的尺寸与输入相同。 
- **Full Padding:** 尽可能多地添加填充，使得卷积核的每个元素都至少在输入图像上滑动一次。输出尺寸会增大。

**Padding的选择：**取决于具体的应用场景和网络架构

- **Valid Padding:** 适用于不需要保持输出尺寸的场景，或者输入图像足够大，边缘信息丢失不重要的情况。
- **Same Padding:** 广泛应用于各种CNN架构中，因为它可以保持特征图的尺寸，方便网络设计和计算。
- **Full Padding:** 较少使用，因为它会增加计算量，并且可能会在边缘引入一些伪影。

{% asset_img 05.png "Hexo 博客封面示例" %}

{% asset_img 1663574693963.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-09_17-55-37.png.png "Hexo 博客封面示例" %}

### Stride（步长）

Stride（步长）指的是**卷积核在图像上滑动时的步伐大小**，即每次卷积时卷积核在图像中向右（或向下）移动的像素数。步长直接影响卷积操作后输出特征图的尺寸，以及计算量和模型的特征提取能力。

**Stride的作用:**

- **降低计算复杂度：**更大的步长意味着卷积核移动的次数更少，从而减少了计算量，并加快了训练和推理速度。
- **减1长越大，生成的特征图尺寸越小。这类似于池化的降维效果。

- **增大感受野：**虽然更大的步长会减小特征图的尺寸，但它同时也会增大每个神经元在输入数据上的感受野。这意味着每个神经元能够捕捉到更大范围的输入信息。

**Stride的选择：**取决于具体的应用场景和网络架构

- **Stride = 1:** 这是最常见的设置，尤其是在网络的早期层。它允许保留更多的空间细节。
- **Stride > 1:** 通常用于减小特征图的尺寸和增大感受野，例如在网络的后期层或需要进行快速降维时。 常见的设置包括 stride=2 或 stride=4。

按照步长为1来移动卷积核，计算特征图如下所示：

{% asset_img 06.png "Hexo 博客封面示例" %}

如果把Stride增大为2，也是可以提取特征图的，如下图所示：
步长：只能在一个方向上走指定步长，如果剩下的格子不够，则跳过当行操作

{% asset_img 07.png "Hexo 博客封面示例" %}

###  多通道卷积计算

实际中的图像都是多个通道组成的，我们怎么计算卷积呢？

{% asset_img 08.png "Hexo 博客封面示例" %}
{% asset_img Snipaste_2026-10-09_18-14-57.png.png "Hexo 博客封面示例" %}

计算方法如下：

1. 当输入有多个通道(Channel), 例如 RGB 三个通道, 此时要求卷积核需要拥有相同的通道数（图像有多少通道，每个卷积核就有多少通道）.
2. 每个卷积核通道与对应的输入图像的各个通道进行卷积.
3. 将每个通道的卷积结果按位相加得到最终的特征图.

如下图所示:

{% asset_img 1734662691092.png "Hexo 博客封面示例" %}

### 多卷积核卷积计算

上面的例子里我们只使用一个卷积核进行特征提取, 实际对图像进行特征提取时, 我们需要使用多个卷积核进行特征提取. 这个多个卷积核可以理解为从不同到的视角、不同的角度对图像特征进行提取.

那么, 当使用多个卷积核时, 应该怎么进行特征提取呢?

{% asset_img 10.png "Hexo 博客封面示例" %}

通过以下例子查看多卷积核卷积计算流程:

{% asset_img 20160707204048899.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-09_18-18-26.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-09_18-27-18.png "Hexo 博客封面示例" %}

记得加偏置

可以看到：

- 两个神经元，意味着有两个滤波器
- 数据窗口每次移动两个步长取3*3的局部数据，即stride=2
- zero-padding=1。输入数据由`5*5*3`变为`7*7*3`
- 左边是输入（**7\*7\*3**中，7*7代表图像的像素/长宽，3代表R、G、B 三个颜色通道）
- 中间部分是两个不同的滤波器Filter w0、Filter w1
- 最右边则是两个不同的输出

### 特征图大小

输出特征图的大小与以下参数息息相关:

想象一下卷积核走动画面即可知
1. size: 卷积核/过滤器大小，一般会选择为奇数，比如有 `1*1`, `3*3`， `5*5`
2. Padding: 零填充的方式 
3. Stride: 步长

那计算方法如下图所示:

1. 输入图像大小: W x W
2. 卷积核大小: F x F
3. Stride: S
4. Padding: P
5. 输出图像大小: N x N

{% asset_img 11.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-09_18-32-11.png "Hexo 博客封面示例" %}

卷积核是可控的，一般设为5x5，7x7，但原图不可控

以下图为例:

1. 图像大小: 5 x 5
2. 卷积核大小: 3 x 3
3. Stride: 1
4. Padding: 1
5. (5 - 3 + 2) / 1 + 1 = 5（如果除不尽向下取整）, 即得到的特征图大小为: 5 x 5

{% asset_img 05.png "Hexo 博客封面示例" %}

###  PyTorch卷积层API

在PyTorch中进行卷积的API是：

```python
conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

"""
参数说明：
in_channels: 输入通道数，RGB图片一般是3
out_channels: 输出通道，也可以理解为卷积核kernel的数量
kernel_size：卷积核的高和宽设置，一般为3,5,7...
stride：卷积核移动的步长
	整数stride：表示在所有维度上使用相同的步长 stride=2 表示在水平和垂直方向上每次移动2个像素
	元组stride: 允许在不同维度上设置不同的步长 stride=(2, 1) 表示在水平方向上步长为2，在垂直方向上步长为1
padding：在四周加入padding的数量，默认补0
	padding=0：不进行填充。
	padding=1：在每个维度上填充 1 个像素（常用于保持输出尺寸与输入相同 padding=输入形状大小-输出形状大小）。
	padding='same'（从 PyTorch 1.9+ 开始支持）：让输出特征图的尺寸与输入保持一致。PyTorch会自动计算需要的填充量。stride必须等于1，不支持跨行，因为计算padding时可能出现小数
	padding=kernel_size-1：Full Padding 完全填充
"""
```

我们接下来对下面的图片进行特征提取:

{% asset_img 12.png "Hexo 博客封面示例" %}

下面演示多通道多卷积核卷积:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 对图像进行卷积
# 1 读取图像 显示图像
# 2 定义卷积层
# 3 变换数据形状 1-转成tensor 2-通道要求[C H W] 3-批次数要求 [batch, C, H, W]
# 4 给卷积层喂数据 [1, 3, 640, 640] ---> [1, 4, 319, 319])
# (H-F+2p)/s +1 = (640-3+0)/2 + 1 = 319.5向下取整319
def test01():

    # 1 读取图像
    img = plt.imread('./data/img.jpg')
    print('img.shape', img.shape)

    plt.imshow(img)
    plt.show()

    # 2 定义卷积层
    myconv2d = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
    print('myconv2d--->', myconv2d)

    # 3 变换数据形状 
	# ①转换成tensor
    # ②通道要求[C, H, W], 默认[H, W, C]
    # ③批次数要求[batch, C, H, W],多少个图像(一个图像是三维数组,四维有多少个三维就是多少个图像)
    # [0, 1, 2] --> [2, 0, 1]
    img2 = torch.tensor(img).permute(2, 0, 1)
    print('img2.shape--->', img2.shape)
	# 图像数为1,变为4维张量
    img3 = img2.unsqueeze(0)
    print('img3.shape--->', img3.shape)

    # 4 给卷积层喂数据 [1, 3, 640, 640] ---> [1, 4, 319, 319])
    # (H-F+2p)/s + 1 = (640-3+0)/2 + 1 = 319.5向下取整319
    img4 = myconv2d(img3.type(torch.float32))
    print('img4-->', img4.shape)


if __name__ == '__main__':
    test01()
```

**输出结果:**

```python
img.shape---> (640, 640, 3)
myconv2d---> Conv2d(3, 4, kernel_size=(3, 3), stride=(2, 2))
img2.shape---> torch.Size([3, 640, 640])
img3.shape---> torch.Size([1, 3, 640, 640])
img4--> torch.Size([1, 4, 319, 319])
```

对生成的特征图进行显示:

```python
# 对图像卷积 并显示卷积以后的特征图
# 思路：去掉批次数 -> 转成[HWC] -->按照通道拿数据[:,:,0123]
def test02():

    # 1 读取图像
    img = plt.imread('./data/img.jpg')
    print('img.shape', img.shape)

    plt.imshow(img)
    plt.show()

    # 2 定义卷积层
    myconv2d = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
    print('myconv2d--->', myconv2d)

    # 3 变换数据形状 1-转成tensor 2-通道要求[C H W] 3-批次数要求 [batch, C, H, W]
    # [0, 1, 2] --> [2, 0, 1]
    img2 = torch.tensor(img).permute(2, 0, 1)
    print('img2.shape--->', img2.shape)

    img3 = img2.unsqueeze(0)
    print('img3.shape--->', img3.shape)

    # 4 给卷积层喂数据 [1, 3, 640, 640] ---> [1, 4, 319, 319])
    # (H-F+2p)/s +1 = (640-3+0)/2 + 1 = 319.5向下取整319
    img4 = myconv2d(img3.type(torch.float32))
    print('img4-->', img4.shape)

    # 5 查看特征图思路：去掉批次数 -> 转成[HWC] -->按照通道拿数据[:,:,0123]
    img5 = img4[0] # 去掉批次数
    print('去掉批次数img5.shape-->', img5.shape)
    img6 = img5.permute(1, 2, 0)
    print('去掉批次数以后，再进行HWC img6.shape-->', img6.shape)

    feature1 = img6[:, :, 0].detach().numpy()
    feature2 = img6[:, :, 1].detach().numpy()
    feature3 = img6[:, :, 2].detach().numpy()
    feature4 = img6[:, :, 3].detach().numpy()

    plt.imshow(feature1)
    plt.show()

    plt.imshow(feature2)
    plt.show()

    plt.imshow(feature3)
    plt.show()

    plt.imshow(feature4)
    plt.show()

    
if __name__ == '__main__':
    test02()
```

课堂代码：
```python
"""
案例:
    演示卷积层的API, 用于 提取图像的局部特征, 获取: 特征图(Feature Map)

卷积神经网络介绍:
    概述:
        全称叫: Convolutional neural network, 即: 包含卷积层的神经网络.
    组成:
        卷积层(Convolutional):
            用于提取图像的 局部特征, 结合 卷积核(每个卷积核 = 1个神经元) 实现, 处理后的结果叫: 特征图.
        池化层(Pooling):
            用于 降维, 降采样
        全连接层(Full Connected, fc, linear):
            用于 预测结果, 并输出结果的.
    特征图计算方式:
        N = (W - F + 2*P) / S   +  1
        W: 输入图像的大小
        F: 卷积核的大小
        P: 填充的大小
        S: 步长
        N: 输出图像的大小(特征图大小)
"""

# 导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 定义函数, 用于完成图像的加载, 卷积, 特征图可视化操作.
def dm01():
    # 1. 加载RGB真彩图.
    img = plt.imread('./data/a.jpg')

    # 2. 打印读取到的图像信息.
    # print(f'img: {img}, shape: {img.shape}')     # HWC (640, 640, 3)

    # 3. 把图像的形状从 HWC -> CHW, 思路: img -> 张量 -> 转换维度.
    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute(2, 0, 1)

    # print(f'img2: {img2}, shape: {img2.shape}')    # [3, 640, 640]

    # 4. 因为这里只有1张图, 所以我们给它在增加1个维度, 从 CHW -> (1, C, H, W), 1张3通道的 640*640像素的 图
    img3 = img2.unsqueeze(dim=0)

    # print(f'img3: {img3}, shape: {img3.shape}')      # [1, 3, 640, 640]

    # 5. 创建卷积层对象, 提取 特征图.
    # 参1: 输入图像的通道数, 参2: 输出图像的通道数(几个特征图),4个卷积核, 参3: 卷积核的大小, 参4: 步长stride, 参5: 填充.
    conv = nn.Conv2d(3, 4, 3, 2, 0)

    # 6. 具体的卷积计算.
    conv_img = conv(img3)

    # 7. 打印卷积后的结果.   1张4通道的 319*319像素的 图，套那个卷积计算公式
    # print(f'conv_img: {conv_img}, shape: {conv_img.shape}') # (1, 4, 319, 319)

    # 8. 查看提取到的 4个 特征图.
    img4 = conv_img[0] #(1,4,319,319)里面的第一个元素
    # print(f'img4: {img4}, shape: {img4.shape}')     # (4, 319, 319) -> CHW

    # 9. 把上述的图从 CHW -> HWC，以便于可视化操作
    img5 = img4.permute(1, 2, 0)
    # print(f'img5: {img5}, shape: {img5.shape}')       # (319, 319, 4) -> HWC

    # 10. 可视化第1个通道的特征图.
    feature1 = img5[:, :, 3].detach().numpy()           # 第0通道(即: 第1通道的)  (319, 319)像素图
    plt.imshow(feature1)
    plt.show()


# 2. 测试
if __name__ == '__main__':
    dm01()
```    

难点解析
“通道”就是数据的“层”或者“维度”，用来描述同一个空间位置上不同种类的信息
彩色图（RGB）：有3个通道。每个像素点由红（R）、绿（G）、蓝（B）三个数值**叠加**而成。
通道数 = 卷积核的个数 = 提取出的特征种类的数量。

如果 img5 的形状是 (H, W, 128)，你把它直接丢给 plt.imshow()，Matplotlib 会直接报错或者不知如何渲染。因为 imshow 通常只接受 (H, W)（灰度图）或者 (H, W, 3) / (H, W, 4)（RGB / RGBA 图像）。
我们无法在屏幕上同时看 128 维的彩色数据，只能将其拆解为多个二维灰度图（热力图），一次看一个。

注解：三通道24bit，四通道32bit

img5[:, :, 3] 的意思是：取出所有高度、所有宽度，但只取索引为 3 的通道。
得到的结果形状是 (H, W)，这是一个纯粹的二维矩阵

原图 (RGB, 3通道)：就像一匹彩色的布料。它由红、绿、蓝三种颜色的线交织而成。你可以直接看这匹布，因为它有颜色。
卷积核 (4个)：就像工厂里的 4 个质检员。
质检员 1 专门找“布料上的划痕”。
质检员 2 专门找“布料上的褶皱”。
质检员 3 专门找“布料上的破洞”。
质检员 4 专门找“布料上的污渍”。
卷积后的特征图 (4通道)：就是这 4 个质检员各自画出来的黑白草图（二维矩阵）。
质检员 1 画的草图上，有划痕的地方是白色的，没划痕的地方是黑色的。
质检员 2 画的草图上，有褶皱的地方是白色的，没褶皱的地方是黑色的。
这时候你能把这 4 张黑白草图叠在一起，说这是一张红绿蓝的照片吗？不能。
叠在一起只会是一团糟。你只能把质检员 1 的草图单独拿出来看，感叹一句：“哦，原来这个位置有划痕啊！”

既然显示器可以一次看三个通道，那么为什么还要，一个通道一个通道索引查看呢

从技术实现上来说：你完全可以写 img5[:, :, :3]，把前三个特征通道抽出来，强行当成 RGB 图片让 plt.imshow() 显示。
1. 语义完全不同：“颜色” 不同于“特征”
原图的 3 通道：是红、绿、蓝。它们叠加在一起，物理上真的能混出真实的彩色画面（比如一只橘猫）。

卷积后的 4 通道：是 4 个卷积核提取出的“特征强度图”（比如：通道0找横线，通道1找竖线，通道2找圆点）。
如果你把“横线特征”当成红色，“竖线特征”当成绿色，“圆点特征”当成蓝色去混合，你看到的将是一张花里胡哨、毫无逻辑的色块图。你不能指着一块紫色说“这里是紫色的猫”，因为它仅仅代表“这里既检测到了横线，又检测到了圆点”。

2. 数值范围不匹配
原图的 RGB 数值通常在 0 ~ 255 或者 0.0 ~ 1.0。
但卷积层的输出（特征图）没有固定的数值范围。它们可能是负数，也可能很大（比如 -15.3 到 48.7）。
如果直接把这些数值当成 RGB 丢给 plt.imshow()，画图库会不知所措：

小于 0 的值会被强行截断成黑色（0）。

大于 1 的值会被强行截断成白色或最大色值。
结果就是：你看到的大概率是一张全黑或者全白、只剩几根色带的废图，完全丢失了特征分布的细节。

3. 通道数量不匹配
如果卷积核是 64 个，输出就是 64 通道。
显示器只看 3 个通道，那你取哪 3 个？
如果只取前 3 个，意味着你丢弃了后面 61 个卷积核学到的特征。这种可视化是不完整的。

4. 可视化目的不同
看原图：目的是看“画面里有什么”（猫、狗、车）。

看特征图：目的是看“网络学到了什么”（比如某块区域对水平边缘极其敏感）。
为了看“某一种特征的分布”，最好的方式是把单一通道当成二维热力图（灰度图）来看。

亮的地方 = 这个特征在此处非常强烈。

暗的地方 = 这个特征在此处不存在。
如果你强行把三个特征混成彩色，人眼反而无法分辨某个亮点到底是哪个特征带来的。

我就要一次看三个通道

单看灰度图太无聊，真的想看多通道的组合，业界通常有两种正确做法：

做法一：热力图（最常用）
把单一通道的值，映射成彩虹色（而不是灰度色）。

```python
# 使用 cmap='viridis' 或 'jet'，数值高低会呈现蓝->红的变化
plt.imshow(img5[:, :, 0], cmap='viridis')
plt.colorbar()  # 显示颜色条，告诉你什么颜色代表什么数值
plt.show()

做法二：并排显示（Grid 布局）
如果你想看前 3 个通道，你可以画一个 1 行 3 列的画布，并排显示三张独立的图：

python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axes[i].imshow(img5[:, :, i], cmap='gray')
    axes[i].set_title(f'Channel {i}')
    axes[i].axis('off')
plt.show()
```

HWC (640, 640, 3) 是一个数据形状（Shape）。它描述的是一个三维数组的“整体结构”。
img5[:, :, 3] 是一个切片操作（Slicing）。它是从某个三维数组中“抽取一部分”的代码动作。

. 梳理 HWC (640, 640, 3)
这是一张原图的形状描述：

H (Height) = 640：图片高 640 像素。
W (Width) = 640：图片宽 640 像素。
C (Channel) = 3：图片有 3 个通道（即 RGB 红绿蓝）。
整体理解：这是一张三维数据 (640, 640, 3)。
可视化：可以直接丢给 plt.imshow()，因为它恰好对应显示器的红绿蓝三个通道，能立刻显示出一张彩色照片。

. 梳理 img5[:, :, 3]
假设你手里有一张经过卷积后的特征图 img5，假设它形状是 (319, 319, 4)（4个特征通道）。
这行代码的意思是：

: (第一个) = 取所有行（高度 319）。
: (第二个) = 取所有列（宽度 319）。
3 (第三个) = 只取第 4 个通道（索引从 0 开始，0、1、2、3）。
操作结果：它把一个 (319, 319, 4) 的三维数组，切成了一个 (319, 319) 的二维矩阵。
可视化：这个二维矩阵丢给 plt.imshow()，显示的是一张单通道的灰度热力图。


RGB是有三个通道，img5是取第四个
| | `HWC (640, 640, 3)` 里的 3 | `img5[:, :, 3]` 里的 3 |
| :--- | :--- | :--- |
| **含义** | **数量**：代表有 3 个通道 | **索引**：代表取第 4 个通道（索引从0开始） |
| **维度意义** | 它是形状的一部分，表示这是一个三维数据。 | 它是切片操作，表示要取第三个维度上的第 4 个元素。 |
| **结果形状** | 三维 `(640, 640, 3)` | 二维 `(319, 319)` |
| **语义** | 物理颜色（红、绿、蓝） | 抽象特征（卷积核提取的第 4 种特征） |
| **能否直接显示** | 能，直接显示彩色图 | 不能直接显示彩色图，只能显示单通道的灰度热力图 |

**生成特征图显示:**

{% asset_img image-20220706160149097.png "Hexo 博客封面示例" %}

## 池化层

> 池化层（Pooling Layer）是用于降低输入数据的空间维度（例如图像的高度和宽度），从而减少计算量、减少内存消耗，并提高模型的鲁棒性。
>
> 池化层通常位于卷积层之后，它通过对卷积层输出的特征图进行下采样，保留最重要的特征信息，同时丢弃一些不重要的细节。
>
> **池化层的主要作用如下:**
>
> - **降维和计算量减少**：池化层通过减少特征图的尺寸，从而降低了计算量，特别是在多层网络中，随着层数的增加，池化能够显著减少计算资源的消耗。
>
> - **提高鲁棒性**：池化操作可以使得特征对小的变换、平移和旋转变得更加不敏感。这样，模型在面对噪声或图像的轻微变化时，依然能够稳定工作。
>
> - **防止过拟合**：通过池化减少了特征图的大小，减少了模型的复杂度，从而有助于防止过拟合，尤其是在较小的数据集上。
>
> - **抽象特征**：通过池化层的操作，可以提取更为抽象和高层次的特征，使得网络能够学习到更具泛化能力的表示。

### 池化层计算

- 最大池化(Max Pooling) ：通过池化窗口进行最大池化，**取窗口中的最大值作为输出**

{% asset_img 1663589028409.png "Hexo 博客封面示例" %},用得多

- 平均池化(Avg Pooling) ：**取窗口内的所有值的均值作为输出**

{% asset_img 1663589040380.png "Hexo 博客封面示例" %}

### Padding（填充）

{% asset_img 1734674337777..png "Hexo 博客封面示例" %}

### Stride（步长）

{% asset_img 1734674385856.png "Hexo 博客封面示例" %}



### 多通道池化计算

在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各个通道的输入相加。这意味着==池化层的输出和输入的通道数是相等。==

**池化只在宽高维度上池化**，**在通道上是不发生池化**（池化前后，多少个通道还是多少个通道）

{% asset_img 1734674742627.png "Hexo 博客封面示例" %}

### PyTorch池化层API

在PyTorch中进行池化的API是：

```python
# 最大池化
nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
# 平均池化
nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
"""
参数说明：
kernel_size：核的高和宽设置，一般为3,5,7...
stride：核移动的步长
padding：在四周加入padding的数量，默认补0
"""
```

- 单通道池化

  ```python
  import torch
  import torch.nn as nn
  
  
  # 1. 单通道池化
  # 定义输入数据 [1,3,3]
  inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=torch.float)
  # 修改stride，padding观察效果
  # 1. 最大池化
  pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
  output = pooling(inputs)
  print("最大池化：\n", output)
  # 2. 平均池化
  pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
  output = pooling(inputs)
  print("平均池化：\n", output)
  ```

  **输出结果:**

  ```python
  最大池化：
   tensor([[[4., 5.],
           [7., 8.]]])
  平均池化：
   tensor([[[2., 3.],
           [5., 6.]]])
  ```

- 多通道池化

  ```python
  # 2. 多通道池化
  # 定义输入数据 [3,3,3]
  inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                         [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                         [[11, 22, 33], [44, 55, 66], [77, 88, 99]]], dtype=torch.float)
  # 最大池化
  pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
  output = pooling(inputs)
  print("多通道池化：\n", output)
  ```

  **输出结果:**

  ```python
  多通道池化：
   tensor([[[ 4.,  5.],
           [ 7.,  8.]],
  
          [[50., 60.],
           [80., 90.]],
  
          [[55., 66.],
           [88., 99.]]])
  ```

课堂代码：
```python
"""
案例:
    演示池化层相关操作.

池化层解释(Pooling):
    目的:
        降维.
    思路:
        最大池化.
        平均池化.
    特点:
        池化不会改变数据的 通道数.
"""

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 演示单通道池化.
def dm01():
    # 1. 创建1个 1通道 3*3的二维矩阵.
    inputs = torch.tensor([     # 1 通道C
        [                       # 3 高度H
            [0, 1, 2],          # 3 宽度W
            [3, 4, 5],
            [6, 7, 8]
        ]
    ])
    # print(f'inputs: {inputs}, shape: {inputs.shape}')   # (1, 3, 3)

    # 2. 创建最大池化层.
    # 参1: 池化核(池化窗口)大小, 参2: 步长, 参3: 填充.
    pool1 = nn.MaxPool2d(2, 1, 0)
    outpus = pool1(inputs)
    print(f'outpus: {outpus}, shape: {outpus.shape}')   #  (1, 2, 2)

    # 3. 创建平均池化层.
    pool2 = nn.AvgPool2d(2, 1, 0)
    outpus = pool2(inputs)
    print(f'outpus: {outpus}, shape: {outpus.shape}')   # (1, 2, 2)


# 2. 定义函数, 演示多通道池化.
def dm02():
    # 1. 创建1个 3通道 3*3的二维矩阵.
    inputs = torch.tensor([     # 3 通道C
        [                       # 通道1, HW 3,3
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],

        [                       # 通道2, HW 3,3
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],

        [                       # 通道3, HW 3,3
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ]
    ])
    # print(f'inputs: {inputs}, shape: {inputs.shape}')   # (3, 3, 3)

    # 2. 创建最大池化层.
    # 参1: 池化核(池化窗口)大小, 参2: 步长, 参3: 填充.
    pool1 = nn.MaxPool2d(2, 1, 0)
    outpus = pool1(inputs)
    print(f'outpus: {outpus}, shape: {outpus.shape}')   #  (3, 2, 2)

    # 3. 创建平均池化层.
    pool2 = nn.AvgPool2d(2, 1, 0)
    outpus = pool2(inputs)
    print(f'outpus: {outpus}, shape: {outpus.shape}')   # (3, 2, 2)



# 3. 测试.
if __name__ == '__main__':
    # dm01()
    dm02()

```

## 图像分类案例

> 咱们使用前面学习到的知识来构建一个卷积神经网络, 并训练该网络实现图像分类。要完成这个案例，咱们需要学习的内容如下:
> 了解 CIFAR10 数据集
> 搭建卷积神经网络
> 编写训练函数
> 编写预测函数

导入工具包

```python
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8
```

### CIFAR10 数据集

CIFAR-10数据集5万张训练图像、1万张测试图像、10个类别、每个类别有6k个图像，图像大小32×32×3。下图列举了10个类，每一类随机展示了10张图片：

{% asset_img 1663596223425.png "Hexo 博客封面示例" %}

PyTorch 中的 torchvision.datasets 计算机视觉模块封装了 CIFAR10 数据集, 使用方法如下:

CIFAR10 是 PyTorch 官方内置封装好的标准数据集。
这里的陌生参数train=True 和 train=False，其实是官方为了方便开发者，提前帮你把数据切分好了。


```python
# 1. 数据集基本信息
def create_dataset():
    # 加载数据集:训练集数据和测试数据
    # ToTensor: 将image（一个PIL.Image对象）转换为一个Tensor
    train = CIFAR10(root='data', train=True, transform=ToTensor())
    valid = CIFAR10(root='data', train=False, transform=ToTensor())
    # 返回数据集结果
    return train, valid


if __name__ == '__main__':
    # 数据集加载
    train_dataset, valid_dataset = create_dataset()
    # 数据集类别
    print("数据集类别:", train_dataset.class_to_idx)
    # 数据集中的图像数据
    print("训练集数据集:", train_dataset.data.shape)
    print("测试集数据集:", valid_dataset.data.shape)
    # 图像展示
    plt.figure(figsize=(2, 2))
    plt.imshow(train_dataset.data[1])
    plt.title(train_dataset.targets[1])
    plt.show()
```

**输出结果：**

```python
数据集类别: {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9} 
训练集数据集: (50000, 32, 32, 3) 
测试集数据集: (10000, 32, 32, 3)
```

{% asset_img 1734675588681.png "Hexo 博客封面示例" %}

### 搭建图像分类网络

搭建的CNN网络结构如下:

{% asset_img 1663596553400.png "Hexo 博客封面示例" %}


{% asset_img Snipaste_2026-10-10_11-02-24.png "Hexo 博客封面示例" %}

全连接层只能接受二维数据，可是经过pool2后（图像是仅仅以一个通道为例形状为[(16,6,6)]），（实际是三通道一起，形状为[(N,6,16,16)]）,是三维（实际是四维的数据），故而需要进行reshape

这里所谓第一个全连接层其实是在代码中省略的，因为经过pool2后reshape出来的就是全连接层1，接下来只需要经过神经元即可，那么实际代码中只有三个全连接层

#### 对reshape的理解
结合你提供的图和之前的问题，我们来把这几个核心概念（输入通道3、576、Reshape）彻底揉碎了讲清楚。按照深度学习中经典的**CHW（通道、高度、宽度）**格式来梳理。

---

一、 当输入通道为3时



在真实的CNN（比如处理RGB彩色图片）中，输入是一个三维张量：`(3, 32, 32)`。
这里的 `3` 代表红（R）、绿（G）、蓝（B）三个通道。

当这3个通道的数据进入**第一个卷积层**时，发生的核心动作是**“跨通道融合”**：
1. 假设第一个卷积层有 **6个卷积核**。
2. 每一个卷积核的形状不再是二维的 `3x3`，而是三维的 **`3x3x3`**（高、宽、深度）。深度必须等于输入的通道数（即3）。
3. 卷积操作时，这个 `3x3x3` 的立方体滑过输入的 `(3, 32, 32)`。
4. 在每一个滑动位置，卷积核里的27个权重会与输入对应位置的27个像素值相乘并求和，最后加上一个偏置值。
5. **关键点：** 这27个数的计算，将3个通道的信息压缩成了**1个数值**。
6. 因为这层有6个不同的卷积核，所以最终输出了 **6个** 特征图，形状变成 `(6, 30, 30)`。

**结论**：输入的“3通道”在第一个卷积层就被“消化”掉了，这个三通道的概念就没有了，之后的层（Conv2、Pool2等）看到的通道数，**不再取决于输入通道数，而是取决于当前层有多少个卷积核**（比如Conv2有16个核，输出就是16通道）。

---

二、 576 是怎么算出来的？
经过第二个池化层（Pool2）后，单张图片的维度变成了 `(16, 6, 6)`。
*   `16`：因为第二个卷积层用了16个卷积核，所以输出了16个通道（特征图）。
*   `6`：经过两次池化和卷积后，特征图的高度。
*   `6`：特征图的宽度。

全连接层（FC）只能接受一维向量（或者二维的 `批次大小 x 特征数` 矩阵），它看不懂三维立体的东西。

所以我们要把 `(16, 6, 6)` 这个立体方块，**“拍扁”成一条线**。
这条线的长度就是：**16 × 6 × 6 = 576**。

这就相当于把16张尺寸为6x6的照片，首尾相连拼成一条长线，这条线上一共有576个像素点。这就是图中标注 `576x1` 以及 `576 = 16 * 6 * 6` 的由来。

---

三、 什么是 Reshape（展平/Flatten）？

**Reshape 的本质是：数据没有增加，也没有减少，仅仅是物理排列形状改变了。**

*   **形状变化**：从 `(16, 6, 6)` 变成了 `(576)`。在代码里（比如PyTorch），通常写成 `x = x.view(x.size(0), -1)` 或 `x = x.flatten(1)`。这里的 `-1` 就是让程序自动帮你算出 16*6*6=576。
*   **加上批次（Batch_Size）**：通常真实的训练数据是四维的：`(N, C, H, W)`，其中 N 是批次大小（一次训练几张图）。
    *   经过 Pool2 后，真实的张量形状是 `(N, 16, 6, 6)`。
    *   Reshape 之后，就变成了 **`(N, 576)`**。这是一个标准的二维矩阵。

**为什么全连接层必须Reshape？（数学原因）**
全连接层的计算是矩阵乘法：`Y = X * W^T + b`。
矩阵乘法要求 `X` 必须是二维的（也就是 `行数 x 列数`）。`(N, 16, 6, 6)` 是四维的，无法直接参与矩阵乘法，所以必须通过 Reshape 把它变成 `(N, 576)`。这样，权重矩阵 `W` 的形状就是 `(120, 576)`（对应图中第一个全连接层有120个神经元）。

---

### 四、 顺便纠正一下你的描述中的小误区

你在问题中说：“图像是仅仅以一个通道为例为[(6,16,16)]”、“实际是三通道一起，[(3,6,16,16)]”。

*   **问题1**：`(6, 16, 16)` 不对。经过 Pool2 后，通道数是第二层卷积核的数量（16），高宽是 6x6。所以单样本是 `(16, 6, 6)`，不是 `(6, 16, 16)`。
*   **问题2**：`(3, 6, 16, 16)` 不对。那个“3”是你一开始输入的通道数，经过两层卷积后，通道数早就变了（变成了16）。真实的四维形状应该是 `(批次大小N, 16, 6, 6)`。

**总结整个数据流（单张图片）**：
`(3, 32, 32)`  -->  Conv1(6个核) --> `(6, 30, 30)`  -->  Pool1 --> `(6, 15, 15)`  -->  Conv2(16个核) --> `(16, 13, 13)`  -->  Pool2 --> `(16, 6, 6)`  -->  **Reshape**  --> `(576)`  -->  FC1 --> `(120)`  -->  FC2 --> `(84)`  -->  FC3(输出层) --> `(10)`。


我们要搭建的网络结构如下:

1. 输入形状: 32x32
2. 第一个卷积层输入 3 个 Channel, 输出 6 个 Channel, Kernel Size 为: 3x3
3. 第一个池化层输入 30x30, 输出 15x15, Kernel Size 为: 2x2, Stride 为: 2
4. 第二个卷积层输入 6 个 Channel, 输出 16 个 Channel, Kernel Size 为 3x3
5. 第二个池化层输入 13x13, 输出 6x6, Kernel Size 为: 2x2, Stride 为: 2
6. 第一个全连接层输入 576 维, 输出 120 维
7. 第二个全连接层输入 120 维, 输出 84 维
8. 最后的输出层输入 84 维, 输出 10 维

我们在每个卷积计算之后应用 relu 激活函数来给网络增加非线性因素。

{% asset_img Snipaste_2026-10-10_11-47-22.png "Hexo 博客封面示例" %}



**构建网络代码实现如下:**

```python
# 模型构建
class ImageClassification(nn.Module):
    # 定义网络结构
    def __init__(self):
        super(ImageClassification, self).__init__()
        # 定义网络层：卷积层+池化层
        # 第一个卷积层, 输入图像为3通道,输出特征图为6通道,卷积核3*3
        self.conv1 = nn.Conv2d(3, 6, stride=1, kernel_size=3)
        # 第一个池化层, 核宽高2*2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层, 输入图像为6通道,输出特征图为16通道,卷积核3*3
        self.conv2 = nn.Conv2d(6, 16, stride=1, kernel_size=3)
        # 第二个池化层, 核宽高2*2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        # 第一个隐藏层 输入特征576个(一张图像为16*6*6), 输出特征120个
        self.linear1 = nn.Linear(576, 120)
        # 第二个隐藏层
        self.linear2 = nn.Linear(120, 84)
        # 输出层
        self.out = nn.Linear(84, 10)
        
	# 定义前向传播
    def forward(self, x):
        # 卷积+relu+池化
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        # 卷积+relu+池化
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # 将特征图做成以为向量的形式：相当于特征向量 全连接层只能接收二维数据集
        # 由于最后一个批次可能不够8，所以需要根据批次数量来改变形状
        # x[8, 16, 6, 6] --> [8, 576] -->8个样本,576个特征
        # x.size(0): 第1个值是样本数 行数
        # -1：第2个值由原始x剩余3个维度值相乘计算得到 列数(特征个数)
        x = x.reshape(x.size(0), -1)
        # 全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # 返回输出结果
        return self.out(x)


if __name__ == '__main__':
    # 模型实例化
    model = ImageClassification()
    summary(model, input_size=(3,32,32), batch_size=1)
```

{% asset_img 1734675868165.png "Hexo 博客封面示例" %}

### 编写训练函数

在训练时，使用多分类交叉熵损失函数，Adam 优化器。具体实现代码如下:

```python
def train(model, train_dataset):
    # 构建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss() # 构建损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 构建优化方法
    epoch = 100  # 训练轮数
    for epoch_idx in range(epoch):
        sum_num = 0   # 样本数量
        total_loss = 0.0  # 损失总和
        correct = 0  # 预测正确样本数
        start = time.time()  # 开始时间
        # 遍历数据进行网络训练
        for x, y in dataloader:
            model.train()
            output = model(x)
            loss = criterion(output, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            correct += (torch.argmax(output, dim=-1) == y).sum()  # 计算预测正确样本数
            # 计算每次训练模型的总损失值 loss是每批样本平均损失值
            total_loss += loss.item()*len(y)  # 统计损失和
            sum_num += len(y)
        print('epoch:%2s loss:%.5f acc:%.2f time:%.2fs' %(epoch_idx + 1,total_loss / sum_num,correct / sum_num,time.time() - start))
    # 模型保存
    torch.save(model.state_dict(), 'model/image_classification.pth')
            

if __name__ == '__main__':
    # 数据集加载
    train_dataset, valid_dataset = create_dataset()
    # 模型实例化
    model = ImageClassification()
    # 模型训练
    train(model,train_dataset)
```

**输出结果：**

```python
epoch: 1 loss:1.59926 acc:0.41 time:28.97s
epoch: 2 loss:1.32861 acc:0.52 time:29.98s
epoch: 3 loss:1.22957 acc:0.56 time:29.44s
epoch: 4 loss:1.15541 acc:0.59 time:30.45s
epoch: 5 loss:1.09832 acc:0.61 time:29.69s
...
epoch:96 loss:0.30592 acc:0.89 time:37.28s
epoch:97 loss:0.29255 acc:0.90 time:37.11s
epoch:98 loss:0.29470 acc:0.90 time:36.98s
epoch:99 loss:0.29472 acc:0.90 time:36.79s
epoch:100 loss:0.29903 acc:0.90 time:37.66s
```

### 编写预测函数

加载训练好的模型，对测试集中的1万条样本进行预测，查看模型在测试集上的准确率。

```python
def test(valid_dataset):
    # 构建数据加载器
    dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 加载模型并加载训练好的权重
    model = ImageClassification()
    model.load_state_dict(torch.load('model/image_classification.pth'))
    # 模型切换评估模式, 如果网络模型中有dropout/BN等层, 评估阶段不进行相应操作
    model.eval()
    # 计算精度
    total_correct = 0
    total_samples = 0
    # 遍历每个batch的数据，获取预测结果，计算精度
    for x, y in dataloader:
        output = model(x)
        total_correct += (torch.argmax(output, dim=-1) == y).sum()
        total_samples += len(y)
    # 打印精度
    print('Acc: %.2f' % (total_correct / total_samples))
    

if __name__ == '__main__':
    test(valid_dataset)
```

**输出结果：**

```python
Acc: 0.57
```

###  模型优化

CNN网络模型在训练集样本上的准确率远远高于测试集,说明模型产生了过拟合问题,我们把学习率由1e-3修改为1e-4、增加网络参数量和增加dropout正则化

```python
class ImageClassification(nn.Module):
	def __init__(self):
		super(ImageClassification, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(32, 128, stride=1, kernel_size=3)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.linear1 = nn.Linear(128 * 6 * 6, 2048)
		self.linear2 = nn.Linear(2048, 2048)
		self.out = nn.Linear(2048, 10)
		# Dropout层，p表示神经元被丢弃的概率
		self.dropout = nn.Dropout(p=0.5)

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = self.pool1(x)
		x = torch.relu(self.conv2(x))
		x = self.pool2(x)
		# 由于最后一个批次可能不够 32，所以需要根据批次数量来 flatten
		x = x.reshape(x.size(0), -1)
		x = torch.relu(self.linear1(x))
		# dropout正则化
		# 训练集准确率远远高于测试准确率,模型产生了过拟合
		x = self.dropout(x)
		x = torch.relu(self.linear2(x))
		x = self.dropout(x)
		return self.out(x)
```

经过训练，模型在测试集的准确率由 0.57，提升到了 0.93，同学们也可以自己修改相应的网络结构、训练参数等来提升模型的性能。



课堂代码：
```python
"""
案例:
    演示CNN的综合案例, 图像分类.

回顾: 深度学习项目的步骤
    1. 准备数据集.
        这里我们用的时候 计算机视觉模块 torchvision自带的 CIFAR10数据集, 包含6W张 (32,32,3)的图片, 5W张训练集, 1W张测试集, 10个分类, 每个分类6K张图片.
        你需要单独安装一下 torchvision包, 即: pip install torchvision
    2. 搭建(卷积)神经网络
    3. 模型训练.
    4. 模型测试.

卷积层:
    提取图像的局部特征 -> 特征图(Feature Map), 计算方式:  N = (W - F + 2P) // S + 1
    每个卷积核都是1个神经元.

池化层:
    降维, 有最大池化 和 平均池化.
    池化只在HW上做调整, 通道上不改变.

案例的优化思路:
    1. 增加卷积核的输出通道数(大白话: 卷积核的数量)
    2. 增加全连接层的参数量.
    3. 调整学习率
    4. 调整优化方法(optimizer...)
    5. 调整激活函数...
    6. ...
"""

# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8


# 1. 准备数据集.
def create_dataset():
    # 1. 获取训练集.
    # 参1: 数据集路径. 参2: 是否是训练集. 参3: 数据预处理 -> 张量数据. 参4: 是否联网下载(直接用我给的, 不用下)
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 2. 获取测试集.
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 3. 返回数据集.
    return train_dataset, test_dataset


# 2. 搭建(卷积)神经网络
class ImageModel(nn.Module):
    # 1. 初始化父类成员, 搭建神经网络.
    def __init__(self):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 搭建神经网络.
        # 第1个卷积层, 输入 3通道, 输出6通道, 卷积核大小3*3, 步长1, 填充0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)

        # 第1个池化层, 窗口大小 2*2, 步长2, 填充0
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        # 第2个卷积层, 输入 6通道, 输出16通道, 卷积核大小3*3, 步长1, 填充0
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)

        # 第2个池化层, 窗口大小 2*2, 步长2, 填充0
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 第1个隐藏层(全连接层), 输入: 576, 输出: 120
        self.linear1 = nn.Linear(576, 120)
        # 第2个隐藏层 (全连接层), 输入: 120, 输出: 84
        self.linear2 = nn.Linear(120, 84)
        # 第3个隐藏层 (全连接层) -> 输出层,  输入: 84, 输出: 10
        self.output = nn.Linear(84, 10)

    # 2. 定义前向传播
    def forward(self, x):
        # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        # 分解版.
        # x = self.conv1(x)   # 卷积层
        # x = torch.relu(x)   # 激励层
        # x = self.pool1(x)   # 池化层

        # 合并版   池化   +  激活函数  +  卷积
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool2(torch.relu(self.conv2(x)))

        # 细节: 全连接层只能处理二维数据, 所以要将数据进行拉平 (8, 16, 6, 6) ->  (8, 576)
        # 参1: 样本数(行数), 参2: 列数(特征数), -1表示自动计算.
        x = x.reshape(x.size(0), -1)  # 8行576列
        # print(f'x.shape: {x.shape}')

        # 第3层: 全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear1(x))

        # 第4层: 全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear2(x))

        # 第5层: 全连接层(加权求和) -> 输出层
        return self.output(x)  # 后续用 多分类交叉熵损失函数CrossEntropyLoss = softmax()激活函数 + 损失计算.


# 3. 模型训练.
def train(train_dataset):
    # 1. 创建数据加载器.
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 2. 创建模型对象.
    model = ImageModel()
    # 3. 创建损失函数对象.
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数 = softmax()激活函数 + 损失计算.
    # 4. 创建优化器对象.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. 循环遍历epoch, 开始 每轮的 训练动作.
    # 5.1 定义变量, 记录训练的总轮数.
    epochs = 20

    # 5.2 遍历, 完成每轮的 所有批次的 训练动作.
    for epoch_idx in range(epochs):
        # 5.2.1 定义变量, 记录: 总损失, 总样本数据量, 预测正确样本个数, 训练(开始)时间
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        # 5.2.2 遍历数据加载器, 获取到 每批次的 数据.
        for x, y in dataloader:

            # 5.2.3 切换训练模式.
            model.train()

            # 5.2.4 模型预测.
            y_pred = model(x)

            # 5.2.5 计算损失.
            loss = criterion(y_pred, y)

            # 5.2.6 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5.2.7 统计预测正确的样本个数.
            # print(y_pred)     # 批次中, 每张图 每个分类的 预测概率.

            # argmax() 返回最大值对应的索引, 充当 -> 该图片的 预测分类.
            # tensor([9, 8, 5, 5, 1, 5, 8, 5])
            # print(torch.argmax(y_pred, dim=-1))     # -1这里表示行.   预测分类
            # print(y)                                # 真实分类
            # print(torch.argmax(y_pred, dim=-1) == y)    # 是否预测正确
            # print((torch.argmax(y_pred, dim=-1) == y).sum())    # 预测正确的样本个数.
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()

            # 5.2.8 统计当前批次的总损失.          第1批平均损失 * 第1批样本个数
            total_loss += loss.item() * len(y)  # [第1批总损失 +  第2批总损失 +  第3批总损失 +  ...]

            # 5.2.9  统计当前批次的总样本个数.
            total_samples += len(y)

            # break   每轮只训练1批, 提高训练效率, 减少训练时长, 只有测试会这么写, 实际开发绝不要这样做.

        # 5.2.10 走这里, 说明一轮训练完毕, 打印该轮的训练信息.
        print(f'epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.2f}, time:{time.time() - start:.2f}s')
        # break     # 这里写break, 意味着只训练一轮.

    # 6. 保存模型.
    torch.save(model.state_dict(), './model/image_model.pth')

# 4. 模型测试.
def evaluate(test_dataset):
    # 1. 创建测试集 数据加载器.
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 创建模型对象.
    model = ImageModel()

    # 3. 加载模型参数.
    model.load_state_dict(torch.load('./model/image_model.pth'))    # pickle文件

    # 4. 定义变量统计 预测正确的样本个数, 总样本个数.
    total_correct, total_samples = 0, 0

    # 5. 遍历数据加载器, 获取到 每批次 的数据.
    for x, y in dataloader:
        
        # 5.1 切换模型模式.
        model.eval()

        # 5.2 模型预测.
        y_pred = model(x)

        # 5.3 因为训练的时候用了CrossEntropyLoss, 所以搭建神经网络时没有加softmax()激活函数, 这里要用 argmax()来模拟.
        # argmax()函数功能:  返回最大值对应的索引, 充当 -> 该图片的 预测分类.
        y_pred = torch.argmax(y_pred, dim=-1)   # -1 这里表示行.

        # 5.4 统计预测正确的样本个数.
        total_correct += (y_pred == y).sum()

        # 5.5 统计总样本个数.
        total_samples += len(y)

    # 6. 打印正确率(预测结果).
    print(f'Acc: {total_correct / total_samples:.2f}')


# 5. 测试
if __name__ == '__main__':
    # 1. 获取数据集.
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集: {train_dataset.data.shape}')       # (50000, 32, 32, 3)
    # print(f'测试集: {test_dataset.data.shape}')        #  (10000, 32, 32, 3)
    # # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'数据集类别: {train_dataset.class_to_idx}')
    #
    # # 图像展示
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[1111])      # 索引为1111的图像
    # plt.title(train_dataset.targets[1111])
    # plt.show()

    # 2. 搭建神经网络.
    # model = ImageModel()
    # 查看模型参数, 参1: 模型, 参2: 输入维度(CHW, 通道, 高, 宽), 参3: 批次大小
    # summary(model, (3, 32, 32), batch_size=1)

    # 3. 模型训练.
    # train(train_dataset)

    # 4. 模型测试.
    evaluate(test_dataset)

```

训练结果：
{% asset_img Snipaste_2026-10-10_12-53-01.png "Hexo 博客封面示例" %}
(10,576),10就是批次，意味着10张图片一起，每张图片576个特征，根据这些特征预测图片类别
{% asset_img Snipaste_2026-10-10_12-57-47.png "Hexo 博客封面示例" %}
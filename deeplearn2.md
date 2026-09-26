---
title: 深度学习的框架
date: 2026-09-26 19:34:49
tags:
---


李沐在《动手学深度学习》里讲数据操作的时候，推荐 PyTorch 的 tensor，而不是 NumPy 的 array。
同样为张量类但是NumPy 的数组运算默认在 CPU 上跑。一个稍微像样的模型，参数几百万起步，矩阵乘法每秒要做几十亿次。CPU 那点算力，训到猴年马月。GPU 不一样，它天生适合做大规模并行计算，做矩阵乘法等。但 NumPy 不支持 GPU，你得自己写 CUDA 代码。

其次，不会自动求导。 反向传播需要算梯度。你要是用 NumPy，就得手推每个参数的梯度公式，然后一行行写出来。网络一深，梯度公式能写到你怀疑人生。而且改一下网络结构，梯度推导就得重来。



框架的作用就呼之欲出了：
把张量放到 GPU 上算
自动帮你算梯度，我们只用搭网络、写前向传播，反向传播它自己搞定。

其实PyTorch 的官方教程说得很直白：PyTorch 核心就两个东西——一个能跑在 GPU 上的 n 维张量，和一个自动微分引擎。

举个例子吧
不写框架，用 NumPy 拟合一个多项式
import numpy as np

x = np.linspace(-np.pi, np.pi, 2000)
y = np.sin(x)

a = np.random.randn()
b = np.random.randn()
# ... 初始化 c, d

for t in range(2000):
    y_pred = a + b * x + c * x**2 + d * x**3
    loss = np.square(y_pred - y).sum()

    # 手动算梯度，手推公式
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # 手动更新
    a -= learning_rate * grad_a
    # ...

这是最简单的三阶多项式，只有一个输入特征，梯度公式就已经长这样了。要是换成多层神经网络，几百个参数，呃，我不行了

但是使用 PyTorch 写
import torch

x = torch.linspace(-torch.pi, torch.pi, 2000)
y = torch.sin(x)

a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
# ...

for t in range(2000):
    y_pred = a + b * x + c * x**2 + d * x**3
    loss = (y_pred - y).pow(2).sum()

    loss.backward()      # 就这一行，梯度全算好了

    with torch.no_grad():
        a -= learning_rate * a.grad
        # ...
        a.grad.zero_()
        # ...

调一下loss.backward() ，所有 requires_grad=True 的张量，.grad 属性里就存好了对应的梯度。你不用管公式长什么样，框架的自动微分引擎会帮我们计算好


那么接下来介绍以下主流框架

PyTorch，Meta（就是以前的 Facebook）2016 年开源的。它的前身是 Torch，一个用 Lua 写的框架，Meta 的研究员把它用 Python 重写了一遍，就成了 PyTorch。现在由 Linux 基金会下面的 PyTorch 基金会管，不只是 Meta 一家说了算。

TensorFlow，Google 2015 年开源。它的前身是 Google Brain 团队内部用的 DistBelief，Google 把经验教训总结之后，重新搞了一套开源出来。TensorFlow 1.x 时代用的是静态计算图，写起来比较绕。2.x 之后默认改成动态执行了，加上 Keras 的集成，上手难度降了不少。

JAX，Google 后来搞的另一个东西。它长得特别像 NumPy，但能在 GPU 和 TPU 上跑，而且做了大量编译优化。JAX 不是传统意义上的“深度学习框架”，更像一个高性能数值计算库，深度学习只是它能干的事情之一。Anthropic、xAI 这些做头部大模型的公司也在用它。

飞桨（PaddlePaddle） ，百度的，国内第一个自主研发的开源深度学习平台。它的定位很明确：产业级。围绕它有一整套工具，OCR、图像分类、目标检测、部署推理，该有的都有。它的 PaddleOCR 在 GitHub 上 Star 数过了 7 万，是全球 Star 最高的 OCR 项目。


延申概念：

动态图和静态图

静态图，是先把整个计算流程“画”好，定义成一个图，然后再把数据喂进去跑。好处是框架可以提前优化整个图，跑起来快。坏处是调试很痛苦，因为没法像写普通 Python 一样，中间打断点看看某个变量到底是多少。TensorFlow 1.x 就是这个方法。

动态图，是写一行代码，它就跑一行。跟写普通 Python 没区别。可以在循环里打断点，可以打印中间张量，可以随便改网络结构。PyTorch 从第一天起就是这个设计。

JetBrains 的博客有一句话：PyTorch 的动态图让调试变得自然，你用标准的 Python 工具就行，训练循环里随便设断点，执行到一半也能检查张量值。

这也是为什么 PyTorch 在学术界几乎一统天下——顶级 AI 会议上 85% 的深度学习论文用的是 PyTorch。研究者要的就是灵活和好调试，动态图正好满足这个需求。

TensorFlow 2.x 也改成了动态执行，但保留了静态图编译的选项，算是两条路都给你留着。

TensorFlow在工业部署上更成熟，TF Serving、TF Lite、TensorFlow.js 这一套部署工具链，是经过大规模生产验证的。

JAX 适合已经有一定基础，想在大规模训练上榨性能的人。它的函数式编程范式跟 PyTorch 差别很大，纯函数、不可变状态、grad 是函数变换而不是方法调用。我直接被劝退ing

飞桨适合什么人？适合在国内做产业落地、需要适配国产硬件的人。它的硬件适配层支持超过 60 款芯片，跟国产 GPU 的兼容性做得比较到位。如果你做的是国内政企项目，飞桨可能是更合适的选择。


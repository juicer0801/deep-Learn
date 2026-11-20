---
title: transformer理解
date: 2026-10-14 21:27:51
tags:
---

Encoder 的每一层：包含 1 个多头自注意力模块 + 1 个前馈神经网络（FFN）。

Decoder 的每一层：包含 2 个多头注意力模块（一个是带掩码的自注意力，一个是交叉注意力） + 1 个前馈神经网络（FFN）。

注意：这就是为什么 Decoder 的每一层比 Encoder 的每一层更复杂、计算量更大。

Encoder（编码器大楼）：有 
N 层（比如原版 Transformer 是 6 层）。

Decoder（解码器大楼）：有 
M 层（通常也是 6 层，但可以不等）。

关系：它们是完全独立的两个堆叠模块。Decoder 的每一层会去“眺望”Encoder 对应那一层的输出（做 Cross-Attention）。

在任何一个多头注意力模块内部，都包含了 
n
n 个并行的“头”（Heads）。

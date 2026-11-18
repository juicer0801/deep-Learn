---
title: 注意力机制
date: 2026-10-14 21:27:44
tags:
---
以下是基于上述对话内容整理的严谨版学习笔记：

---

#### 一、 注意力机制的核心计算流程
注意力机制的核心本质是**加权求和（Weighted Sum）**，分为以下三个步骤：

1.  **相似度打分（Score Calculation）**：
    *   将解码器当前时间步的状态（Query）与编码器的所有隐藏状态（Keys）进行相似度计算（通常通过点积或全连接层）。
    *   *例：图中解码器状态与两个编码器状态计算，得出分数 -0.55 和 -0.52。*
2.  **权重分配（Attention Weights）**：
    *   将得到的分数输入第一个 Softmax 层进行归一化，使所有分数之和为 1，得到注意力权重。
    *   *例：Softmax 后得到权重 0.49 和 0.51。*
3.  **加权求和生成上下文向量（Context Vector）**：
    *   将注意力权重与对应的编码器状态（Values）逐项相乘，并将所有乘积相加。
    *   *数学公式：`$Context Vector = \sum (Attention\_Weight_i \times Encoder\_State_i)$`*

#### 二、 解码器输出与自回归生成
1.  **融合与预测**：
    *   计算出的**上下文向量**与**解码器当前状态**进行拼接（Concatenate）或相加。
    *   融合后的向量经过全连接层（Fully Connected Layer），输出 Logits（未归一化得分）。
    *   经过第二个 Softmax 层，计算出词表（Vocabulary）中每个词的概率分布。
2.  **自回归循环（Autoregressive Loop）**：
    *   当前时间步选出的概率最大的词，将在下一个时间步作为输入（Input）喂回解码器。
    *   解码器状态更新后，重新计算 Query，重复上述注意力及预测步骤，直到生成结束符 `<EOS>` 或达到最大长度。
3.  **训练与推理的差异**：
    *   **训练阶段（Teacher Forcing）**：下一步的输入强制使用真实标签（Ground Truth）中的词，而非模型预测出的词，以加速收敛。
    *   **推理阶段（Free Running）**：下一步的输入使用模型上一步自己预测出的词。

#### 三、 关键概念与细节辨析

1.  **两种 Softmax 的本质区别**：
    *   **注意力 Softmax（打分用）**：输入为相似度分数，输出为注意力权重。数值为连续的小数，表示对各编码状态的“关注度”百分比。
    *   **预测 Softmax（输出用）**：输入为全连接层得分，输出为词表概率。数值向两端极化（接近 1 或 0）。
    *   **图解中的 0 和 1**：书上的 `[0, 1, 0, 0]` 并非 Softmax 的真实概率输出，而是经过 Argmax 筛选后转化为的 **One-Hot 编码**，代表最终选中的词。

2.  **矩阵在框架中的意义（并行计算）**：
    *   在网络底层（如 PyTorch）实现时，不会用 for 循环逐个计算。编码器的所有输出状态会被“收纳”为一个矩阵（Tensor）。
    *   通过 `torch.matmul` 进行矩阵乘法，可以**并行、一次性**计算出 Query 与所有 Keys 的相似度，以及一次性完成加权求和操作，极大提升 GPU 计算效率。

3.  **多层 LSTM（Stacked LSTM）的处理**：
    *   注意力机制通常只与编码器**最顶层（最后一层）**的 LSTM 输出发生交互。
    *   底层 LSTM 提取词法、句法特征，顶层 LSTM 包含完整的高阶语义信息。通常取顶层输出构成 `[序列长度, 隐藏层维度]` 的矩阵作为 Keys 和 Values。
    *   若解码器也是多层，Query 同样取解码器最顶层的状态。

4.  **多次注意力计算的独立性**：
    *   在整个解码过程中，编码器输出的键值矩阵（Keys/Values）保持不变。
    *   解码器每生成一个词，内部状态都会更新（生成新的 Query），因此需要**重新计算**注意力权重，生成**全新的上下文向量**。

    以下是根据你的手写草稿与书本架构，结合严谨的术语整理的完整流程图。

由于文本界面无法手绘，这里采用 **Mermaid 流程图语法**（支持 Markdown 渲染）。你可以直接查看图表，或将其复制到支持 Mermaid 的编辑器（如 Typora、Notion）中查看图形。

---

### 核心流程图：基于注意力机制的 Seq2Seq 模型

```mermaid
graph TD
    %% 编码阶段
    subgraph Encoder [编码器阶段 (Encoder)]
        A[输入序列<br>词向量序列] --> B[多层 LSTM 网络]
        B -->|提取高阶语义信息| C(编码器隐藏状态矩阵 H <br> 维度: 序列长度 × 隐藏层维度)
        C -->|作为 Keys 和 Values| Attention[注意力模块]
    end

    %% 注意力模块 (对应你的草稿本核心)
    subgraph AttentionModule [注意力模块 (Attention)]
        Q[解码器当前隐藏状态<br>(Query)] --> D{矩阵乘法<br>计算相似度}
        C -->|传入 Keys| D
        D -->|得出所有位置的<br>相似度得分 (Logits)| E[注意力 SoftMax]
        E -->|归一化| F[注意力权重矩阵]
        C -->|传入 Values| G{加权求和<br>(矩阵乘法并行计算)}
        F -->|传入权重| G
        G -->|得出| H[上下文向量 (Context Vector)]
    end

    %% 解码与预测模块 (对应草稿本右侧)
    subgraph Decoder [解码与预测阶段 (Decoder)]
        Q -->|同时作为解码输入| I[拼接层 (Concatenate)]
        H -->|融合上下文| I
        I --> J[全连接层 (FC Layer)]
        J -->|输出未归一化得分| K[预测 SoftMax]
        K -->|输出词表概率分布| L{Argmax 选取<br>概率最大的词}
    end

    %% 循环反馈机制
    subgraph Loop [自回归循环 (Autoregressive)]
        L --> M{是否为 <EOS> ?}
        M -->|否| N(将预测出的词作为下一步输入<br>更新解码器隐藏状态)
        N -->|生成新的 Query| Q
        M -->|是| O[输出最终序列<br>结束解码]
    end
    
    %% 样式调整
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef encoder fill:#e1f5fe,stroke:#03a9f4;
    classDef attention fill:#f3e5f5,stroke:#9c27b0;
    classDef decoder fill:#fff3e0,stroke:#ff9800;
    classDef loop fill:#e8f5e9,stroke:#4caf50;
    
    class Encoder encoder;
    class AttentionModule attention;
    class Decoder decoder;
    class Loop loop;
```


#### 一、 编码器阶段 (Encoder)
*   **过程**：输入序列经过多层 LSTM，注意力机制仅提取**最顶层（最后一层）**的隐藏状态。
*   **结果**：将所有时间步的输出拼接为**编码器隐藏状态矩阵 \(H\)**。该矩阵同时充当注意力机制中的 **Keys（用于求相似度）** 和 **Values（用于加权求和）**。

#### 二、 注意力模块 (Attention) —— *对应草稿本左侧*
*   **计算相似度**：解码器的当前状态（Query）与矩阵 \(H\)（Keys）进行**矩阵乘法**。这一步用并行运算替代了草稿本中的“某个编码值 × 解码值”。
*   **第一个 SoftMax**：将相似度得分输入 Softmax，得出**注意力权重**。*(注：此处输出连续小数，非 0/1)*。
*   **加权求和**：将注意力权重与矩阵 \(H\)（Values）再次进行矩阵乘法。
*   **结果**：得出 **上下文向量 (Context Vector)**，代表当前解码步对整句输入信息的凝练。*(注：草稿本中的“重复并求和”在此由矩阵乘法一次性完成)*。

#### 三、 解码与预测阶段 (Decoder) —— *对应草稿本右侧*
*   **信息融合**：上下文向量 与 解码器当前状态（Query）进行**拼接 (Concatenate)**。*(对应草稿本中的大括号汇合)*。
*   **全连接**：融合后的向量进入**全连接层 (Linear)**，映射到词表维度。
*   **第二个 SoftMax**：全连接输出输入 Softmax，得到**词表概率分布**。*(注：概率最高者经 Argmax 变为 1，其余变为 0，图示中的 [0,1,0,0] 即为该 One-Hot 结果)*。

#### 四、 自回归循环 (Autoregressive Loop)
*   **判断终止**：若预测出的词为 `<EOS>`（结束符），则停止解码。
*   **状态更新**：若未结束，将预测出的词作为下一时刻的输入，更新解码器的隐藏状态（生成**新的 Query**）。
*   **重新计算**：携带新的 Query 返回注意力模块，重新计算权重，生成**全新的上下文向量**。
*   *训练阶段补充（Teacher Forcing）*：模型训练时，此处的输入强制使用真实标签词（Ground Truth），而非模型预测词，以防误差累积。
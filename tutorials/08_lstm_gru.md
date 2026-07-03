---
title: "08. LSTM 和 GRU：门控记忆"
status: "已完成"
summary: "在同一个字符级分类任务上比较普通 RNN、GRU 和 LSTM，理解门控如何缓解长依赖问题。"
---

# 08. LSTM 和 GRU：门控记忆

## 本章学习契约

- 新增概念：门控、cell state、update/reset gate、长依赖。
- 实验要验证：LSTM/GRU 通过门控机制，比普通 RNN 更容易保留或丢弃序列中的信息。
- 实验不验证：它不证明门控网络能解决所有长上下文问题，也不和 Transformer 做最终性能竞赛。
- 跑完重点看：RNN/LSTM/GRU 的参数量、验证指标、收敛速度，以及输出里不同 cell 的对比。

普通 RNN 的 hidden state 每个时间步都会被整体更新：

```text
h_t = RNNCell(x_t, h_{t-1})
```

这个结构很简洁，但也有一个问题：模型缺少明确机制来决定“哪些旧信息应该保留，哪些新信息应该写入”。序列变长后，早期信息很容易被后面的更新冲淡。

LSTM 和 GRU 的核心想法是加入门控。门控不是玄学，它本质上是一组取值在 0 到 1 之间的连续权重，像逐元素的旋钮一样控制信息流动。

门控权重通常来自 sigmoid：

```text
gate = sigmoid(...)
```

所以它的每个元素都在 0 和 1 之间。接近 0 表示“少通过一点”，接近 1 表示“多通过一点”。它和普通线性层权重不同：线性层权重直接参与加权求和，门控更像对信息流做逐元素开关或比例控制。

## LSTM 的直觉

LSTM 维护两类状态：

- hidden state：当前时间步对外输出的表示；
- cell state：更像一条长期记忆通道。

它有几个常见门：

- forget gate：决定旧记忆保留多少；
- input gate：决定新信息写入多少；
- output gate：决定当前输出暴露多少。

你不需要一开始背公式。先记住一句话：LSTM 把“记住”和“忘掉”变成了可学习的动作。

## GRU 的直觉

GRU 比 LSTM 更简洁。它没有单独的 cell state，而是用 hidden state 承担记忆，并通过两个主要门控制更新：

- update gate：旧状态保留多少，新状态写入多少；
- reset gate：生成候选状态时，过去信息参与多少。

GRU 通常参数更少、速度更快；LSTM 更经典，长期依赖场景下仍然很常见。

## 为什么门控能缓解长依赖

普通 RNN 每一步都把旧状态和新输入混在一起更新。时间步很多时，早期信息要穿过很长的计算链，梯度容易变小或变大。

门控结构给模型提供了更稳定的信息通路。比如 LSTM 的 cell state 可以在多个时间步里相对平滑地传递，模型再通过门决定什么时候写入、遗忘和输出。

这不代表 LSTM/GRU 彻底解决所有长上下文问题，但它们确实比普通 RNN 更擅长处理中等长度的序列依赖。

## 运行对比实验

```bash
uv run python -m llm_tutor.experiments.compare_recurrent_cells --epochs 12
```

快速 smoke test：

```bash
uv run python -m llm_tutor.experiments.compare_recurrent_cells --epochs 1
```

脚本会在同一个内置姓名分类任务上训练三种模型：

```text
rnn
gru
lstm
```

最后输出：

```text
summary
cell   val_loss  val_acc  test_acc  test_macro_f1
rnn       ...
gru       ...
lstm      ...
```

这个数据集很小，所以不要把一次结果读成“谁永远更好”。它的作用是让你看到三种 cell 可以在同一训练框架下互换，并理解它们差异来自 recurrent cell 内部。

还要注意：脚本使用相同 `hidden_size`，但 RNN、GRU、LSTM 的参数量并不相同。因此这不是同参数量的公平 benchmark，而是教学对比。

如果输出里某次普通 RNN 反而更高，也不要紧张。小数据集、短训练、随机初始化都会影响结果。本章更重要的观察是：三种 cell 的输入输出接口几乎一致，但内部记忆机制不同。

## 这个实验能说明什么

- RNN、GRU、LSTM 都可以处理序列输入。
- GRU/LSTM 在普通 RNN 的基础上加入了可学习门控。
- 同一个数据、训练循环和分类头可以比较不同 recurrent cell。

## 这个实验不能证明什么

- 它不能证明 LSTM 或 GRU 在所有任务上优于 RNN。
- 它不能证明小样本姓名分类能代表真实 NLP。
- 它不能解决后面机器翻译和语言模型里的全部长依赖问题。

## 和后续章节的关系

下一步我们会从“分类一个序列”走向“输入一个序列，输出另一个序列”。这就是 Seq2Seq。

RNN/LSTM/GRU 可以作为 Seq2Seq 的 encoder 和 decoder。当前分类任务只用最后表示做一次预测；Seq2Seq 会把 encoder 状态交给 decoder，让 decoder 一步步生成输出序列。等 Seq2Seq 遇到瓶颈时，注意力机制就会自然出现。

上一章：[07. RNN：让模型读序列](07_rnn_sequence_modeling.md)  
下一章：[09. Seq2Seq：从分类到生成](09_seq2seq_translation.md)

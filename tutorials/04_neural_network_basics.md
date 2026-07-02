---
title: "04. 神经网络基础：从线性模型到多层非线性函数"
status: "已完成"
summary: "理解多层线性变换、非线性激活和表示学习如何组成最小神经网络。"
---

# 04. 神经网络基础：从线性模型到多层非线性函数

线性模型只能表达一种形式：

```text
x @ W + b
```

如果数据规律本身比较复杂，单层线性变换就不够了。神经网络的基本想法是：把多个可学习变换和非线性函数组合起来。

## 为什么只堆线性层没有意义

如果没有激活函数：

```text
Linear -> Linear -> Linear
```

无论堆多少层，整体仍然等价于一个更大的线性变换。它不会获得真正新的表达能力。

所以神经网络中间需要非线性激活，例如 ReLU：

```text
Linear -> ReLU -> Linear -> ReLU -> Linear
```

ReLU 的形式很简单：

```text
ReLU(x) = max(0, x)
```

它让模型可以组合出分段非线性的决策边界。

## 神经网络为什么能起作用

可以从三个层面理解：

1. 每一层把输入变成一种新的表示。
2. 非线性激活让多层组合不再退化成一个线性模型。
3. 反向传播和优化器会根据 loss 调整所有层的参数。

模型并不是被人手写规则，而是在训练过程中自己找到一组参数，让输入经过层层变换后更容易被分类。

## 本章模型

代码在 `src/llm_tutor/models/feedforward.py`：

```text
输入特征
-> Linear
-> ReLU
-> Linear
-> ReLU
-> Linear
-> 类别 logits
```

这个模型仍然很小，但已经包含神经网络最重要的几个组件：层、激活和输出 logits。Dropout、weight decay 这类正则化技巧会放到下一章单独讲。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_neural_network_basics --epochs 30
```

可以和第一章线性分类器对比：

```bash
uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 30
```

观察两者在验证集和测试集上的表现。不要只看最终 accuracy，也要看 loss 曲线是否稳定、是否过拟合。

样例输出大致会长这样：

```text
epoch=01 train_loss=... val_loss=... val_acc=... val_f1=...
...
test loss=... acc=... malignant_precision=... malignant_recall=... malignant_f1=...
```

在这个小表格数据集上，神经网络不一定显著超过线性模型。这是正常现象：数据本身可能已经接近线性可分，模型更复杂不等于一定更好。

## 这一章要留下的直觉

神经网络不是一团神秘公式。它仍然是：

```text
前向传播 -> 损失函数 -> 反向传播 -> 参数更新
```

只是模型从单层线性函数，变成了多层非线性函数。后面的 CNN、RNN、Transformer，本质上也都遵循这个训练闭环，只是结构中加入了更适合图像、序列或文本生成的归纳偏置。

## 同一个闭环在 GPT 里长什么样

现在的分类模型是：

```text
x [batch, features]
-> logits [batch, classes]
-> labels [batch]
-> cross entropy loss
```

GPT 训练时会变成：

```text
input_ids [batch, seq_len]
-> logits [batch, seq_len, vocab_size]
-> labels [batch, seq_len]
-> cross entropy loss
```

区别看起来很大，但训练动作仍然是同一套：前向传播、计算 loss、反向传播、更新参数。后面所有复杂结构都不会改变这个核心。

## 这个实验不能证明什么

- 它不能证明神经网络总比线性模型强。
- 它不能证明层数越多越好。
- 它不能证明模型学到了人能理解的医学规则。

## 常见失败

- 训练 loss 下降很慢：可以适当增大学习率，或检查输入是否标准化。
- 训练集很好、验证集变差：可能是模型过拟合，下一章会讲正则化。
- 神经网络没超过线性模型：这不是 bug，小数据集上常常如此。

上一章：[03. PyTorch 快速生存指南](03_pytorch_training_loop.md)  
下一章：[05. 训练神经网络的基本功](05_training_neural_networks.md)

---
title: "06. CNN 为什么适合图像"
status: "已完成"
summary: "用 Fashion-MNIST 图像分类理解卷积核、局部感受野、参数共享、池化和图像张量 shape。"
---

# 06. CNN 为什么适合图像

## 本章学习契约

- 新增概念：图像张量、卷积核、局部感受野、参数共享、池化。
- 实验要验证：当输入从表格变成图像时，训练闭环不变，但模型结构要尊重数据的空间结构。
- 实验不验证：它不是视觉方向完整入门，也不能说明 Fashion-MNIST 高分等于真实视觉可靠。
- 跑完重点看：输入 shape 是否是 `[batch, channels, height, width]`，`val_acc`/`macro_f1` 是否随训练改善，第一次运行是否成功缓存数据。

前几章处理的是表格特征：

```text
x [batch, num_features]
```

图像不一样。图像天然有空间结构：相邻像素之间的关系很重要，横向、纵向、局部纹理都携带信息。

你可能会问：既然后面重点是 LLM，为什么还要学 CNN？原因不是要把你带到视觉方向，而是 CNN 是理解“模型结构如何利用数据形状”的最好例子之一。

到目前为止，输入形状在一路变化：

| 阶段 | 输入 shape | 模型主要偏置 |
|---|---|---|
| 表格 | `[batch, features]` | 每个特征是普通数值列 |
| 图像 | `[batch, channels, height, width]` | 局部空间模式和参数共享 |
| 文本序列 | `[batch, seq_len]` | 顺序、上下文和 token 依赖 |

训练闭环仍然是前向传播、loss、反向传播、更新参数；真正变化的是模型如何看待输入结构。

在 PyTorch 里，一批灰度图像通常长这样：

```text
x [batch, channels, height, width]
```

Fashion-MNIST 是 28x28 灰度图，所以输入 shape 是：

```text
[batch, 1, 28, 28]
```

## MLP 看图像的问题

如果把图像直接拉平成一条向量：

```text
[1, 28, 28] -> [784]
```

普通全连接网络确实也能训练，但它会丢掉一个重要直觉：相邻像素本来是相邻的。模型必须从数据里重新学会“局部像素组合很重要”这件事。

CNN 直接把这个直觉写进结构里。

参数量也能说明问题。第一层 `Conv2d(1, 16, 3x3)` 的参数量是：

```text
16 * 1 * 3 * 3 + 16 = 160
```

如果把 28x28 图像拉平后直接接到 16 个隐藏单元，全连接层参数是：

```text
784 * 16 + 16 = 12560
```

卷积层参数更少，并不是因为它偷懒，而是因为它假设“同一种局部模式可以出现在图像不同位置”。

## 卷积核和局部感受野

卷积层不会一次看完整张图，而是用一个小窗口在图像上滑动。例如 `3x3` 卷积核每次只看一个局部区域。

这叫局部感受野。它适合图像，因为边缘、角点、纹理、局部形状通常都出现在小区域里。

## 参数共享

同一个卷积核会在整张图上重复使用。它在左上角检测边缘，也可以在右下角检测边缘。

这叫参数共享。它带来两个好处：

- 参数更少，不需要每个位置都学一套独立权重。
- 对位置变化更稳健，同一种局部模式出现在不同位置时仍能被识别。

## Padding、Stride 和 Pooling

`padding` 会在图像边缘补像素，让卷积后尺寸不要缩得太快。  
`stride` 控制卷积核每次移动几格。  
`pooling` 会把局部区域压缩成更小的特征图，例如 MaxPool 会保留局部最大响应。

本章的小 CNN 结构是：

```text
[batch, 1, 28, 28]
-> Conv2d(1 -> 16, 3x3, padding=1)
-> ReLU
-> MaxPool2d(2)         # [batch, 16, 14, 14]
-> Conv2d(16 -> 32, 3x3, padding=1)
-> ReLU
-> MaxPool2d(2)         # [batch, 32, 7, 7]
-> Flatten              # [batch, 32 * 7 * 7] = [batch, 1568]
-> Linear(1568 -> 64)
-> ReLU
-> Linear(64 -> 10)
-> logits [batch, 10]
```

其中高宽从 28 变成 14，再变成 7，主要来自 `MaxPool2d(2)`：每次把 `2x2` 局部区域压成一个值，所以 height/width 大约减半。`padding=1` 的 `3x3` 卷积会尽量保持卷积前后的高宽不变。

## 运行实验

首次运行会把 Fashion-MNIST 下载到 `data/vision`。下载只发生一次，后续会直接复用缓存。

快速 smoke test：

```bash
uv run python -m llm_tutor.experiments.train_cnn \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 128 \
  --test-limit 128
```

稍微认真一点的 CPU 实验：

```bash
uv run python -m llm_tutor.experiments.train_cnn \
  --epochs 5 \
  --train-limit 10000 \
  --val-limit 2000 \
  --test-limit 2000
```

样例输出：

```text
epoch=01 train_loss=... val_loss=... val_acc=... val_f1=...
test loss=... acc=... macro_f1=...
```

这里训练日志里的 `val_f1` 是 10 类 macro-F1，不再是前几章二分类里的 malignant F1。最终测试输出会显式写成 `macro_f1`，提醒你它是多分类平均指标。

## 这个实验能说明什么

- 图像 batch 的 shape 和表格数据不同。
- CNN 把“局部模式”和“参数共享”写进模型结构。
- 同一个训练循环可以从表格分类迁移到图像分类。

## 这个实验不能证明什么

- 它不能说明 CNN 是所有视觉任务的最终答案。
- 它不能说明 Fashion-MNIST 高分等于真实视觉系统可靠。
- 它不能说明模型理解了衣服，只能说明它学到了对这个数据集有用的视觉模式。

## 常见失败

- 第一次运行卡在下载：通常是网络慢，不是训练慢。
- shape 报错：检查输入是否是 `[batch, channels, height, width]`。
- 训练很慢：先减少 `train-limit` 和 `epochs`，确认流程通了再扩大。
- F1 很低但 accuracy 还行：可能是某些类别学得很差，后面可以看混淆矩阵。

上一章：[05. 训练神经网络的基本功](05_training_neural_networks.md)  
下一章：[07. RNN：让模型读序列](07_rnn_sequence_modeling.md)

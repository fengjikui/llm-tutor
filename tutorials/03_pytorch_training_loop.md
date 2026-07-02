---
title: "03. PyTorch 快速生存指南"
status: "已完成"
summary: "建立后续所有实验都会复用的 PyTorch 训练循环和 shape 读法。"
---

# 03. PyTorch 快速生存指南

PyTorch 训练脚本的核心结构非常稳定。后面的 CNN、RNN、Transformer、GPT 都会反复出现这几个动作。

## Tensor

Tensor 是 PyTorch 里的基本数据容器。它可以是一维向量、二维矩阵，也可以是更高维的图像或序列 batch。

常见 shape：

```text
表格 batch: [batch_size, num_features]
图像 batch: [batch_size, channels, height, width]
文本 batch: [batch_size, sequence_length]
分类 logits: [batch_size, num_classes]
GPT input_ids: [batch_size, sequence_length]
GPT logits: [batch_size, sequence_length, vocab_size]
GPT labels: [batch_size, sequence_length]
```

理解 shape 是读懂深度学习代码的第一件事。

## Dataset 和 DataLoader

`Dataset` 负责告诉 PyTorch “第 i 个样本是什么”。  
`DataLoader` 负责把样本打包成 batch，并控制 shuffle。

第一阶段代码里，表格数据被包装成：

```python
TensorDataset(features, labels)
```

每次训练循环拿到：

```python
x, y = batch
```

## nn.Module

模型通常继承 `nn.Module`，并实现 `forward`：

```python
class LinearClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.linear(x)
```

当我们写：

```python
logits = model(x)
```

PyTorch 实际会调用 `model.forward(x)`。

## 标准训练循环

核心代码在 `src/llm_tutor/training/loop.py`：

```python
logits = model(x)
loss = loss_fn(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

每一行的作用：

- `model(x)`：前向传播，得到预测。
- `loss_fn(logits, y)`：计算预测和标签之间的差距。
- `optimizer.zero_grad()`：清掉上一轮残留的梯度。
- `loss.backward()`：反向传播，计算新梯度。
- `optimizer.step()`：更新参数。

## 训练模式和评估模式

真实训练代码里还会出现几行看似仪式感很强、但非常重要的语句：

```python
model.train()
model.eval()
with torch.no_grad():
    ...
x = x.to(device)
```

- `model.train()`：告诉模型现在是训练阶段。后面讲 Dropout、BatchNorm 时会看到它的影响。
- `model.eval()`：告诉模型现在是评估阶段，不要使用训练时的随机行为。
- `torch.no_grad()`：评估时不需要记录计算图，可以省内存、加快速度。
- `.to(device)`：把数据和模型放到同一个设备上，例如 CPU 或 GPU。

GPT 生成文本时通常也会用 `model.eval()` 和 `torch.no_grad()`，因为生成阶段只需要前向传播，不需要更新参数。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 5
```

读者应该重点观察：

- train loss 是否整体下降。
- val loss 是否同步下降。
- val accuracy/F1 是否提高。

如果训练 loss 下降但验证指标变差，往往意味着过拟合或数据切分有问题。

## 这个实验能说明什么

- PyTorch 训练循环有稳定套路。
- 训练集和验证集应该分开统计。
- 同一套 loop 后续可以服务图像、序列和语言模型。

## 这个实验不能说明什么

- 它不能说明所有任务都只需要分类指标。
- 它不能说明训练 loop 越抽象越好。教学阶段保留显式步骤更重要。
- 它不能说明 GPU 是必须的。第一阶段所有实验都可以在 CPU 上跑。

上一章：[02. 损失函数、梯度下降和优化器](02_loss_gradient_optimizer.md)  
下一章：[04. 神经网络基础](04_neural_network_basics.md)

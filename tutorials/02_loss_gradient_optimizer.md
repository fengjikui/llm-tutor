---
title: "02. 损失函数、梯度下降和优化器"
status: "已完成"
summary: "从手写线性回归梯度开始，理解 loss、gradient、learning rate 和 autograd。"
---

# 02. 损失函数、梯度下降和优化器

## 本章学习契约

- 新增概念：损失函数、梯度、学习率、参数更新、autograd。
- 实验要验证：手写梯度和 PyTorch 自动求导在同一个简单问题上会给出一致的下降方向。
- 实验不验证：它不覆盖所有 loss，也不说明真实神经网络的梯度都应该手算。
- 跑完重点看：`loss` 是否变小，`w` 是否接近 `3`，`b` 是否接近 `-2`，手写版和 autograd 版趋势是否一致。

模型训练需要三个核心角色：

- 模型：给出预测。
- 损失函数：衡量预测有多错。
- 优化器：根据梯度更新参数。

如果没有损失函数，模型不知道自己错在哪里。如果没有梯度，模型不知道参数应该往哪个方向改。如果没有优化器，梯度就不会变成真正的参数更新。

## 从一条直线开始

我们构造一个 toy 数据集：

```text
y = 3x - 2 + noise
```

模型一开始不知道真实的 `3` 和 `-2`，只知道自己有两个参数：

```text
y_hat = x * w + b
```

训练的目标是让 `w` 接近 `3`，让 `b` 接近 `-2`。

你可能会问：第一章明明是分类，为什么第二章突然换成回归直线？原因很简单：直线回归的梯度可以完整手算，适合把“loss 怎么变成参数更新”这件事看清楚。分类和回归的训练闭环相同：

```text
预测 -> loss -> 梯度 -> 参数更新
```

只是分类常用交叉熵，回归常用 MSE。本章用回归讲梯度，不代表课程主线从分类变成了回归。

## MSE 损失

回归任务常用均方误差：

```text
loss = mean((y_hat - y)^2)
```

预测越接近真实值，loss 越小。

## 从 MSE 到交叉熵

第一章做的是分类任务，不是回归任务。分类里模型输出的是每个类别的 `logits`，也就是还没有归一化的分数：

```text
logits = [malignant_score, benign_score]
```

如果想把 logits 变成概率，可以使用 softmax：

```text
prob_i = exp(logit_i) / sum(exp(logit_j))
```

交叉熵关注的是“真实类别对应的概率有多高”。如果真实类别的概率越高，loss 越小；如果真实类别概率很低，loss 就会很大。

PyTorch 的 `nn.CrossEntropyLoss` 有一个容易误解的点：它直接接收 logits，不需要你先手动 softmax。也就是说，分类训练里常见代码是：

```python
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
```

后面的 GPT 训练也会用交叉熵。区别是：表格分类通常每个样本只有一个标签，而 GPT 会在每个 token 位置预测下一个 token。

一个粗略选择原则是：

| 任务 | 模型输出 | 常见 loss | 例子 |
|---|---|---|---|
| 回归 | 连续数值 | MSE | 预测房价、温度、直线参数 |
| 分类 | 类别 logits | Cross Entropy | 肿瘤良恶性、图像类别、下一个 token |

MSE 惩罚数值距离，交叉熵惩罚“真实类别概率太低”。所以 GPT 虽然是在生成文本，本质上仍然是在每个位置做词表分类。

## 梯度下降在做什么

对 `y_hat = x * w + b` 来说，loss 对 `w` 和 `b` 的梯度可以手动写出来：

```text
dloss/dw = mean(2 * (y_hat - y) * x)
dloss/db = mean(2 * (y_hat - y))
```

参数更新：

```text
w = w - lr * dloss/dw
b = b - lr * dloss/db
```

`lr` 是 learning rate。它决定每一步走多远。

为什么是减去梯度，而不是加上梯度？梯度指向 loss 增长最快的方向。我们想让 loss 变小，就要往相反方向走：

```text
new_param = old_param - learning_rate * gradient
```

如果某个位置梯度为正，说明参数继续增大会让 loss 变大，所以应该把参数减小；如果梯度为负，减去一个负数就会让参数变大。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.manual_gradient_descent
```

脚本会打印两组结果：

- 手动计算梯度的版本。
- PyTorch autograd 的版本。

你应该能看到两者都逐渐接近同一条直线。

样例输出大致会长这样：

```text
manual gradient descent
step=001 w=+0.411 b=-0.199 loss=...
...
step=080 w=+2.987 b=-1.991 loss=...

PyTorch autograd
step=001 w=+0.411 b=-0.199 loss=...
...
step=080 w=+2.987 b=-1.991 loss=...
```

最终参数不会精确等于 `3` 和 `-2`，因为数据里故意加了噪声；学习率和训练步数也会影响最终接近程度。

## 为什么还需要 PyTorch autograd

一条直线的梯度可以手算，但神经网络有成千上万甚至上百亿个参数。手动推导和实现每个参数的梯度不现实。

PyTorch 的 autograd 会记录前向传播的计算图，并在 `loss.backward()` 时自动应用链式法则，把每个参数的梯度算出来。

这不是魔法。它只是把我们刚才手写的梯度计算扩展到了复杂计算图上。

`loss.backward()` 之后，梯度会存到每个可训练参数的 `.grad` 里。比如线性模型里的 `w.grad`、`b.grad` 会从 `None` 或上一轮残留值，变成本轮计算出的梯度张量。随后 `optimizer.step()` 读取这些 `.grad` 来更新参数。

这也解释了为什么训练循环里总要先清梯度：PyTorch 默认会累积梯度。如果不清空，下一轮 batch 的梯度会叠加到上一轮上，参数更新幅度就不再对应当前 batch。

## 这个实验不能证明什么

- 它不能说明所有 loss 都像 MSE 一样容易手算。
- 它不能说明 learning rate 越大越好。太大的学习率可能让 loss 震荡甚至发散。
- 它不能说明 autograd 和手写梯度永远一样；这里只是因为模型足够简单，方便对照。

## 常见失败

- loss 降不下去：先把 learning rate 调小，或者增加训练步数。
- 参数方向错：检查梯度更新是不是 `参数 -= lr * 梯度`。
- 手写版和 autograd 不一致：检查是不是忘了清空梯度，或者记录的是更新前/更新后的不同状态。

上一章：[01. 机器学习到底在学什么](01_machine_learning_basics.md)  
下一章：[03. PyTorch 快速生存指南](03_pytorch_training_loop.md)

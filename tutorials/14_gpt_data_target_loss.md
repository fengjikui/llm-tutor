---
title: "14. GPT 的训练数据、目标数据和损失函数"
status: "已完成"
summary: "用一个显微镜实验拆开 next-token prediction：x/y 错位、logits shape、逐 token cross entropy 和 causal mask。"
---

# 14. GPT 的训练数据、目标数据和损失函数

## 本章学习契约

- 新增概念：next-token target、逐 token cross entropy、label shift、causal mask 检查。
- 实验要验证：GPT 的监督信号来自文本自身，`x` 和 `y` 只差一位，loss 是很多个 token 分类 loss 的平均。
- 实验不验证：它不增加新结构，也不比较不同 tokenizer 或大型语料。
- 跑完重点看：`x -> y` 的错位打印、logits/labels shape、逐 token loss 和 `MiniGPT.forward` 返回 loss 是否一致。

上一章已经把 mini-GPT 跑起来了。这一章不增加新结构，只盯住 GPT 预训练里最容易混淆的三件事：

- 输入数据 `x` 是什么；
- 目标数据 `y` 是什么；
- loss 到底在哪些位置上计算。

## 训练标签来自文本本身

分类任务通常需要人工标签，例如“这张图是鞋子”。GPT 预训练不需要这种标签，它直接从文本中构造监督信号。

给定字符 token：

```text
the model
```

如果 block size 是 8，那么一个训练样本可以是：

```text
x = "the mode"
y = "he model"
```

`y` 就是把 `x` 向右平移一位。模型在每个位置预测下一个 token。

## logits 和 target 的 shape

mini-GPT 的输入是：

```text
input_ids: [batch, seq_len]
```

输出是：

```text
logits: [batch, seq_len, vocab_size]
```

目标是：

```text
target_ids: [batch, seq_len]
```

注意：`logits[b, t]` 是第 `b` 个样本、第 `t` 个位置对整个词表的打分；`target_ids[b, t]` 是这个位置真正应该预测的下一个 token。

## Cross Entropy 是逐位置分类

GPT 的 next-token prediction 可以理解成很多个小分类任务：

```text
位置 0：根据 t 预测 h
位置 1：根据 t h 预测 e
位置 2：根据 t h e 预测 空格
...
```

代码里通常会先展平：

```text
logits:  [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
targets: [batch, seq_len]             -> [batch * seq_len]
```

然后调用 `cross_entropy`。默认情况下，PyTorch 会对所有 token 位置的 loss 取平均。

后面 SFT 会出现 label mask。假设一条 prompt+response 总共 8 个 target，其中 prompt 部分 5 个位置被设成 `-100`：

```text
labels = [-100, -100, -100, -100, -100, 12, 9, 4]
```

`cross_entropy(ignore_index=-100)` 只会平均最后 3 个 response token 的 loss。分母不是 8，而是参与训练的 3 个位置。

## 为什么还要检查 Causal Mask

训练时虽然所有 `x/y` 都已经在 batch 里，但模型不能在位置 `t` 看到位置 `t+1` 的输入。否则它就能偷看答案。

所以第 13 章的 `MiniGPT.forward` 会在内部创建 causal mask。第 14 章的实验会打印：

```text
future_weight_sum=0.0000
```

它表示 attention 权重没有落到未来位置上。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.inspect_gpt_loss
```

样例输出：

```text
text='the model learns'
vocab_size=12 block_size=8
x_ids=[...]
y_ids=[...]
x_text='the mode'
y_text='he model'
logits_shape=(1, 8, 12)
target_shape=(1, 8)
per_token_loss_shape=(1, 8)
per_token_loss=[...]
model_loss=...
manual_mean_loss=...
loss_match=True
future_weight_sum=0.0000
```

这个实验可以帮你确认四件事：

- `x/y` 的错位是按 token 做的；
- logits 的最后一维是词表大小；
- `MiniGPT.forward` 返回的 loss 等于逐 token loss 的平均值；
- causal mask 阻止模型看未来 token。

## 和 SFT 的关系

后面的 SFT 仍然会用 next-token loss。不同点在于，SFT 的文本通常是 prompt + response，并且经常会 mask 掉 prompt 部分的 label，只让 response 部分贡献 loss。

在 PyTorch 里，常见做法是把不参与 loss 的 label 位置设成 `ignore_index`，很多语言模型训练代码会使用 `-100`。这样 `cross_entropy` 会跳过这些位置，只平均剩下的 response token loss。

所以先理解这一章非常重要：SFT 不是换了一种神秘损失函数，而是在同一个语言模型 loss 上改变了数据格式和 label mask。

## 下一步

下一章会把第 13、14 章合起来，开始补 mini-GPT 预训练工程化：checkpoint、validation loss、梯度裁剪和采样参数。

上一章：[13. GPT 从零实现：decoder-only 语言模型](13_gpt_from_scratch.md)

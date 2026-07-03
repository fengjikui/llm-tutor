---
title: "13. GPT 从零实现：decoder-only 语言模型"
status: "已完成"
summary: "把 token embedding、position embedding、causal Transformer block 和 LM head 串成一个可训练的字符级 mini-GPT。"
---

# 13. GPT 从零实现：decoder-only 语言模型

## 本章学习契约

- 新增概念：decoder-only、token embedding、position embedding、causal Transformer block、LM head。
- 实验要验证：一个字符级 mini-GPT 可以用 next-token prediction 跑通完整训练闭环。
- 实验不验证：它不是可用聊天模型，也不会在 tiny corpus 上学到通用语言能力。
- 跑完重点看：输入/目标是否错开一位，logits shape 是否是 `[batch, seq_len, vocab_size]`，生成文本是否至少反映训练语料模式。

前面几章已经准备好了 GPT 的主要零件：

- self-attention；
- causal mask；
- Transformer block；
- 自回归生成的基本思想。

这一章把它们组装成一个最小 GPT。这里的目标不是训练一个真正聪明的聊天模型，而是把 GPT 预训练最核心的闭环跑通：

```text
文本 -> token ids -> 输入 x -> 目标 y -> logits -> cross entropy loss -> 参数更新
```

## GPT 的结构

教育版 mini-GPT 使用 decoder-only 结构。这里的 decoder-only 指的是 GPT 这一路模型：它只堆叠“masked self-attention + MLP”的块，不使用 encoder，也没有 encoder-decoder cross-attention。

原始 Transformer 里的 decoder 是为了翻译任务服务的，它会同时看目标端历史 token 和 encoder 输出。GPT 不做这件事。GPT 只看自己左边已经出现的 token，然后预测下一个 token。

那 prompt 信息从哪里来？答案是：prompt 本身就在同一条序列左边。

```text
prompt:  "the "
生成:    "model"
序列:    t h e 空格 m o d e l
```

当模型生成 `m` 之后，下一步预测 `o` 时，self-attention 可以读取左边的 `the m`。GPT 不需要 encoder 把 prompt 另存一份，因为 prompt token 已经在 causal 序列里。

本章结构是：

```text
token ids
-> token embedding
-> position embedding
-> N 个带 causal mask 的 Transformer block
-> final LayerNorm
-> LM head
-> 每个位置的下一个 token logits
```

输入 shape 是：

```text
[batch, seq_len]
```

输出 logits shape 是：

```text
[batch, seq_len, vocab_size]
```

也就是说，序列中每一个位置都会输出一组词表分数，用来预测“下一个 token 是什么”。

## Token Embedding 和 Position Embedding

`token embedding` 把离散 token id 变成连续向量。字符级模型里，token 可以是一个字符；真实 LLM 里，token 通常来自 BPE、SentencePiece 或类似 tokenizer。

`position embedding` 告诉模型每个 token 在序列中的位置。没有位置信息时，self-attention 本身并不知道 token 的顺序。

本章模型里两者直接相加：

```text
x = token_embedding(input_ids) + position_embedding(positions)
```

position embedding 和 causal mask 都和顺序有关，但负责的事情不同：

| 组件 | 负责什么 | 如果缺失 |
|---|---|---|
| Position embedding | 告诉模型 token 在第几个位置 | 模型很难区分同一批 token 的顺序 |
| Causal mask | 禁止当前位置看未来 token | 训练时可能偷看答案，生成时不一致 |

也就是说，position embedding 提供“位置信息”，causal mask 控制“可见范围”。

## 为什么必须有 Causal Mask

GPT 是从左到右预测下一个 token 的模型。训练时，第 `t` 个位置只能看见 `0..t` 的 token，不能看见未来。

如果允许它看未来，任务会变成作弊：模型可以直接从右边读到答案，训练 loss 会虚假变低，生成时却没有未来 token 可看。

所以 mini-GPT 在 `forward` 内部创建 causal mask，并传给每个 Transformer block。

## 输入数据和目标数据

语言模型训练的标签来自文本本身。给定一段 token：

```text
[t0, t1, t2, t3, t4]
```

输入和目标是错开一位的：

```text
x = [t0, t1, t2, t3]
y = [t1, t2, t3, t4]
```

模型看到 `x` 的每个位置，然后在对应位置预测 `y`。

这就是 GPT 预训练的核心监督信号：不是人工标注类别，而是“下一个 token”。

## 损失函数

mini-GPT 的输出是 logits：

```text
logits: [batch, seq_len, vocab_size]
targets: [batch, seq_len]
```

训练时把前两个维度展平：

```text
logits -> [batch * seq_len, vocab_size]
targets -> [batch * seq_len]
```

然后使用 cross entropy。它会在每个 token 位置计算一次分类损失，再对所有位置取平均。

对应代码入口：

- `CharBlockDataset.__getitem__` 负责把一段 token 切成 `x/y`；
- `MiniGPT.forward` 负责 embedding、causal mask、Transformer blocks、LM head 和 loss；
- `train_mini_gpt._run_epoch` 负责标准训练循环。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_mini_gpt --epochs 20
```

样例输出：

```text
vocab_size=27 block_size=48 parameters=...
epoch=001 train_loss=... val_loss=...
epoch=005 train_loss=... val_loss=...
epoch=010 train_loss=... val_loss=...

generation
the ...
```

因为语料非常小，生成文本不会真的像可用模型。这个实验最有价值的地方是：

- 能看到 loss 随训练下降；
- 能检查 logits、target 和 loss 的 shape；
- 能确认 causal mask 不允许看未来；
- 能跑通从训练到生成的完整路径。

## 这一章和真实 GPT 的差距

本章刻意省略了很多工程细节：

- 没有使用 BPE tokenizer；
- 没有大规模清洗语料；
- 没有 checkpoint、学习率调度、混合精度；
- 没有分布式训练；
- 没有系统的采样策略。

这些会在后续章节逐步补上。先把最小 GPT 理解透，后面再增加训练工程和后训练方法，会轻松很多。

## 下一步

下一章会专门拆解 GPT 的训练数据、目标数据和损失函数。我们会用更小的例子手动检查：

- `x` 和 `y` 如何错位；
- 每个位置的 loss 是怎么来的；
- causal mask 如何阻止未来信息泄露；
- 为什么 SFT 本质上仍然是 next-token loss。

上一章：[12. Transformer Block：把 Self-Attention 组装起来](12_transformer_block.md)

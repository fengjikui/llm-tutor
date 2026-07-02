---
title: "10. Attention：让 decoder 不只看一个向量"
status: "已完成"
summary: "在 Seq2Seq 翻译 toy 实验上加入 additive attention，理解对齐权重、context vector 和 Q/K/V 的前身直觉。"
---

# 10. Attention：让 decoder 不只看一个向量

上一章的 Seq2Seq 有一个明显瓶颈：encoder 读完整个输入句子后，只把最后状态交给 decoder。

```text
source sentence -> encoder final hidden -> decoder
```

如果句子很长，所有信息都挤在一个向量里。decoder 生成每个目标 token 时，都只能依赖这个固定摘要。

Attention 的想法是：decoder 每一步生成时，都可以回看 encoder 的所有位置。

## Attention 在做什么

假设 encoder 读完源句后，留下每个位置的输出：

```text
encoder_outputs [batch, source_len, hidden_size]
```

decoder 在生成第 `t` 个目标 token 时，也有一个当前状态：

```text
decoder_output_t [batch, hidden_size]
```

attention 会用当前 decoder 状态去给每个 source 位置打分，再做 softmax：

```text
attention_weights [batch, target_len, source_len]
```

这些权重表示：生成当前目标 token 时，decoder 更关注源句的哪些位置。

## Context Vector

有了 attention weights，就可以对 encoder outputs 做加权求和：

```text
context = attention_weights @ encoder_outputs
```

这个 `context` 是 decoder 当前步骤从源句里取回来的信息。然后模型把 decoder output 和 context 拼起来，再预测下一个目标 token。

```text
[decoder_output, context] -> Linear -> target logits
```

## Q/K/V 的前身直觉

Transformer 里会正式讲 Query、Key、Value。现在可以先建立直觉：

- decoder 当前状态像 query：它在问“我现在需要源句里的什么信息？”
- encoder outputs 像 key/value：它们既参与匹配，也提供被取回的信息。
- attention weights 是 query 和每个 source 位置匹配后的权重。

更精确地说，本实现里 query/key 会先经过线性投影，value 直接使用 encoder outputs。

本章用的是 Bahdanau 风格的 additive 打分函数。为了让代码更容易读，我们先一次性跑完整个 decoder，再对 decoder outputs 计算 attention；这不是完整复刻 Bahdanau decoder 的所有细节。Transformer 会换成 scaled dot-product attention，但“根据当前需要去取信息”的直觉是一脉相承的。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_attention_seq2seq --epochs 80
```

快速 smoke test：

```bash
uv run python -m llm_tutor.experiments.train_attention_seq2seq --epochs 2
```

样例输出：

```text
epoch=001 train_loss=... val_loss=...
...
attention_shape=(batch, target_len, source_len) logits_shape=(batch, target_len, vocab_size)
src='...' target='...' pred='...'
```

这个 toy 实验仍然不是真实翻译评测。它的重点是让你看到 attention 的张量形状和信息流动。

## 这个实验能说明什么

- Decoder 可以在每个目标位置回看所有 encoder output。
- Attention weights 在 source length 维度上做 softmax。
- Context vector 是对 encoder outputs 的加权求和。
- Attention 是后面 Transformer self-attention 的前置直觉。

## 这个实验不能证明什么

- 它不能证明 attention 一定让这个小 toy 数据显著变好。
- 它不能代表真实机器翻译系统。
- 它还不是 Transformer，因为它仍然使用 RNN encoder/decoder。

## 下一步

本章是 decoder 查询 encoder 的 cross-attention。接下来我们会把 attention 从 RNN 结构里拿出来，专门研究 self-attention：query、key、value 都来自同一个序列，一个序列内部的每个 token 可以互相读取信息。

这一步会通向 Transformer。

上一章：[09. Seq2Seq：从分类到生成](09_seq2seq_translation.md)  
下一章：[11. Self-Attention 从零实现](11_self_attention_from_scratch.md)

---
title: "11. Self-Attention 从零实现"
status: "已完成"
summary: "手写 scaled dot-product attention 和 multi-head self-attention，理解 Q/K/V、padding mask、causal mask 和 shape 变化。"
---

# 11. Self-Attention 从零实现

上一章的 attention 是 cross-attention：

```text
decoder 查询 encoder
```

Self-attention 的变化是：query、key、value 都来自同一个序列。

```text
一个序列内部的 token 互相读取信息
```

这一步非常关键。Transformer 的核心就是 self-attention。

## Q、K、V

给定输入：

```text
x [batch, seq_len, embed_dim]
```

Self-attention 会通过线性层得到三组向量：

```text
Q = x @ W_q
K = x @ W_k
V = x @ W_v
```

直觉上：

- Query：当前 token 想找什么信息。
- Key：每个 token 提供什么索引。
- Value：真正被取回的内容。

先用 Q 和 K 算匹配分数，再用分数加权 V。

## Scaled Dot-Product Attention

核心公式：

```text
scores = Q @ K^T / sqrt(head_dim)
weights = softmax(scores)
output = weights @ V
```

为什么要除以 `sqrt(head_dim)`？  
因为维度越大，点积数值越容易变大，softmax 可能变得过于尖锐，训练不稳定。缩放能让分数范围更温和。

## Multi-Head Attention

一个 attention head 只能从一个表示子空间里看关系。Multi-head attention 会先投影出 Q/K/V，再把每个投影后的表示切成多个 head：

```text
[batch, seq_len, embed_dim]
-> [batch, heads, seq_len, head_dim]
```

每个 head 独立做 attention，最后再拼回：

```text
[batch, heads, seq_len, head_dim]
-> [batch, seq_len, embed_dim]
```

这让模型可以同时关注不同类型的关系。你可以把它理解成：每个 head 在自己的子空间里做一次 attention。

## Padding Mask

batch 里的文本长度可能不同，短文本会被 padding 到同一长度。padding token 不是真实内容，所以不应该被其他 token 关注。

本章的 padding mask shape 是：

```text
[batch, 1, 1, seq_len]
```

它可以 broadcast 到 attention scores 的 shape：

```text
[batch, heads, query_len, key_len]
```

更精确地说，本章的 padding mask 屏蔽的是 padded key，也就是“不让其他 token 读取 padding 位置”。如果某个 query 本身是 padding，它的输出通常会在 loss 里被忽略；后面训练语言模型时也会继续使用 label mask。

## Causal Mask

GPT 这类 decoder-only 语言模型不能看未来 token。训练时第 `t` 个位置只能看 `0..t` 的 token。

Causal mask 是一个下三角矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这就是 GPT 能做 next-token prediction 的关键之一。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.inspect_self_attention
```

样例输出：

```text
x_shape=(2, 5, 16)
token_ids=[[1, 2, 3, 4, 0], [6, 7, 8, 0, 0]]
output_shape=(2, 5, 16)
attention_shape=(2, 4, 5, 5)
combined_mask_shape=(2, 1, 5, 5)
first_head_weights_row0=[1.0, 0.0, 0.0, 0.0, 0.0]
last_real_query_can_attend_to=[1, 1, 1, 1, 0]
```

`attention_shape=(2, 4, 5, 5)` 的含义是：

```text
batch=2
heads=4
query_len=5
key_len=5
```

第一行权重里只有第一个位置可见，是因为 causal mask 让第 0 个 token 不能看未来。`last_real_query_can_attend_to` 展示的是第一条样本最后一个真实 token 能看到哪些 key；最后的 `0` 是 padding key，被 padding mask 屏蔽了。

## 这个实验能说明什么

- Self-attention 的输入输出 shape 可以保持一致。
- Padding mask 会阻止模型关注 padding token。
- Causal mask 会阻止模型看未来。
- Multi-head attention 只是把表示拆成多个 head 并行做 attention。

## 这个实验不能证明什么

- 它还不是完整 Transformer block。
- 它没有训练语言模型，只是在检查机制和 shape。
- 它没有实现高性能 attention kernel，教学清晰优先。

## 下一步

下一章会把 self-attention 放进 Transformer block：加上残差连接、LayerNorm 和 MLP。

再往后，GPT 就是把 decoder-only Transformer block 堆起来，并用 causal mask 训练 next-token prediction。不过完整 GPT 还需要 token embedding、位置信息或 RoPE、残差连接、Norm 和 MLP；causal mask 本身并不足以表达顺序。

上一章：[10. Attention：让 decoder 不只看一个向量](10_attention_seq2seq.md)  
下一章：[12. Transformer Block：把 Self-Attention 组装起来](12_transformer_block.md)

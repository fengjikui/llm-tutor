---
title: "12. Transformer Block：把 Self-Attention 组装起来"
status: "已完成"
summary: "实现一个 Pre-LN Transformer block，理解 residual、LayerNorm、MLP 和 causal self-attention 如何组成 GPT 的基本积木。"
---

# 12. Transformer Block：把 Self-Attention 组装起来

上一章我们只实现了 self-attention。完整 Transformer block 还需要几个关键组件：

- residual connection；
- LayerNorm；
- MLP / feed-forward network；
- attention mask。

这一章实现一个 pre-LN Transformer block。它的结构是：

```text
x
-> LayerNorm
-> Multi-Head Self-Attention
-> residual add
-> LayerNorm
-> MLP
-> residual add
```

输出 shape 和输入保持一致：

```text
[batch, seq_len, embed_dim] -> [batch, seq_len, embed_dim]
```

这就是它可以被堆很多层的原因。

## Residual Connection

Residual connection 会把模块输出加回原输入：

```text
x = x + module(x)
```

这样做有两个好处：

- 信息可以绕过某一层直接传下去；
- 梯度更容易穿过很多层。

深层 Transformer 如果没有 residual，很难稳定训练。

## LayerNorm

LayerNorm 会对每个 token 的隐藏向量做归一化。它让不同层之间的数值尺度更稳定。

本章用的是 pre-LN：

```text
h = x + Attention(LayerNorm(x))
y = h + MLP(LayerNorm(h))
```

现代 decoder-only LLM 常见这种结构，因为它对深层网络训练更友好。

## MLP

Attention 负责 token 之间的信息交换。MLP 负责对每个 token 的表示做非线性变换。

常见结构：

```text
Linear(embed_dim -> 4 * embed_dim)
GELU
Linear(4 * embed_dim -> embed_dim)
```

注意：MLP 不混合不同 token，它是逐 token 工作的。token 间交互主要发生在 self-attention 里。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.inspect_transformer_block
```

样例输出：

```text
x_shape=(2, 5, 16)
output_shape=(2, 5, 16)
attention_shape=(2, 4, 5, 5)
block_shape_preserved=True
future_weight_sum=0.0000
```

`future_weight_sum=0.0000` 表示 causal mask 生效，attention 没有把权重放到未来 token 上。注意，`TransformerBlock` 本身不自动创建 causal mask；本章实验是从外部传入 mask。下一章 mini-GPT 的 `forward` 会负责创建并传入这个 mask。

## 这个实验能说明什么

- Transformer block 的输入输出 shape 保持一致。
- Residual connection 让 block 可以堆叠。
- LayerNorm 稳定每层输入。
- MLP 提供逐 token 的非线性变换。
- Causal self-attention 是 GPT decoder block 的核心组件。

## 这个实验不能证明什么

- 它还不是完整 GPT。
- 它没有 token embedding 和位置信息。
- 它没有训练语言模型，只检查 block 结构和 mask 行为。

## 下一步

下一章会把多个 Transformer block 堆起来，加入 token embedding、position embedding 和语言模型头，开始实现 mini-GPT。

上一章：[11. Self-Attention 从零实现](11_self_attention_from_scratch.md)

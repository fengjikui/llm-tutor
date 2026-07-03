---
title: "21. 现代 LLM 组件：RoPE、GQA、MLA、稀疏注意力与 KV Cache"
status: "已完成"
summary: "补上现代厂商模型里常见的工程组件：旋转位置编码、Grouped-Query Attention、DeepSeek MLA、稀疏注意力、FlashAttention 和推理 KV cache。"
---

# 21. 现代 LLM 组件：RoPE、GQA、MLA、稀疏注意力与 KV Cache

## 本章学习契约

- 新增概念：RoPE、GQA/MQA、MLA、稀疏注意力、FlashAttention、KV Cache、prefill/decode。
- 本章要解决：读者看 Llama、Mistral、Qwen、DeepSeek 这类现代开源模型代码时，知道那些陌生组件在解决什么问题。
- 本章不验证：不做真实训练，也不复刻厂商 CUDA kernel；代码只是能读懂 shape 和数据流的教育版实现。
- 看完重点：哪些组件改变模型结构，哪些组件主要优化推理速度，哪些组件主要优化显存和带宽。

前面第 11-15 章使用的是教育版 GPT：绝对位置编码、标准 multi-head attention、简单 causal mask。真实厂商模型通常不会这么朴素。现代 LLM 的很多改动不是为了“让 Transformer 变得神秘”，而是围绕三个压力点展开：

| 压力点 | 典型组件 | 它想省什么 |
|---|---|---|
| 长上下文 | RoPE、稀疏/滑窗注意力、FlashAttention | 位置泛化、显存、计算 |
| 推理吞吐 | GQA/MQA、KV Cache、paged cache | KV cache 显存和带宽 |
| 超大模型成本 | MLA、MoE、FP8 kernel | 激活参数、缓存、通信和算子效率 |

本章配套代码在 `src/llm_tutor/models/modern_attention.py`。它不是生产实现，但尽量让每个概念都能落到一个可读函数。

## RoPE：位置不是加上去，而是旋转进去

教育版 GPT 常用 learned absolute position embedding：

```python
x = token_embedding(input_ids) + position_embedding(positions)
```

RoPE 的思路不同。它不把位置向量加到 hidden state 上，而是在 attention 里旋转 query 和 key。直觉上，每个二维小平面按当前位置转一个角度；两个 token 做点积时，点积里自然带上了相对距离的信息。

简化实现：

```python
from llm_tutor.models.modern_attention import build_rope_cache, apply_rope

cos, sin = build_rope_cache(seq_len=128, head_dim=64)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

RoPE 常出现在现代 decoder-only LLM 中，因为它和 causal attention 很契合，也比固定绝对位置更适合外推到较长上下文。不过 RoPE 不是“无限长上下文开关”。长上下文能力还会受训练长度、数据、attention 实现、cache 策略和位置缩放方法影响。

## GQA：Query 头很多，Key/Value 头少一点

标准 MHA 里，query、key、value 的 head 数相同：

```text
q_heads = k_heads = v_heads
```

推理时，生成每个新 token 都要读取历史 token 的 key/value。上下文越长，KV cache 越大，显存带宽越紧张。GQA 的想法是：保留较多 query heads，但让多个 query heads 共享同一组 key/value heads。

```text
MHA: q=32, kv=32
GQA: q=32, kv=8
MQA: q=32, kv=1
```

代码里最关键的一步是把 KV head repeat 到 query head 数量：

```python
from llm_tutor.models.modern_attention import repeat_kv

k = repeat_kv(k, num_query_heads=32)
v = repeat_kv(v, num_query_heads=32)
```

GQA 介于 MHA 和 MQA 之间。它通常能明显减少 KV cache，同时比只用一个 KV head 的 MQA 更稳。Llama 3 和 Mistral 7B 都采用了 GQA 方向的设计；Mistral 还结合了 sliding window attention 来降低长上下文推理成本。

## KV Cache：推理时不要重复算过去

训练时，一整段 token 并行进入模型；推理生成时，模型一次只多生成一个 token。如果每一步都把完整前缀重新计算一遍，会浪费大量计算。

KV Cache 保存每层 attention 已经算过的 key/value：

```text
prefill:  prompt tokens -> 计算所有层 K/V -> 存入 cache
decode:   新 token -> 只计算新 token 的 K/V -> append 到 cache -> 读取全部历史 K/V
```

简化实现：

```python
from llm_tutor.models.modern_attention import KVCache

cache = KVCache()
past_k, past_v = cache.append(prompt_k, prompt_v)
past_k, past_v = cache.append(new_token_k, new_token_v)
```

这就是为什么长对话会吃显存：模型参数占一部分，KV cache 也会占一大部分。对于 batch serving，工程系统还会做 paged KV cache、prefix cache、cache eviction、quantized cache、cache offload 等优化。

## MLA：把 KV Cache 压到 latent 空间

DeepSeek-V2/V3 的 MLA 可以粗略理解成：不要直接缓存完整 K/V，而是先把 token 表示压成一个更小的 latent，再在需要时展开成 key/value。它的目标不是简单“少几个 head”，而是通过低秩联合压缩减少 KV cache 和内存带宽压力。

教育版示意：

```python
from llm_tutor.models.modern_attention import LatentKVProjection

projector = LatentKVProjection(embed_dim=4096, latent_dim=512, num_kv_heads=8, head_dim=128)
latent_cache, k, v = projector(hidden_states)
```

真实 MLA 比这个复杂得多。DeepSeek 代码里还会处理 RoPE 维度、non-RoPE 维度、低秩 query 压缩、权重吸收、FP8 cache 和专门 kernel。本章只希望你抓住一句话：MLA 重点在减少推理阶段 KV cache 和带宽，不只是普通 GQA 的另一种写法。

## 稀疏注意力：不是每个 token 都看所有 token

完整 causal attention 的注意力矩阵是 `seq_len x seq_len`。上下文变长以后，计算和显存压力会快速上升。稀疏注意力把“看谁”改成一种模式：

| 模式 | 代表 | 直觉 |
|---|---|---|
| 滑动窗口 | Longformer、Mistral SWA | 每个 token 只看附近一段历史 |
| 局部 + 全局 | Longformer、BigBird | 大多数 token 看局部，少数全局 token 看全局 |
| 随机/块稀疏 | BigBird、Sparse Transformer | 用结构化稀疏近似全局连通 |
| 动态稀疏 | DeepSeek Sparse Attention 等 | 根据 token 或块选择更重要的上下文 |

滑窗 causal mask 的最小实现：

```python
from llm_tutor.models.modern_attention import sliding_window_causal_mask

mask = sliding_window_causal_mask(seq_len=8192, window_size=4096)
```

稀疏注意力的关键取舍是：速度和长度上去了，但模型不再能在每一层直接看见所有历史 token。工程上常常用多层堆叠、全局 token、检索、压缩记忆或动态选择来补。

## FlashAttention：不改变数学结果，改变搬数据方式

FlashAttention 很容易和稀疏注意力混在一起。它们不是一回事：

- FlashAttention：仍然计算精确 attention，主要通过 IO-aware tiling 减少 HBM 和 SRAM 之间的数据搬运。
- 稀疏注意力：改变 attention 可见模式，不再计算完整矩阵。

DeepSeek 的 FlashMLA 则是另一层：它是围绕 MLA 推理 workload 写的高性能 kernel，服务 DeepSeek-V3/V3.2 这类模型。你可以把它理解为“模型结构是 MLA，生产推理还需要专门 kernel 把它跑快”。

## 现代 Decoder Block 的心理地图

今天看一个现代 LLM block，可以按这个顺序定位：

```text
input hidden states
-> RMSNorm / LayerNorm
-> q/k/v projection
-> RoPE on q/k
-> MHA / GQA / MLA attention
-> KV cache read/write if inference
-> residual
-> RMSNorm / LayerNorm
-> MLP / SwiGLU / MoE
-> residual
```

不同厂商的差异通常在这些位置出现：

- 位置编码：RoPE、scaled RoPE、multimodal RoPE。
- attention：MHA、GQA、MQA、MLA、sliding window、sparse attention。
- FFN：普通 MLP、SwiGLU、MoE。
- 推理系统：KV cache、paged attention、FlashAttention、FlashMLA、FP8/INT8 cache。

所以读源码时不要先背名词。先问四个问题：

1. Q/K/V 的 head 数是否相同？
2. 位置编码是加到 embedding，还是作用在 Q/K？
3. 推理时 cache 保存的是完整 K/V，还是压缩后的 latent？
4. attention 是完整可见、滑窗可见，还是动态稀疏可见？

## 参考

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Meta Llama 3 announcement](https://ai.meta.com/blog/meta-llama-3/)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [BigBird](https://arxiv.org/abs/2007.14062)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Hugging Face KV cache explanation](https://huggingface.co/docs/transformers/en/cache_explanation)

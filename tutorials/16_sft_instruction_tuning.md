---
title: "16. SFT：让模型学会按指令回答"
status: "已完成"
summary: "用 tiny instruction 数据演示 SFT：prompt/response 模板、response-only label mask，以及为什么 SFT 仍然是 next-token loss。"
---

# 16. SFT：让模型学会按指令回答

SFT 是 supervised fine-tuning，通常翻译成监督微调。它的目标不是改变 GPT 的基本结构，而是让模型看到“指令 -> 回答”的样例，从而学会按这种格式输出。

这一章的重点是一个容易被讲玄的事实：

```text
SFT 仍然是 next-token prediction。
```

区别主要在数据格式和 label mask。

## Instruction / Response 格式

本章使用最小 instruction 模板：

```text
Instruction: repeat: hi
Response: hi
```

真实 ChatGPT 风格模型会有更复杂的 chat template，例如 system/user/assistant 多轮消息。本章不用多轮格式，是为了把 mask 和 loss 看清楚；本质上，它们最后仍然会被 tokenizer 转成一条 token 序列。

## 为什么要 Mask Prompt

如果把整条序列都作为 loss，模型会同时学习：

- 复述 prompt；
- 生成 response。

但 SFT 更关心“看到 prompt 后生成 response”。因此常见做法是把 prompt 部分的 label 设成 `ignore_index`，只让 response token 贡献 loss。

在 PyTorch 里，本章使用：

```python
IGNORE_INDEX = -100
F.cross_entropy(..., ignore_index=IGNORE_INDEX)
```

例如：

| 位置含义 | input token | target token | 计入 loss |
|---|---|---|---|
| prompt 内部 | `Instruction...` | prompt 的下一个 token | 否，label 是 `-100` |
| prompt 末尾 | `Response: ` 的最后一个 token | response 第一个 token，比如 `h` | 是 |
| response 内部 | `h` | `i` | 是 |
| response 结尾 | `i` | 换行符 | 是 |

prompt token 仍然会作为输入上下文被模型看到，但它们对应位置不计入 loss。

## 和预训练的关系

预训练数据通常是普通文本：

```text
the model reads tokens one by one
```

SFT 数据通常是结构化对话或指令：

```text
Instruction: ...
Response: ...
```

两者的模型结构和 loss 类型一样，都是 causal LM cross entropy。SFT 只是把监督信号集中到回答部分。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_sft --epochs 30
```

smoke 版会更短：

```bash
uv run python -m llm_tutor.experiments.train_sft \
  --epochs 2 \
  --batch-size 3 \
  --embed-dim 32 \
  --num-layers 1
```

样例输出：

```text
vocab_size=23 examples=6
epoch=001 sft_loss=...
epoch=002 sft_loss=...

generation
Instruction: repeat: hi
Response: ...
```

因为数据极小，生成质量不是重点。重点是：

- `SFTDataset` 如何格式化 prompt/response；
- prompt 部分 label 如何变成 `-100`；
- `sft_cross_entropy` 如何跳过这些位置；
- 训练循环仍然是 forward、loss、backward、optimizer step。

## 常见误解

SFT 不是 RLHF。SFT 没有 reward model，也没有 PPO/DPO/GRPO 的偏好优化目标。

SFT 也不是“教模型事实”的万能方式。它更常用于教模型遵循格式、语气、任务分布和回答习惯。

## 下一步

下一章进入 PPO。我们会从 policy、reward、KL penalty 和 clipping 开始，做一个最小可运行的 RLHF 优化器实验。

上一章：[15. 小型 GPT 预训练工程化](15_mini_gpt_pretraining.md)

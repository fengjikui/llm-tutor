---
title: "09. Seq2Seq：从分类到生成"
status: "已完成"
summary: "用一个不带 attention 的 Encoder-Decoder 翻译 toy 实验理解 BOS/EOS、teacher forcing、自回归解码和序列损失。"
---

# 09. Seq2Seq：从分类到生成

## 本章学习契约

- 新增概念：encoder-decoder、BOS/EOS、teacher forcing、自回归解码、序列级 cross entropy。
- 实验要验证：模型不再只输出一个类别，而是可以一步步生成一个目标序列。
- 实验不验证：它不是正式机器翻译评测，也不能代表真实语料上的翻译能力。
- 跑完重点看：source/target/prediction 三者是否对齐，EOS 是否正常结束，loss 是否按目标 token 计算。

前几章的序列模型做的是分类：

```text
输入序列 -> 一个类别
```

机器翻译不一样。它需要：

```text
输入序列 -> 输出序列
```

例如：

```text
"you are smart" -> "tu es intelligent"
```

这类任务叫 sequence-to-sequence，简称 Seq2Seq。

## Encoder-Decoder

最经典的 Seq2Seq 结构分成两部分：

- Encoder：读完整个输入序列，把它压成一个状态。
- Decoder：从这个状态出发，一步步生成输出序列。

本章先实现一个不带 attention 的 GRU Encoder-Decoder。它的瓶颈很明显：encoder 必须把整句信息压进最后一个 hidden state。这个瓶颈是下一章 attention 的概念动机。

实验使用的是人工构造、ASCII 化的 toy 句子。为了避免分词、重音和真实语料清洗把主线冲散，法语里的重音会被去掉，例如 `drole`、`prets`。它适合观察数据流，不代表真实翻译数据集。

## BOS、EOS 和 padding

生成任务需要告诉 decoder 什么时候开始、什么时候结束：

- `BOS`：beginning of sequence，输出序列开始。
- `EOS`：end of sequence，输出序列结束。
- `PAD`：把不同长度的句子补齐成同一个 batch。

目标句子 `tu es intelligent` 会被拆成两条序列：

```text
target_input:  <bos> tu es intelligent
target_output: tu es intelligent <eos>
```

训练时，decoder 看到 `target_input`，每个位置都要预测 `target_output` 的对应 token。

把它展开成表格：

| decoder 当前看到 | 需要预测 |
|---|---|
| `<bos>` | `tu` |
| `tu` | `es` |
| `es` | `intelligent` |
| `intelligent` | `<eos>` |

这就是生成任务里的错位监督。第 13、14 章 GPT 也会做类似事情，只是 GPT 没有单独的 encoder，输入文本自己同时提供上下文和下一个 token 标签。

这和 GPT 的 next-token prediction 已经很接近了。区别是：Seq2Seq decoder 还会接收 encoder 给出的源句信息。

## Teacher Forcing

训练时我们把正确的前一个 token 喂给 decoder，这叫 teacher forcing。

例如预测第二个法语 token 时，decoder 看到的是正确的 `tu`，而不是自己上一步预测出来的 token。

这样训练更稳定。但推理时没有标准答案可喂，decoder 只能把自己上一步生成的 token 接着喂回去，这叫自回归解码。

本章的 greedy decode 为了清晰，每生成一步都会重跑 encoder 和当前 decoder prefix。真实系统会缓存 encoder 状态和 decoder 状态，但那会让第一版代码更难读。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_seq2seq_translation --epochs 80
```

快速 smoke test：

```bash
uv run python -m llm_tutor.experiments.train_seq2seq_translation --epochs 2
```

样例输出：

```text
epoch=001 train_loss=... val_loss=...
...
translations
src='you are smart' target='tu es intelligent' pred='...'
```

跑完时不要只看 `pred` 是否完全等于 `target`。还要看两件事：它有没有生成 `<eos>` 后停止，常见词是否出现在合理位置。toy 数据很小，完全正确和完全错乱之间会有很多中间状态。

## 怎么看结果

你可能会看到训练 loss 持续下降，但验证 loss 不一定下降，测试翻译也可能不准确。这不是坏事，它暴露了本章 toy 设置的多个限制：

- 数据很小，容易记住训练样本。
- 验证/测试只是见过词的重组，不是严格真实翻译评估。
- 不带 attention 的 encoder 必须把整句压成一个向量，这是 attention 的概念动机，不是本 toy 实验单独证明出的结论。
- decoder 推理时只能依赖自己前面生成的 token，错误会累积。

下一章加入 attention 后，decoder 不必只依赖一个固定向量，而是可以在生成每个 token 时回看 encoder 的不同位置。

## 这个实验能说明什么

- Seq2Seq 把“一个输入对应一个标签”扩展成“一个输入序列对应一个输出序列”。
- BOS/EOS 让生成过程有明确开始和结束。
- Teacher forcing 让训练阶段更稳定。
- 序列损失本质上还是 cross entropy，只是每个目标 token 都要算一次。

## 这个实验不能证明什么

- 它不能证明这个 toy 模型有真实翻译能力。
- 它不能证明训练 loss 低就代表生成质量好。
- 它不能解决长句翻译和复杂对齐问题。

## 和 Attention 的关系

本章 decoder 只拿到 encoder 最后的 hidden state。所有源句信息都挤在一个向量里。

Attention 的想法是：decoder 每生成一个 token，都可以根据当前需要去看 encoder 的所有输出位置。这样模型不必把整句一次性压进一个固定向量。

上一章：[08. LSTM 和 GRU：门控记忆](08_lstm_gru.md)  
下一章：[10. Attention：让 decoder 不只看一个向量](10_attention_seq2seq.md)

---
title: "15. 小型 GPT 预训练工程化"
status: "已完成"
summary: "在 mini-GPT 上加入 checkpoint、validation loss、梯度裁剪和 temperature/top-k/top-p 采样，让实验从能跑变成可复现。"
---

# 15. 小型 GPT 预训练工程化

## 本章学习契约

- 新增概念：validation loss、checkpoint、梯度裁剪、temperature/top-k/top-p 采样。
- 实验要验证：mini-GPT 的训练结果可以保存、加载、生成，并通过采样参数观察输出变化。
- 实验不验证：它不是大模型预训练工程，也不覆盖分布式训练、混合精度或数据管线。
- 跑完重点看：train/val loss、checkpoint 路径、重新加载后能否生成，以及固定 prompt 下采样参数改变带来的差异。

第 13、14 章已经讲清楚了 GPT 的结构和 loss。第 15 章开始补一点训练工程。

这里的“工程化”不是大规模分布式训练，而是先把最小实验变得可复现、可保存、可加载、可比较：

- 训练时记录 train loss 和 validation loss；
- 反向传播后做 gradient clipping；
- 保存带训练配置和数据元信息的 checkpoint；
- 单独写生成脚本；
- 比较 temperature、top-k、top-p 对输出的影响。

## Validation Loss

训练 loss 只能说明模型在训练样本上的拟合情况。validation loss 更接近“没见过的数据上预测下一个 token 的能力”。

上一章我们已经修正了语言模型数据切分：先按连续文本范围切 train/val，再各自生成滑动窗口，中间留出 `block_size` 的 gap，避免几乎相同的窗口同时出现在训练集和验证集。

## Gradient Clipping

语言模型训练中，梯度偶尔会变得很大。第 15 章训练脚本在 `loss.backward()` 后加入：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它不会让模型变强，但能让小实验更稳定。

## Checkpoint

checkpoint 至少需要保存：

- 模型参数 `model_state`；
- 优化器状态 `optimizer_state`；
- 模型结构配置 `config`；
- 字符词表 `vocab_id_to_token`；
- 训练参数 `training_args`；
- 数据切分信息 `data_meta`；
- 当前 epoch；
- train/validation loss。

训练并保存：

```bash
uv run python -m llm_tutor.experiments.train_mini_gpt \
  --epochs 20 \
  --checkpoint-path checkpoints/mini_gpt.pt
```

加载并生成：

```bash
uv run python -m llm_tutor.experiments.generate_with_mini_gpt \
  --checkpoint-path checkpoints/mini_gpt.pt \
  --prompt "the " \
  --max-new-tokens 120
```

## Temperature

temperature 控制采样分布的尖锐程度：

- `temperature < 1`：更保守，更偏向高概率 token；
- `temperature = 1`：使用原始 logits；
- `temperature > 1`：更随机，更容易发散。

用三个候选 token 做个直觉例子：

```text
原始概率:  A=0.70, B=0.20, C=0.10
低温采样:  A 更接近 1，输出更保守
高温采样:  B/C 的机会变大，输出更多样也更容易乱
```

temperature 不会创造新知识，它只改变“从当前模型认为可能的 token 里怎么抽样”。

## Top-k

top-k 每一步通常只保留概率最高的 `k` 个 token，再从这些 token 里采样。本项目教学实现按阈值过滤，如果第 `k` 名附近出现完全相同的 logits，并列 token 可能会被一起保留。

```bash
uv run python -m llm_tutor.experiments.generate_with_mini_gpt \
  --checkpoint-path checkpoints/mini_gpt.pt \
  --top-k 5
```

top-k 越小，输出越保守。

## Top-p

top-p 又叫 nucleus sampling。它不是固定保留 `k` 个 token，而是从高到低累计概率，保留累计概率达到 `p` 的最小 token 集合。

```bash
uv run python -m llm_tutor.experiments.generate_with_mini_gpt \
  --checkpoint-path checkpoints/mini_gpt.pt \
  --top-p 0.9
```

真实 LLM 生成常常会组合使用 temperature 和 top-p。

本项目的组合顺序是：先用 temperature 缩放 logits，再做 top-k 过滤，再在过滤后的分布上做 top-p。

## 运行 smoke 版实验

为了让 smoke test 很快完成，课程里使用较小模型和很少 epoch：

```bash
uv run python -m llm_tutor.experiments.train_mini_gpt \
  --epochs 2 \
  --batch-size 16 \
  --block-size 32 \
  --embed-dim 32 \
  --num-layers 1
```

这个输出不会像自然语言，但它能验证训练、保存、加载和生成这条路径。

默认脚本会优先使用 CUDA；没有 GPU 时会自动落到 CPU。当前课程的默认 smoke 参数按 CPU 可跑设计，4090 这类显卡只有在你想扩大语料、模型尺寸、训练轮数或做更真实的后训练实验时才有必要。

## 下一步

下一章进入 SFT。你会看到：SFT 不是把 GPT 换成另一个模型，而是把数据组织成 instruction/response 格式，并对 label 做 mask。

上一章：[14. GPT 的训练数据、目标数据和损失函数](14_gpt_data_target_loss.md)

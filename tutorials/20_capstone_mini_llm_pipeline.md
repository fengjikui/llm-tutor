---
title: "20. Capstone：Mini LLM Pipeline"
status: "已完成"
summary: "把 mini-GPT 预训练、loss 检查、SFT、PPO、DPO、GRPO 串成一条可运行的学习流水线。"
---

# 20. Capstone：Mini LLM Pipeline

## 本章学习契约

- 新增概念：把预训练、生成检查、loss 显微镜、SFT、PPO、DPO、GRPO 放到同一张学习地图里。
- 实验要验证：这些章节虽然代码抽象层级不同，但都在重复“数据 -> 目标 -> loss/奖励 -> 更新 -> 检查”的训练闭环。
- 实验不验证：它不是同一个大模型 checkpoint 从预训练连续训练到 post-training 的生产流水线。
- 跑完重点看：每个阶段的命令是否可执行，哪些阶段共享 mini-GPT checkpoint，哪些阶段只是教学版 bandit policy。

这一章不引入新算法，而是把前面几个阶段串起来。

完整大模型训练流水线可能非常复杂，但它的学习骨架可以先压缩成：

```text
预训练 -> 生成检查 -> loss 检查 -> SFT -> PPO -> DPO -> GRPO
```

本项目的 Capstone 用 tiny 脚本做这件事。它不是生产训练框架，而是一张可以运行的路线图。

这里的“串起来”指的是统一训练目标视角的学习 pipeline，不是同一个 checkpoint 从预训练一路继续进入 SFT、PPO、DPO、GRPO 的生产流水线。当前代码里只有 `pretrain` 和 `generate` 共享 mini-GPT checkpoint；SFT 会重新初始化一个 tiny mini-GPT；PPO、DPO、GRPO 使用独立的 bandit policy 来讲清各自的优化目标。

真实项目里，PPO、DPO、GRPO 经常是不同对齐路线或不同任务阶段的选择，不一定全部依次执行一遍。本章把它们放在一起，是为了让你比较“训练信号如何变化”，不是暗示生产训练必须按这个顺序全做。

## 预览 Pipeline

默认 dry-run，只打印每个阶段要执行的命令：

```bash
uv run python -m llm_tutor.experiments.run_capstone_pipeline
```

你会看到：

```text
capstone_mode=dry-run stages=7

[01] pretrain
goal=训练一个 tiny causal LM，并保存 checkpoint。
command=...
```

## 执行短版 Pipeline

真正执行：

```bash
uv run python -m llm_tutor.experiments.run_capstone_pipeline --execute
```

脚本会用临时目录保存 mini-GPT checkpoint，执行完自动清理。

如果你希望保留 checkpoint，可以显式传入路径：

```bash
uv run python -m llm_tutor.experiments.run_capstone_pipeline \
  --execute \
  --checkpoint-path checkpoints/capstone_mini_gpt.pt
```

不传 `--checkpoint-path` 时，`--execute` 使用临时路径并自动清理；传入后会写入你指定的位置。

## 每个阶段在检查什么

| 阶段 | 数据 | 目标 | loss / 优化信号 | 教学实现 |
|---|---|---|---|---|
| pretrain | 普通文本 | 下一个 token | causal LM cross entropy | mini-GPT checkpoint |
| generate | prompt | 采样后续 token | temperature / top-k / top-p | 读取 mini-GPT checkpoint |
| inspect-loss | 短文本 | 手动检查 `x/y` | 逐 token cross entropy | 独立 loss 显微镜 |
| SFT | instruction/response | response token | masked next-token loss | 独立 tiny SFT |
| PPO | prompt/action/reward | 提高高 reward action | clipped policy objective | bandit policy |
| DPO | chosen/rejected | 偏向 chosen | preference log-ratio loss | preference bandit |
| GRPO | group samples/reward | 组内相对优势 | relative advantage policy loss | group bandit |

可以把这张表再压缩成一条主线：

```text
预训练/SFT：答案来自文本 label
PPO/GRPO：答案好坏来自采样后的 reward
DPO：答案好坏来自 chosen/rejected 偏好对
```

所有方法都在更新模型参数，但“什么算好答案”的定义不同。

## 这条流水线不能代表什么

它不是生产级 LLM 训练系统：

- 没有大语料清洗；
- 没有 BPE tokenizer；
- 没有分布式训练；
- 没有大模型 checkpoint 管理；
- 没有真实 reward model；
- 没有系统评测集。
- 没有同一模型的连续 post-training；
- 没有真实偏好数据管线；
- 没有 value model；
- 没有评测回归门禁。

它真正想证明的是：LLM 的几个训练阶段可以用统一语言理解。

## 回到主线

从第 1 章到第 20 章，我们一直在重复同一个基本动作：

```text
准备数据 -> 前向传播 -> 构造训练目标 -> 计算 loss -> 反向传播 -> 更新参数 -> 检查结果
```

差别在于训练目标越来越复杂：

- 分类任务：预测标签；
- 语言模型：预测下一个 token；
- SFT：只让 response token 贡献 loss；
- PPO：用 reward 和 clipping 更新 policy；
- DPO：直接利用 chosen/rejected 偏好；
- GRPO：用组内相对 reward 代替 value model。

这就是本教程的最终地基。

上一章：[19. GRPO：面向可验证任务的组内相对优化](19_grpo_verifiable_tasks.md)

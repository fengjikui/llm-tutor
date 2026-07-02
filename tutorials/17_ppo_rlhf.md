---
title: "17. PPO：经典 RLHF 的核心优化器"
status: "已完成"
summary: "用一个 tiny bandit 实验解释 PPO 的 policy、old policy、advantage、ratio、clipping、entropy 和 KL penalty。"
---

# 17. PPO：经典 RLHF 的核心优化器

PPO 是 RLHF 里最经典的一类优化器。真实 RLHF 会让语言模型生成回答，再用 reward model 打分，并用 PPO 更新 policy。

这一章先不直接训练 mini-GPT 生成长文本，而是用一个 tiny bandit 实验把 PPO 公式讲清楚：

```text
prompt -> policy 选择 action -> rule reward 打分 -> PPO 更新 policy
```

这样可以先看懂核心优化目标，再把它迁移到语言模型。

这个实验刻意省略了完整 RLHF 里的很多组件：没有 token-level generation、没有 reward model、没有 value model/GAE、没有 rollout buffer，也没有多轮 prompt-response。它只负责讲清楚 PPO 更新本身。

## Policy、Action 和 Reward

本章实验里有三个 prompt：

```text
say yes
say no
say ok
```

policy 要在三个 action 里选一个：

```text
yes / no / ok
```

规则 reward 很简单：

- action 等于目标，reward = `1.0`；
- action 不等于目标，reward = `-0.2`。

真实 RLHF 里的 action 是一整段生成文本，reward 通常来自 reward model 或规则评估器。

## Old Policy 和 Ratio

PPO 更新前会先用当前 policy 采样一批 action，并记录当时的 log probability：

```text
old_log_prob = log pi_old(action | prompt)
```

更新时，再用新的 policy 计算：

```text
new_log_prob = log pi_new(action | prompt)
```

两者相减再取指数：

```text
ratio = exp(new_log_prob - old_log_prob)
```

如果 ratio 很大，说明新 policy 比旧 policy 更想选择这个 action；如果 ratio 很小，说明新 policy 想远离它。

## Advantage

advantage 表示“这个 action 比预期好多少”。完整 PPO 常常会用 value model 估计 baseline。

本章为了保持实验透明，直接用 rule reward 作为 advantage：

```text
advantage = reward
```

这样读者可以先看懂 clipping，后面再补 value/GAE 等更完整的 RL 细节。

## Clipped Objective

PPO 的核心是限制策略更新不要太猛：

```text
unclipped = ratio * advantage
clipped = clamp(ratio, 1 - eps, 1 + eps) * advantage
loss = -mean(min(unclipped, clipped))
```

PPO 论文常说“最大化 objective”。代码里写成负号，是因为 PyTorch 优化器默认最小化 loss。

直觉上：

- 如果一个 action 很好，PPO 鼓励提高它的概率；
- 但概率不能一下子提高太多；
- 如果一个 action 很差，PPO 鼓励降低它的概率；
- 但也不希望一步把策略推得太远。

## KL Penalty 和 Entropy

真实 RLHF 通常还会让 policy 不要偏离 reference model 太远。这个约束常写成 KL penalty：

```text
loss = ppo_policy_loss + beta * KL(policy || reference)
```

本章实验里 reference 是一个固定的均匀策略。

真实语言模型训练通常按生成 token 估计 KL penalty。本章为了教学，直接计算 categorical policy 的完整 `KL(policy || reference)`。

entropy 则鼓励策略保持一定探索性：

```text
loss = loss - entropy_coef * entropy
```

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_ppo_bandit --epochs 30
```

smoke 版：

```bash
uv run python -m llm_tutor.experiments.train_ppo_bandit --epochs 3
```

样例输出：

```text
prompts=3 actions=3
epoch=001 sampled_reward=... greedy_reward=... policy_loss=... ratio_min=... ratio_max=... clipped_fraction=... kl=... entropy=...

policy
prompt='say yes' action='yes' target='yes'
...
```

这个实验的目标不是得到一个强模型，而是确认：

- old log prob 和 new log prob 如何形成 ratio；
- clipping 如何进入 loss；
- 同一批 rollout 多轮 PPO 更新后，ratio 如何偏离 1；
- KL penalty 如何约束 policy；
- PPO 本质上是在优化“被 reward 引导的采样行为”。

## 和 SFT 的区别

SFT 直接模仿标准答案：

```text
prompt + response -> next-token loss
```

PPO 则先让 policy 采样 action，再根据 reward 强化或削弱这个 action：

```text
sample -> reward -> policy update
```

这就是 RLHF 相比 SFT 多出来的关键一步。

## 下一步

下一章会讲 DPO。DPO 不需要 PPO 这种在线采样循环，而是直接用 chosen/rejected 偏好对构造 loss。

上一章：[16. SFT：让模型学会按指令回答](16_sft_instruction_tuning.md)

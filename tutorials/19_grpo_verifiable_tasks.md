---
title: "19. GRPO：面向可验证任务的组内相对优化"
status: "已完成"
summary: "用 group sampling 和 rule-based reward 实现 GRPO 的最小实验，理解组内 relative advantage 为什么适合可验证任务。"
---

# 19. GRPO：面向可验证任务的组内相对优化

GRPO 可以理解成一种更适合可验证任务的 PPO 变体。它常用于数学、代码、Countdown 这类任务：答案可以用规则打分，不一定需要训练 reward model。

这一章继续使用 tiny bandit 实验，但会加入 GRPO 的关键动作：

```text
同一个 prompt 采样多条 action -> 规则 reward -> 组内相对 advantage -> 更新 policy
```

## 为什么要 Group Sampling

PPO 通常需要 value model 来估计 baseline。GRPO 的思路是：对同一个 prompt 采样一组回答，用这组回答的 reward 均值和方差构造相对优势。

给定同一个 prompt 的一组 reward：

```text
[0, 1, 1, 0]
```

组内相对 advantage 是：

```text
advantage = (reward - group_mean) / group_std
```

这样就不需要单独训练 value model。

## 可验证 Reward

本章使用三个可验证 prompt：

```text
2+2?
capital of france?
first letter of abc?
```

动作空间是：

```text
4 / paris / a
```

如果 action 等于目标答案，reward = `1`；否则 reward = `0`。

真实 GRPO 可以把这个规则替换成：

- 数学答案是否正确；
- 代码是否通过测试；
- JSON 格式是否可解析；
- 工具调用结果是否满足条件。

## GRPO Loss

本章使用一个教学版 loss：

```text
loss = -mean(log_prob(action) * group_relative_advantage)
       + kl_coef * KL(policy || reference)
```

本章的 bandit 实验可以直接计算完整 categorical KL，所以输出里的 `mean_kl` 是非负的完整分布 KL。真实语言模型通常不会枚举整个生成分布，工程里会用 token 级近似来约束 policy 不要偏离 reference 太远。

## 和 PPO / DPO 的区别

PPO：

- 在线采样；
- reward 打分；
- 常用 value model / advantage；
- clipped objective。

DPO：

- 离线 chosen/rejected 偏好对；
- 不需要在线采样；
- 不需要显式 reward model。

GRPO：

- 在线采样多条；
- 使用规则 reward；
- 不用 value model；
- 用组内相对 reward 构造 advantage。

真实 GRPO 系统里往往仍会保留 PPO-style ratio/clipping 或类似 trust-region 约束。本章教学版主要保留 group sampling、rule reward 和 relative advantage 这条主线。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_grpo_bandit --epochs 40 --group-size 4
```

smoke 版：

```bash
uv run python -m llm_tutor.experiments.train_grpo_bandit --epochs 3 --group-size 4
```

样例输出：

```text
prompts=3 actions=3 group_size=4
epoch=001 mean_reward=... greedy_reward=... adv_mean=... adv_std=... mean_kl=...

policy
prompt='2+2?' action='4' target='4'
...
```

这个实验仍然不是完整大模型 GRPO。它没有 token-level generation，也没有复杂 rollout buffer。它只负责把 group sampling、rule reward 和 relative advantage 讲清楚。

## 下一步

下一章会做 Capstone，把预训练、SFT、DPO/GRPO 这些概念整理成一条 Mini LLM Pipeline。

上一章：[18. DPO：直接偏好优化](18_dpo_preference_optimization.md)

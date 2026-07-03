---
title: "18. DPO：直接偏好优化"
status: "已完成"
summary: "用 chosen/rejected 偏好对实现 DPO loss，理解 policy/reference log-ratio 如何替代 PPO 的在线 reward 优化。"
---

# 18. DPO：直接偏好优化

## 本章学习契约

- 新增概念：chosen/rejected、reference policy、log-ratio、DPO loss。
- 实验要验证：不用在线采样 reward，也可以直接从偏好对里学习“更偏向 chosen”。
- 实验不验证：它不是完整偏好数据训练流程，也没有对长回答 token log probability 求和。
- 跑完重点看：`pref_acc` 是否提高，policy 相对 reference 是否更偏 chosen，loss 是否随训练下降。

DPO 是 Direct Preference Optimization。它常被看作 PPO 式 RLHF 的一个更简单替代方案。

PPO 的流程是：

```text
policy 采样回答 -> reward model 打分 -> PPO 更新
```

DPO 直接使用偏好对：

```text
prompt, chosen response, rejected response
```

它不需要在训练时在线采样，也不需要显式训练一个 reward model。

## Chosen / Rejected

本章仍然使用 bandit 版小实验。每个 prompt 有一个 chosen action 和一个 rejected action：

```text
prompt='say yes' chosen='yes' rejected='no'
prompt='say no'  chosen='no'  rejected='yes'
prompt='say ok'  chosen='ok'  rejected='yes'
```

真实 DPO 中，chosen/rejected 通常是两段回答文本。模型会在 prompt 条件下，对 response token 的 log probability 求和；prompt token 不计入 chosen/rejected 分数。这个点和第 16 章 SFT 的 response-only label mask 是同一个方向：我们关心回答部分。

## Reference Model

DPO 需要一个 reference policy。它通常来自 SFT 后的模型拷贝，并在 DPO 训练中冻结。

本章代码里：

```text
policy:    会被训练
reference: 初始等于 policy，但冻结
```

这样可以约束 policy 不要无限偏离原模型。

真实大模型训练里，reference 通常会切到 `eval()` 并保持冻结。本章的 `TinyPromptPolicy` 没有 dropout，所以只需要冻结参数即可。

## DPO Loss

先计算 policy 对 chosen 和 rejected 的 log probability 差：

```text
policy_log_ratio = log pi(chosen | prompt) - log pi(rejected | prompt)
```

再计算 reference 的同样差值：

```text
reference_log_ratio = log ref(chosen | prompt) - log ref(rejected | prompt)
```

DPO 的核心 logit 是：

```text
z = beta * (policy_log_ratio - reference_log_ratio)
```

loss 是：

```text
loss = -log sigmoid(z)
```

如果 policy 相比 reference 更偏向 chosen，`z` 会变大，loss 会变小。

## 一个 Sanity Check

如果 policy 和 reference 完全一样：

```text
policy_log_ratio == reference_log_ratio
z = 0
loss = -log sigmoid(0) = log(2)
```

本章测试会专门检查这个性质。它是 DPO 实现是否靠谱的一个很好的信号。

实验输出里的 `pref_acc` 表示 `policy_log_ratio > reference_log_ratio` 的比例。它不是“chosen 概率绝对大于 rejected 概率”的准确率，而是“policy 相比 reference 更偏向 chosen”的比例；如果二者完全打平，tie 不算正确。

举个数值例子：

```text
reference: logp(chosen)=-4, logp(rejected)=-2, reference_log_ratio=-2
policy:    logp(chosen)=-3, logp(rejected)=-2, policy_log_ratio=-1
```

policy 里 chosen 的绝对概率仍然可能低于 rejected，但它已经比 reference 更偏向 chosen 了，所以 DPO 的方向是正确的。DPO 不是把偏好对当普通二分类标签，而是在比较 policy 相对 reference 的偏好变化。

DPO 不显式训练 reward model，不等于没有“奖励”概念。它把人类或规则给出的 chosen/rejected 偏好，隐式转成一个优化信号。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.train_dpo_bandit --epochs 40
```

smoke 版：

```bash
uv run python -m llm_tutor.experiments.train_dpo_bandit --epochs 3
```

样例输出：

```text
preferences=3 actions=3
epoch=001 dpo_loss=0.6931 pref_acc=0.000 logit_mean=+0.0000
epoch=003 dpo_loss=...

policy
prompt='say yes' action='yes' chosen='yes' rejected='no'
...
```

## 和 PPO 的区别

PPO 更像在线强化学习：

- policy 采样；
- reward 打分；
- 用 ratio、advantage、clipping 更新。

DPO 更像离线偏好学习：

- 已经有 chosen/rejected；
- 直接比较 chosen 和 rejected 的 log probability；
- 用 reference model 控制偏离程度。

DPO 的工程链路通常更简单，但它依赖偏好数据质量。

## 下一步

下一章会讲 GRPO。GRPO 会回到“采样多条答案并打分”的路径，但不使用 value model，而是用组内相对 reward 构造 advantage。

上一章：[17. PPO：经典 RLHF 的核心优化器](17_ppo_rlhf.md)

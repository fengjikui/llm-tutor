# Mini LLM Pipeline 实验报告

这份报告对应第 20 章 Capstone。它用 tiny 实验总结从预训练到后训练的完整路线。

## 阶段总览

这是一条学习 pipeline，不是同一个 checkpoint 的连续生产训练链。只有 Pretrain 和 Generate 共享 mini-GPT checkpoint；后训练章节使用独立 tiny 实验讲清优化目标。

| 阶段 | 输入数据 | 训练目标 | loss 或优化信号 | 是否使用 mini-GPT checkpoint | 关键输出字段 |
|---|---|---|---|---|---|
| Pretrain | tiny 普通文本 | 下一个 token | causal LM cross entropy | 写入 checkpoint | `train_loss`, `val_loss`, `checkpoint_saved` |
| Generate | prompt | 采样后续 token | temperature / top-k / top-p | 读取 checkpoint | `loaded_epoch`, generated text |
| Loss Inspect | 短文本 | 检查 `x/y` 错位 | 逐 token cross entropy | 否 | `loss_match`, `future_weight_sum` |
| SFT | instruction/response | 只学习 response token | masked next-token loss | 否，独立 tiny SFT | `sft_loss` |
| PPO | prompt/action/reward | 提高高 reward action | clipped policy objective + KL | 否，bandit policy | `ratio_min`, `ratio_max`, `clipped_fraction` |
| DPO | chosen/rejected | 相对 reference 偏向 chosen | preference log-ratio loss | 否，preference bandit | `dpo_loss`, `pref_acc` |
| GRPO | group samples/reward | 组内相对优势 | relative advantage policy loss + KL | 否，group bandit | `mean_reward`, `adv_mean`, `mean_kl` |

## 复现实验

预览：

```bash
uv run python -m llm_tutor.experiments.run_capstone_pipeline
```

执行短版：

```bash
uv run python -m llm_tutor.experiments.run_capstone_pipeline --execute
```

## 解读方式

不要把这些 tiny 结果当成模型能力评估。它们的作用是让每个训练目标都能被看见、被测试、被单独修改。

真正的大模型训练会把这些思想放大到更复杂的数据、tokenizer、模型规模、优化器、评估集和工程系统里。

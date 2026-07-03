# 教程目录

建议按顺序阅读。当前只在导航里放已经可以阅读和运行的章节，完整路线图见 [../COURSE_OUTLINE.md](../COURSE_OUTLINE.md)。

| 状态 | 章节 |
|---|---|
| 已完成 | [01. 机器学习到底在学什么](01_machine_learning_basics.md) |
| 已完成 | [02. 损失函数、梯度下降和优化器](02_loss_gradient_optimizer.md) |
| 已完成 | [03. PyTorch 快速生存指南](03_pytorch_training_loop.md) |
| 已完成 | [04. 神经网络基础](04_neural_network_basics.md) |
| 已完成 | [05. 训练神经网络的基本功](05_training_neural_networks.md) |
| 已完成 | [06. CNN 为什么适合图像](06_cnn_image_classification.md) |
| 已完成 | [07. RNN：让模型读序列](07_rnn_sequence_modeling.md) |
| 已完成 | [08. LSTM 和 GRU：门控记忆](08_lstm_gru.md) |
| 已完成 | [09. Seq2Seq：从分类到生成](09_seq2seq_translation.md) |
| 已完成 | [10. Attention：让 decoder 不只看一个向量](10_attention_seq2seq.md) |
| 已完成 | [11. Self-Attention 从零实现](11_self_attention_from_scratch.md) |
| 已完成 | [12. Transformer Block：把 Self-Attention 组装起来](12_transformer_block.md) |
| 已完成 | [13. GPT 从零实现：decoder-only 语言模型](13_gpt_from_scratch.md) |
| 已完成 | [14. GPT 的训练数据、目标数据和损失函数](14_gpt_data_target_loss.md) |
| 已完成 | [15. 小型 GPT 预训练工程化](15_mini_gpt_pretraining.md) |
| 已完成 | [16. SFT：让模型学会按指令回答](16_sft_instruction_tuning.md) |
| 已完成 | [17. PPO：经典 RLHF 的核心优化器](17_ppo_rlhf.md) |
| 已完成 | [18. DPO：直接偏好优化](18_dpo_preference_optimization.md) |
| 已完成 | [19. GRPO：面向可验证任务的组内相对优化](19_grpo_verifiable_tasks.md) |
| 已完成 | [20. Capstone：Mini LLM Pipeline](20_capstone_mini_llm_pipeline.md) |
| 已完成 | [21. 现代 LLM 组件：RoPE、GQA、MLA、稀疏注意力与 KV Cache](21_modern_llm_components.md) |
| 已完成 | [22. 多模态小系列：从 ViT、CLIP、ClipCap 到现代 VLM/VLA](22_multimodal_vit_clip_vlm.md) |
| 已完成 | [23. Diffusion 生图：从加噪去噪到 Stable Diffusion 和 DiT](23_diffusion_image_generation.md) |

到第 20 章为止，训练闭环已经从表格分类、图像分类、序列分类、翻译任务一路扩展到自回归语言模型，并进入了 SFT、PPO、DPO、GRPO 和 Capstone：

```text
准备数据 -> 前向传播 -> 计算损失 -> 反向传播 -> 更新参数 -> 评估指标
```

第 21-23 章是主线之后的现代组件补充：它们不强行设计训练实验，而是帮助读者读懂真实厂商模型代码、现代多模态架构和生图模型。

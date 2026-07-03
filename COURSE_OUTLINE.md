# 从梯度下降到 GPT：课程大纲

## 课程定位

这套教程面向已经会写 Python、但还没有系统理解深度学习和大语言模型训练的人。它不追求把所有经典模型都讲全，而是沿着一条清晰主线推进：

```text
机器学习训练循环
-> 神经网络基础
-> CNN / RNN / LSTM / GRU
-> Seq2Seq / Attention
-> Transformer
-> GPT 预训练
-> SFT / PPO / DPO / GRPO
```

前半段帮助读者建立损失函数、前向传播、反向传播、梯度下降、优化器、泛化和模型表达能力的概念。后半段重点讲 Transformer 和大语言模型，最终实现一个从零开始的小型 GPT，并继续实现 SFT、PPO、DPO、GRPO 的最小实验脚本。

## 当前实现说明

当前课程已经落地为 20 篇 Markdown 教程和一组 PyTorch 脚本：

- 教程入口：[tutorials/README.md](tutorials/README.md)
- 实验入口：`src/llm_tutor/experiments/`
- 统一 smoke test：`scripts/smoke_test.sh`
- 专栏网站入口：[site/index.html](site/index.html)

本大纲后半部分仍保留一些早期“计划产出”字段，例如 notebooks 或未来扩展实验名。阅读与复现实验时，以 `tutorials/*.md` 正文和实际 `src/llm_tutor/experiments/` 脚本为准；这些旧计划字段会在后续迭代里逐步改成当前文件索引和扩展路线。

## 统一章节模板

每一章尽量保持一致结构：

1. 本章学习契约：新增概念、实验目标、实验边界、重点观察输出。
2. 这一章要解决什么问题，以及为什么它从上一章自然出现。
3. 必要概念和最小数学直觉。
4. PyTorch 实现与关键 shape。
5. 一个真实可运行实验。
6. 指标、实验观察和失败现象解读。
7. 常见错误与下一章连接。

## 当前进度

| 状态 | 章节 |
|---|---|
| 已完成 | 01. 机器学习到底在学什么 |
| 已完成 | 02. 损失函数、梯度下降和优化器 |
| 已完成 | 03. PyTorch 快速生存指南 |
| 已完成 | 04. 神经网络基础 |
| 已完成 | 05. 训练神经网络的基本功 |
| 已完成 | 06. CNN 为什么适合图像 |
| 已完成 | 07. RNN：让模型读序列 |
| 已完成 | 08. LSTM 和 GRU：门控记忆 |
| 已完成 | 09. Seq2Seq：从分类到生成 |
| 已完成 | 10. Attention：让 decoder 不只看一个向量 |
| 已完成 | 11. Self-Attention 从零实现 |
| 已完成 | 12. Transformer Block：把 Self-Attention 组装起来 |
| 已完成 | 13. GPT 从零实现：decoder-only 语言模型 |
| 已完成 | 14. GPT 的训练数据、目标数据和损失函数 |
| 已完成 | 15. 小型 GPT 预训练工程化 |
| 已完成 | 16. SFT：让模型学会按指令回答 |
| 已完成 | 17. PPO：经典 RLHF 的核心优化器 |
| 已完成 | 18. DPO：直接偏好优化 |
| 已完成 | 19. GRPO：面向可验证任务的组内相对优化 |
| 已完成 | 20. Capstone：Mini LLM Pipeline |

## 第一部分：机器学习最小地基

### 01. 机器学习到底在学什么

核心内容：

- 数据、特征、标签、模型、参数、训练集、验证集、测试集。
- 监督学习、分类、回归。
- 经验风险、泛化、过拟合、欠拟合。
- 为什么模型不是“背答案”，而是在学习一个函数。

实验设计：

- 使用一个真实表格分类数据集，例如 Adult Income、Bank Marketing 或 Covertype。
- 先用 `sklearn` 跑一个 baseline，再用 PyTorch 写一个线性分类器。
- 对比 accuracy、precision、recall、F1。

计划产出：

- `notebooks/01_machine_learning_basics.ipynb`
- `src/llm_tutor/data/tabular.py`
- `src/llm_tutor/experiments/train_linear_classifier.py`

### 02. 损失函数、梯度下降和优化器

核心内容：

- MSE、交叉熵、logits、softmax。
- batch、epoch、learning rate。
- 梯度下降如何更新参数。
- SGD、Momentum、Adam 的直觉。

实验设计：

- 从零实现一维线性回归。
- 手动计算梯度并更新参数。
- 再用 PyTorch autograd 和 optimizer 重写。

计划产出：

- `notebooks/02_loss_gradient_optimizer.ipynb`
- `src/llm_tutor/foundations/manual_gradient_descent.py`

### 03. PyTorch 快速生存指南

核心内容：

- Tensor、shape、dtype、device。
- `Dataset`、`DataLoader`。
- `nn.Module`、`forward`、`loss.backward()`、`optimizer.step()`。
- checkpoint、随机种子、训练日志。

实验设计：

- 把前两章的训练流程抽象成通用 train/evaluate loop。
- 为后续 CNN、RNN、Transformer 复用。

计划产出：

- `notebooks/03_pytorch_training_loop.ipynb`
- `src/llm_tutor/training/loop.py`
- `src/llm_tutor/training/metrics.py`

## 第二部分：神经网络基础

### 04. 神经网络基础：从线性模型到多层非线性函数

核心内容：

- 神经元、层、激活函数、隐藏表示。
- 为什么只堆线性层没有意义。
- ReLU、Sigmoid、Tanh 的行为差异。
- 前向传播和反向传播在神经网络里的位置。
- 神经网络为什么能起作用：可微函数组合、非线性、表示学习、数据驱动参数搜索。

实验设计：

- 使用 Fashion-MNIST 或真实表格数据做分类。
- 先训练线性模型，再训练两层/三层神经网络。
- 比较训练曲线和验证集指标。

计划产出：

- `notebooks/04_neural_network_basics.ipynb`
- `src/llm_tutor/models/feedforward.py`
- `src/llm_tutor/experiments/train_neural_network_basics.py`

### 05. 训练神经网络的基本功

核心内容：

- 参数初始化。
- normalization、dropout、weight decay。
- learning rate schedule。
- early stopping。
- 梯度爆炸、梯度消失、梯度裁剪。

实验设计：

- 在同一个分类任务上比较不同学习率、batch size、优化器、dropout。
- 观察 loss 曲线、验证集指标和过拟合现象。

计划产出：

- `notebooks/05_training_neural_networks.ipynb`
- `src/llm_tutor/experiments/compare_optimizers.py`

## 第三部分：CNN 与视觉分类

### 06. CNN 为什么适合图像

核心内容：

- 图像张量的 shape：`NCHW`。
- 卷积核、局部感受野、参数共享。
- padding、stride、pooling。
- 通道与特征图。
- CNN 的归纳偏置：局部模式、平移等变性。

实验设计：

- MNIST 手写数字识别。
- CIFAR-10 或 Fashion-MNIST 图像分类。
- 从 LeNet 风格 CNN 开始，再引入一个简单残差块。

计划产出：

- `notebooks/06_cnn_image_classification.ipynb`
- `src/llm_tutor/models/cnn.py`
- `src/llm_tutor/experiments/train_cnn.py`

## 第四部分：序列模型

### 07. RNN：让模型读序列

核心内容：

- 序列、时间步、hidden state。
- RNN 的时间展开。
- BPTT。
- 梯度消失和梯度爆炸。

实验设计：

- 字符级名字分类，或小型情感分类。
- 可视化不同时间步 hidden state 的影响。

计划产出：

- `notebooks/07_rnn_sequence_modeling.ipynb`
- `src/llm_tutor/models/rnn.py`
- `src/llm_tutor/experiments/train_rnn_classifier.py`

### 08. LSTM 和 GRU：门控记忆

核心内容：

- 普通 RNN 的长依赖问题。
- LSTM 的 forget/input/output gates。
- GRU 的 reset/update gates。
- LSTM、GRU 和普通 RNN 的取舍。

实验设计：

- IMDB、SST-2 或 AG News 分类。
- 比较 RNN、GRU、LSTM 的收敛速度和验证集表现。

计划产出：

- `notebooks/08_lstm_gru.ipynb`
- `src/llm_tutor/models/recurrent.py`
- `src/llm_tutor/experiments/compare_rnn_lstm_gru.py`

## 第五部分：Seq2Seq 与注意力

### 09. Seq2Seq：从分类到生成

核心内容：

- Encoder-Decoder。
- BOS/EOS、padding、mask。
- Teacher forcing。
- 自回归解码。
- 序列到序列损失。

实验设计：

- 小型机器翻译任务，例如英法、英德或中英 toy translation。
- 用 RNN encoder-decoder 先跑通完整翻译流程。

计划产出：

- `notebooks/09_seq2seq_translation.ipynb`
- `src/llm_tutor/models/seq2seq.py`
- `src/llm_tutor/experiments/train_seq2seq_translation.py`

### 10. Attention：让 decoder 不只看一个向量

核心内容：

- Seq2Seq 固定上下文向量的瓶颈。
- Bahdanau / Luong attention 的直觉。
- alignment。
- attention score、softmax、context vector。

实验设计：

- 在上一章翻译模型上加入 attention。
- 可视化源语言和目标语言 token 的对齐热力图。

计划产出：

- `notebooks/10_attention_seq2seq.ipynb`
- `src/llm_tutor/models/attention_rnn.py`
- `src/llm_tutor/visualization/attention_heatmap.py`

## 第六部分：Transformer 主线

### 11. Self-Attention 从零实现

核心内容：

- Query、Key、Value。
- Scaled dot-product attention。
- Multi-head attention。
- Padding mask 和 causal mask。
- 为什么 self-attention 更容易并行。

实验设计：

- 从零写一个 `MultiHeadAttention`。
- 用 shape trace 解释每一步张量变化。
- 在文本分类或小型语言建模任务上测试。

计划产出：

- `notebooks/11_self_attention_from_scratch.ipynb`
- `src/llm_tutor/models/attention.py`

### 12. Transformer Encoder、Decoder 与三类模型

核心内容：

- Transformer block。
- LayerNorm、residual、FFN。
- 为 GPT 铺垫 masked self-attention block。
- Encoder-only、Decoder-only、Encoder-Decoder 的结构差异放到后续补充章节展开。
- BERT、GPT、T5 的对比放到 GPT 主线之后回看。

实验设计：

- 用 Transformer encoder 做文本分类。
- 用 Transformer encoder-decoder 做小型翻译。
- 为下一章 GPT decoder-only 铺垫。

计划产出：

- `notebooks/12_transformer_architectures.ipynb`
- `src/llm_tutor/models/transformer.py`
- `src/llm_tutor/experiments/train_transformer_translation.py`

## 第七部分：GPT 从零实现

### 13. GPT 架构拆解

核心内容：

- Token embedding。
- Position embedding。
- Causal self-attention。
- Pre-LN Transformer block。
- Residual connection。
- MLP/FFN。
- LM head。

实验设计：

- 从零写一个教育版 mini-GPT。
- 在 Tiny Shakespeare、中文小说片段或自制小语料上训练。
- 观察训练 loss 和生成文本变化。

计划产出：

- `notebooks/13_gpt_from_scratch.ipynb`
- `src/llm_tutor/models/gpt.py`
- `src/llm_tutor/data/language_modeling.py`
- `src/llm_tutor/experiments/train_mini_gpt.py`

### 14. GPT 的训练数据、目标数据和损失函数

核心内容：

- Next-token prediction。
- 输入 `x = tokens[:-1]`。
- 目标 `y = tokens[1:]`。
- Causal mask 为什么不能看未来。
- Cross entropy 如何在每个 token 位置计算。
- block size 和 batching。

实验设计：

- 从原始文本到 tokenizer。
- 从 token ids 到 batch。
- 从 logits 到 shifted labels。
- 手动检查 loss 的 shape 和数值。

计划产出：

- `notebooks/14_gpt_data_target_loss.ipynb`
- `src/llm_tutor/data/language_modeling.py`
- `src/llm_tutor/experiments/inspect_gpt_loss.py`

### 15. 小型 GPT 预训练工程化

核心内容：

- checkpoint。
- validation loss。
- gradient clipping。
- temperature、top-k、top-p。
- 训练日志和生成样例。

实验设计：

- 训练一个可复现 mini-GPT。
- 写一个生成脚本。
- 比较不同采样参数下的输出。

计划产出：

- `notebooks/15_mini_gpt_pretraining.ipynb`
- `src/llm_tutor/generation/sampling.py`
- `src/llm_tutor/experiments/generate_with_mini_gpt.py`

## 第八部分：从预训练到指令模型

### 16. SFT：让模型学会按指令回答

核心内容：

- Instruction tuning。
- Prompt/response 格式。
- Chat template。
- Label masking。
- 为什么 SFT 仍然是 next-token loss。

实验设计：

- 构造一个小型 instruction dataset。
- 对 mini-GPT 或 Hugging Face 小模型做 SFT。
- 对比 SFT 前后的回答风格。

计划产出：

- `notebooks/16_sft_instruction_tuning.ipynb`
- `src/llm_tutor/post_training/sft.py`

### 17. PPO：经典 RLHF 的核心优化器

核心内容：

- RLHF 三段式：SFT policy、reward model、PPO。
- Policy、reference model、reward、KL penalty。
- Advantage。
- PPO clipping 的直觉。

实验设计：

- 用规则 reward 或小 reward model 做一个最小 PPO 实验。
- 任务可以是情感控制生成、摘要长度控制或简单数学格式任务。

计划产出：

- `notebooks/17_ppo_rlhf.ipynb`
- `src/llm_tutor/post_training/ppo.py`

### 18. DPO：直接偏好优化

核心内容：

- chosen/rejected 偏好数据。
- Reference model。
- DPO loss 的直觉。
- DPO 相比 PPO 的简化点。

实验设计：

- 构造一个小型偏好数据集。
- 对 SFT 后模型做 DPO。
- 观察 chosen/rejected log-prob 差异。

计划产出：

- `notebooks/18_dpo_preference_optimization.ipynb`
- `src/llm_tutor/post_training/dpo.py`

### 19. GRPO：面向可验证任务的组内相对优化

核心内容：

- Group sampling。
- Rule-based reward。
- Relative advantage。
- 为什么 GRPO 适合数学、代码、Countdown 这类可验证任务。
- GRPO 和 PPO、DPO 的关系。

实验设计：

- Countdown 或 GSM8K 风格小任务。
- 每个 prompt 采样多条答案。
- 用规则 reward 打分。
- 根据组内相对优势更新模型。

计划产出：

- `notebooks/19_grpo_verifiable_tasks.ipynb`
- `src/llm_tutor/post_training/grpo.py`

## 最终项目：从零到一个小型 LLM 训练流水线

### 20. Capstone：Mini LLM Pipeline

核心内容：

- 预训练：小语料 causal LM。
- SFT：指令数据微调。
- DPO：偏好数据对齐。
- GRPO：可验证推理任务优化。
- 评估：loss、生成样例、简单 benchmark、失败案例。

实验设计：

- 使用同一个 mini-GPT 作为主角。
- 逐阶段保存 checkpoint。
- 每阶段写一份实验报告，说明数据、目标、损失函数、指标和失败案例。

计划产出：

- `notebooks/20_capstone_mini_llm_pipeline.ipynb`
- `src/llm_tutor/capstone/`
- `reports/capstone_report.md`

## 推荐实现顺序

第一阶段先完成：

1. 项目脚手架和 PyTorch 环境。
2. 第 01-03 章：机器学习、损失函数、训练循环。
3. 第 04-05 章：神经网络基础和训练技巧。

第二阶段完成：

1. CNN 图像分类。
2. RNN / LSTM / GRU 序列分类。
3. Seq2Seq 翻译和 Attention 可视化。

第三阶段完成：

1. Transformer 从零实现。
2. GPT 从零实现。
3. GPT 数据、目标、损失函数专题。

第四阶段完成：

1. SFT。
2. PPO。
3. DPO。
4. GRPO。
5. Capstone 串联。

## 设计原则

- 前半段只讲对理解 LLM 必要的经典知识。
- 每个概念都要落到可运行代码。
- 每个实验都要有指标和失败案例分析。
- 不用复杂框架遮住训练循环。
- 后期可以引入 Hugging Face Transformers / TRL，但核心损失函数和数据格式必须先手写清楚。
- 代码尽量保持小而清晰，优先教育价值，其次才是性能。

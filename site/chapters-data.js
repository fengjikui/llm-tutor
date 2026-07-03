window.LLM_TUTOR_CHAPTERS = [
  {
    "no": "01",
    "title": "机器学习到底在学什么",
    "summary": "用一个表格二分类实验跑通数据、模型、损失、优化和评估的第一条训练闭环。",
    "phase": "foundation",
    "phaseLabel": "最小地基",
    "tags": [
      "ML",
      "分类",
      "训练循环"
    ],
    "file": "01_machine_learning_basics.md",
    "url": "./chapters/01_machine_learning_basics.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/01_machine_learning_basics.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/data/tabular.py",
        "src/llm_tutor/experiments/train_linear_classifier.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 20"
      ],
      "note": "从真实表格分类数据开始，先跑通 data -> model -> loss -> optimizer -> metrics。",
      "artifactHint": true
    }
  },
  {
    "no": "02",
    "title": "损失函数、梯度下降和优化器",
    "summary": "从手写线性回归梯度开始，理解 loss、gradient、learning rate 和 autograd。",
    "phase": "foundation",
    "phaseLabel": "最小地基",
    "tags": [
      "loss",
      "gradient",
      "SGD"
    ],
    "file": "02_loss_gradient_optimizer.md",
    "url": "./chapters/02_loss_gradient_optimizer.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/02_loss_gradient_optimizer.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/foundations/manual_gradient_descent.py",
        "src/llm_tutor/experiments/manual_gradient_descent.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.manual_gradient_descent"
      ],
      "note": "用手写梯度和 autograd 对照，观察 loss、梯度和参数更新方向。",
      "artifactHint": false
    }
  },
  {
    "no": "03",
    "title": "PyTorch 快速生存指南",
    "summary": "建立后续所有实验都会复用的 PyTorch 训练循环和 shape 读法。",
    "phase": "foundation",
    "phaseLabel": "最小地基",
    "tags": [
      "PyTorch",
      "DataLoader",
      "loop"
    ],
    "file": "03_pytorch_training_loop.md",
    "url": "./chapters/03_pytorch_training_loop.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/03_pytorch_training_loop.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/training/loop.py",
        "src/llm_tutor/experiments/train_linear_classifier.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 5"
      ],
      "note": "把后面章节都会复用的 PyTorch 训练循环拆开看清楚。",
      "artifactHint": true
    }
  },
  {
    "no": "04",
    "title": "神经网络基础：从线性模型到多层非线性函数",
    "summary": "理解多层线性变换、非线性激活和表示学习如何组成最小神经网络。",
    "phase": "foundation",
    "phaseLabel": "最小地基",
    "tags": [
      "NN",
      "ReLU",
      "表示学习"
    ],
    "file": "04_neural_network_basics.md",
    "url": "./chapters/04_neural_network_basics.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/04_neural_network_basics.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/feedforward.py",
        "src/llm_tutor/experiments/train_neural_network_basics.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_neural_network_basics --epochs 30"
      ],
      "note": "比较线性模型和带隐藏层的神经网络，理解非线性表示能力。",
      "artifactHint": true
    }
  },
  {
    "no": "05",
    "title": "训练神经网络的基本功",
    "summary": "比较学习率、优化器、weight decay 和 Dropout，理解训练策略如何影响同一个神经网络。",
    "phase": "foundation",
    "phaseLabel": "最小地基",
    "tags": [
      "optimizer",
      "dropout",
      "schedule"
    ],
    "file": "05_training_neural_networks.md",
    "url": "./chapters/05_training_neural_networks.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/05_training_neural_networks.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/feedforward.py",
        "src/llm_tutor/experiments/compare_training_strategies.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.compare_training_strategies --epochs 12"
      ],
      "note": "在同一任务上比较学习率、优化器、weight decay 和 Dropout 对训练曲线的影响。",
      "artifactHint": true
    }
  },
  {
    "no": "06",
    "title": "CNN 为什么适合图像",
    "summary": "用 Fashion-MNIST 图像分类理解卷积核、局部感受野、参数共享、池化和图像张量 shape。",
    "phase": "sequence",
    "phaseLabel": "视觉与序列",
    "tags": [
      "CNN",
      "vision",
      "Fashion-MNIST"
    ],
    "file": "06_cnn_image_classification.md",
    "url": "./chapters/06_cnn_image_classification.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/06_cnn_image_classification.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/data/vision.py",
        "src/llm_tutor/models/cnn.py",
        "src/llm_tutor/experiments/train_cnn.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_cnn --epochs 2",
        "uv run python -m llm_tutor.experiments.train_cnn --epochs 1 --train-limit 512 --val-limit 128 --test-limit 128"
      ],
      "note": "用 Fashion-MNIST 图像分类把卷积、图像 shape 和真实验证指标连起来。",
      "artifactHint": true
    }
  },
  {
    "no": "07",
    "title": "RNN：让模型读序列",
    "summary": "用字符级姓名分类理解时间步、hidden state、padding 和循环神经网络的基本训练方式。",
    "phase": "sequence",
    "phaseLabel": "视觉与序列",
    "tags": [
      "RNN",
      "sequence",
      "BPTT"
    ],
    "file": "07_rnn_sequence_modeling.md",
    "url": "./chapters/07_rnn_sequence_modeling.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/07_rnn_sequence_modeling.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/data/names.py",
        "src/llm_tutor/models/rnn.py",
        "src/llm_tutor/experiments/train_rnn_classifier.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_rnn_classifier --epochs 20"
      ],
      "note": "用字符级姓名分类理解时间步、hidden state 和序列分类。",
      "artifactHint": true
    }
  },
  {
    "no": "08",
    "title": "LSTM 和 GRU：门控记忆",
    "summary": "在同一个字符级分类任务上比较普通 RNN、GRU 和 LSTM，理解门控如何缓解长依赖问题。",
    "phase": "sequence",
    "phaseLabel": "视觉与序列",
    "tags": [
      "LSTM",
      "GRU",
      "gates"
    ],
    "file": "08_lstm_gru.md",
    "url": "./chapters/08_lstm_gru.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/08_lstm_gru.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/rnn.py",
        "src/llm_tutor/experiments/compare_recurrent_cells.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.compare_recurrent_cells --epochs 12"
      ],
      "note": "在同一数据集上比较 RNN、GRU 和 LSTM 的收敛与验证指标。",
      "artifactHint": true
    }
  },
  {
    "no": "09",
    "title": "Seq2Seq：从分类到生成",
    "summary": "用一个不带 attention 的 Encoder-Decoder 翻译 toy 实验理解 BOS/EOS、teacher forcing、自回归解码和序列损失。",
    "phase": "sequence",
    "phaseLabel": "视觉与序列",
    "tags": [
      "Seq2Seq",
      "translation",
      "decoder"
    ],
    "file": "09_seq2seq_translation.md",
    "url": "./chapters/09_seq2seq_translation.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/09_seq2seq_translation.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/data/translation.py",
        "src/llm_tutor/models/seq2seq.py",
        "src/llm_tutor/experiments/train_seq2seq_translation.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_seq2seq_translation --epochs 80"
      ],
      "note": "从分类任务过渡到序列生成，观察 source、target 和 prediction 的对齐。",
      "artifactHint": true
    }
  },
  {
    "no": "10",
    "title": "Attention：让 decoder 不只看一个向量",
    "summary": "在 Seq2Seq 翻译 toy 实验上加入 additive attention，理解对齐权重、context vector 和 Q/K/V 的前身直觉。",
    "phase": "sequence",
    "phaseLabel": "视觉与序列",
    "tags": [
      "attention",
      "alignment",
      "context"
    ],
    "file": "10_attention_seq2seq.md",
    "url": "./chapters/10_attention_seq2seq.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/10_attention_seq2seq.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/seq2seq.py",
        "src/llm_tutor/experiments/train_attention_seq2seq.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_attention_seq2seq --epochs 80"
      ],
      "note": "给 Seq2Seq 加上 additive attention，重点看 attention 权重和生成结果。",
      "artifactHint": true
    }
  },
  {
    "no": "11",
    "title": "Self-Attention 从零实现",
    "summary": "手写 scaled dot-product attention 和 multi-head self-attention，理解 Q/K/V、padding mask、causal mask 和 shape 变化。",
    "phase": "transformer",
    "phaseLabel": "Transformer",
    "tags": [
      "self-attention",
      "mask",
      "QKV"
    ],
    "file": "11_self_attention_from_scratch.md",
    "url": "./chapters/11_self_attention_from_scratch.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/11_self_attention_from_scratch.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/attention.py",
        "src/llm_tutor/experiments/inspect_self_attention.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.inspect_self_attention"
      ],
      "note": "不训练模型，专门检查 Q/K/V、padding mask 和 causal mask 的 shape。",
      "artifactHint": false
    }
  },
  {
    "no": "12",
    "title": "Transformer Block：把 Self-Attention 组装起来",
    "summary": "实现一个 Pre-LN Transformer block，理解 residual、LayerNorm、MLP 和 causal self-attention 如何组成 GPT 的基本积木。",
    "phase": "transformer",
    "phaseLabel": "Transformer",
    "tags": [
      "Transformer",
      "LayerNorm",
      "MLP"
    ],
    "file": "12_transformer_block.md",
    "url": "./chapters/12_transformer_block.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/12_transformer_block.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/transformer.py",
        "src/llm_tutor/experiments/inspect_transformer_block.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.inspect_transformer_block"
      ],
      "note": "把 self-attention、LayerNorm、residual 和 MLP 组装成一个 GPT block。",
      "artifactHint": false
    }
  },
  {
    "no": "13",
    "title": "GPT 从零实现：decoder-only 语言模型",
    "summary": "把 token embedding、position embedding、causal Transformer block 和 LM head 串成一个可训练的字符级 mini-GPT。",
    "phase": "transformer",
    "phaseLabel": "Transformer",
    "tags": [
      "GPT",
      "decoder-only",
      "LM head"
    ],
    "file": "13_gpt_from_scratch.md",
    "url": "./chapters/13_gpt_from_scratch.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/13_gpt_from_scratch.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/gpt.py",
        "src/llm_tutor/experiments/train_mini_gpt.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_mini_gpt --epochs 20"
      ],
      "note": "从零训练字符级 mini-GPT，观察 next-token loss 和生成样例。",
      "artifactHint": true
    }
  },
  {
    "no": "14",
    "title": "GPT 的训练数据、目标数据和损失函数",
    "summary": "用一个显微镜实验拆开 next-token prediction：x/y 错位、logits shape、逐 token cross entropy 和 causal mask。",
    "phase": "transformer",
    "phaseLabel": "Transformer",
    "tags": [
      "next-token",
      "loss",
      "causal"
    ],
    "file": "14_gpt_data_target_loss.md",
    "url": "./chapters/14_gpt_data_target_loss.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/14_gpt_data_target_loss.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/gpt.py",
        "src/llm_tutor/experiments/inspect_gpt_loss.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.inspect_gpt_loss"
      ],
      "note": "用显微镜实验检查 x/y 错位、logits shape、逐 token loss 和 causal mask。",
      "artifactHint": false
    }
  },
  {
    "no": "15",
    "title": "小型 GPT 预训练工程化",
    "summary": "在 mini-GPT 上加入 checkpoint、validation loss、梯度裁剪和 temperature/top-k/top-p 采样，让实验从能跑变成可复现。",
    "phase": "transformer",
    "phaseLabel": "Transformer",
    "tags": [
      "pretrain",
      "checkpoint",
      "sampling"
    ],
    "file": "15_mini_gpt_pretraining.md",
    "url": "./chapters/15_mini_gpt_pretraining.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/15_mini_gpt_pretraining.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/experiments/train_mini_gpt.py",
        "src/llm_tutor/experiments/generate_with_mini_gpt.py",
        "src/llm_tutor/generation/sampling.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_mini_gpt --epochs 2 --output-dir runs/mini-gpt-smoke",
        "uv run python -m llm_tutor.experiments.generate_with_mini_gpt --checkpoint-path runs/mini-gpt-smoke/mini_gpt.pt"
      ],
      "note": "把 mini-GPT 训练变成可保存、可加载、可复盘的实验。",
      "artifactHint": true
    }
  },
  {
    "no": "16",
    "title": "SFT：让模型学会按指令回答",
    "summary": "用 tiny instruction 数据演示 SFT：prompt/response 模板、response-only label mask，以及为什么 SFT 仍然是 next-token loss。",
    "phase": "post-training",
    "phaseLabel": "Post-training",
    "tags": [
      "SFT",
      "instruction",
      "mask"
    ],
    "file": "16_sft_instruction_tuning.md",
    "url": "./chapters/16_sft_instruction_tuning.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/16_sft_instruction_tuning.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/post_training/sft.py",
        "src/llm_tutor/experiments/train_sft.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_sft --epochs 30"
      ],
      "note": "观察 instruction/response 模板、response-only mask 和 SFT loss。",
      "artifactHint": true
    }
  },
  {
    "no": "17",
    "title": "PPO：经典 RLHF 的核心优化器",
    "summary": "用一个 tiny bandit 实验解释 PPO 的 policy、old policy、advantage、ratio、clipping、entropy 和 KL penalty。",
    "phase": "post-training",
    "phaseLabel": "Post-training",
    "tags": [
      "PPO",
      "RLHF",
      "policy"
    ],
    "file": "17_ppo_rlhf.md",
    "url": "./chapters/17_ppo_rlhf.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/17_ppo_rlhf.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/post_training/ppo.py",
        "src/llm_tutor/experiments/train_ppo_bandit.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_ppo_bandit --epochs 30"
      ],
      "note": "用 bandit 显微镜看 PPO 的 ratio、clipping、KL 和 reward 变化。",
      "artifactHint": true
    }
  },
  {
    "no": "18",
    "title": "DPO：直接偏好优化",
    "summary": "用 chosen/rejected 偏好对实现 DPO loss，理解 policy/reference log-ratio 如何替代 PPO 的在线 reward 优化。",
    "phase": "post-training",
    "phaseLabel": "Post-training",
    "tags": [
      "DPO",
      "preference",
      "KL"
    ],
    "file": "18_dpo_preference_optimization.md",
    "url": "./chapters/18_dpo_preference_optimization.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/18_dpo_preference_optimization.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/post_training/dpo.py",
        "src/llm_tutor/experiments/train_dpo_bandit.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_dpo_bandit --epochs 40"
      ],
      "note": "从 chosen/rejected 偏好对出发，观察 policy 相对 reference 的偏好移动。",
      "artifactHint": true
    }
  },
  {
    "no": "19",
    "title": "GRPO：面向可验证任务的组内相对优化",
    "summary": "用 group sampling 和 rule-based reward 实现 GRPO 的最小实验，理解组内 relative advantage 为什么适合可验证任务。",
    "phase": "post-training",
    "phaseLabel": "Post-training",
    "tags": [
      "GRPO",
      "RLVR",
      "group"
    ],
    "file": "19_grpo_verifiable_tasks.md",
    "url": "./chapters/19_grpo_verifiable_tasks.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/19_grpo_verifiable_tasks.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/post_training/grpo.py",
        "src/llm_tutor/experiments/train_grpo_bandit.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.train_grpo_bandit --epochs 40 --group-size 4"
      ],
      "note": "用 group sampling 和 rule-based reward 理解组内相对 advantage。",
      "artifactHint": true
    }
  },
  {
    "no": "20",
    "title": "Capstone：Mini LLM Pipeline",
    "summary": "把 mini-GPT 预训练、loss 检查、SFT、PPO、DPO、GRPO 串成一条可运行的学习流水线。",
    "phase": "post-training",
    "phaseLabel": "Post-training",
    "tags": [
      "capstone",
      "pipeline",
      "LLM"
    ],
    "file": "20_capstone_mini_llm_pipeline.md",
    "url": "./chapters/20_capstone_mini_llm_pipeline.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/20_capstone_mini_llm_pipeline.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/capstone/pipeline.py",
        "src/llm_tutor/experiments/run_capstone_pipeline.py"
      ],
      "commands": [
        "uv run python -m llm_tutor.experiments.run_capstone_pipeline",
        "uv run python -m llm_tutor.experiments.run_capstone_pipeline --execute"
      ],
      "note": "把 mini-GPT、SFT、PPO、DPO、GRPO 的教学实验放进一条可运行路线图。",
      "artifactHint": true
    }
  },
  {
    "no": "21",
    "title": "现代 LLM 组件：RoPE、GQA、MLA、稀疏注意力与 KV Cache",
    "summary": "补上现代厂商模型里常见的工程组件：旋转位置编码、Grouped-Query Attention、DeepSeek MLA、稀疏注意力、FlashAttention 和推理 KV cache。",
    "phase": "modern",
    "phaseLabel": "现代扩展",
    "tags": [
      "RoPE",
      "GQA",
      "MLA"
    ],
    "file": "21_modern_llm_components.md",
    "url": "./chapters/21_modern_llm_components.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/21_modern_llm_components.md",
    "lab": {
      "codePaths": [
        "src/llm_tutor/models/modern_attention.py",
        "tests/test_modern_attention.py"
      ],
      "commands": [
        "uv run pytest tests/test_modern_attention.py -q"
      ],
      "note": "本章以可读实现和 shape 测试为主，帮助读懂 RoPE、KV Cache、GQA 和 MLA。",
      "artifactHint": false
    }
  },
  {
    "no": "22",
    "title": "多模态小系列：从 ViT、CLIP、ClipCap 到现代 VLM/VLA",
    "summary": "理解图像如何变成 token，CLIP 如何做图文对齐，ClipCap 如何把 CLIP 图像表示映射成语言模型 prefix，并一路过渡到 Qwen3-VL、Qwen3.5-Omni、Qwen-VLA 和 encoder-free multimodal。",
    "phase": "modern",
    "phaseLabel": "现代扩展",
    "tags": [
      "ClipCap",
      "Qwen3-VL",
      "VLA"
    ],
    "file": "22_multimodal_vit_clip_vlm.md",
    "url": "./chapters/22_multimodal_vit_clip_vlm.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/22_multimodal_vit_clip_vlm.md",
    "lab": {
      "codePaths": [],
      "commands": [],
      "note": "本章是多模态概念扩展，当前不要求读者训练 VLM；后续可单独补成 GPU Lab。",
      "artifactHint": false
    }
  },
  {
    "no": "23",
    "title": "Diffusion 生图：从加噪去噪到 Stable Diffusion 和 DiT",
    "summary": "用最少数学理解扩散模型：前向加噪、反向去噪、噪声预测、latent diffusion、classifier-free guidance、Stable Diffusion 与 Diffusion Transformer。",
    "phase": "modern",
    "phaseLabel": "现代扩展",
    "tags": [
      "DDPM",
      "Stable Diffusion",
      "DiT"
    ],
    "file": "23_diffusion_image_generation.md",
    "url": "./chapters/23_diffusion_image_generation.html",
    "githubUrl": "https://github.com/fengjikui/llm-tutor/blob/main/tutorials/23_diffusion_image_generation.md",
    "lab": {
      "codePaths": [],
      "commands": [],
      "note": "本章是 diffusion 概念扩展，当前不要求读者训练生图模型；后续可单独补采样实验。",
      "artifactHint": false
    }
  }
];

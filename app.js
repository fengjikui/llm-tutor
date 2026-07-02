const chapters = [
  {
    no: "01",
    title: "机器学习到底在学什么",
    summary: "用表格二分类跑通数据、模型、损失、优化和评估的第一条训练闭环。",
    phase: "foundation",
    tags: ["ML", "分类", "训练循环"],
    file: "01_machine_learning_basics.md",
  },
  {
    no: "02",
    title: "损失函数、梯度下降和优化器",
    summary: "从一维线性回归手写梯度，再过渡到 autograd 和 optimizer。",
    phase: "foundation",
    tags: ["loss", "gradient", "SGD"],
    file: "02_loss_gradient_optimizer.md",
  },
  {
    no: "03",
    title: "PyTorch 快速生存指南",
    summary: "理解 Tensor、Dataset、DataLoader、nn.Module 和通用训练循环。",
    phase: "foundation",
    tags: ["PyTorch", "DataLoader", "loop"],
    file: "03_pytorch_training_loop.md",
  },
  {
    no: "04",
    title: "神经网络基础",
    summary: "从线性模型走向多层非线性函数，建立前向传播和反向传播直觉。",
    phase: "foundation",
    tags: ["NN", "ReLU", "表示学习"],
    file: "04_neural_network_basics.md",
  },
  {
    no: "05",
    title: "训练神经网络的基本功",
    summary: "比较学习率、batch size、dropout、weight decay 与梯度裁剪。",
    phase: "foundation",
    tags: ["optimizer", "dropout", "schedule"],
    file: "05_training_neural_networks.md",
  },
  {
    no: "06",
    title: "CNN 为什么适合图像",
    summary: "用 Fashion-MNIST 理解卷积核、参数共享、局部感受野和分类指标。",
    phase: "sequence",
    tags: ["CNN", "vision", "Fashion-MNIST"],
    file: "06_cnn_image_classification.md",
  },
  {
    no: "07",
    title: "RNN：让模型读序列",
    summary: "字符级名字分类里观察 hidden state、BPTT 和序列表示。",
    phase: "sequence",
    tags: ["RNN", "sequence", "BPTT"],
    file: "07_rnn_sequence_modeling.md",
  },
  {
    no: "08",
    title: "LSTM 和 GRU：门控记忆",
    summary: "用门控机制解释普通 RNN 面对长依赖时的困难。",
    phase: "sequence",
    tags: ["LSTM", "GRU", "gates"],
    file: "08_lstm_gru.md",
  },
  {
    no: "09",
    title: "Seq2Seq：从分类到生成",
    summary: "用 toy translation 跑通 encoder-decoder、teacher forcing 和自回归解码。",
    phase: "sequence",
    tags: ["Seq2Seq", "translation", "decoder"],
    file: "09_seq2seq_translation.md",
  },
  {
    no: "10",
    title: "Attention：让 decoder 不只看一个向量",
    summary: "在翻译任务中加入 alignment 和 context vector，缓解固定向量瓶颈。",
    phase: "sequence",
    tags: ["attention", "alignment", "context"],
    file: "10_attention_seq2seq.md",
  },
  {
    no: "11",
    title: "Self-Attention 从零实现",
    summary: "拆开 Q/K/V、padding mask、causal mask 和 attention weights。",
    phase: "transformer",
    tags: ["self-attention", "mask", "QKV"],
    file: "11_self_attention_from_scratch.md",
  },
  {
    no: "12",
    title: "Transformer Block",
    summary: "把 attention、残差连接、LayerNorm 和 MLP 组装成可复用 block。",
    phase: "transformer",
    tags: ["Transformer", "LayerNorm", "MLP"],
    file: "12_transformer_block.md",
  },
  {
    no: "13",
    title: "GPT 从零实现",
    summary: "实现 decoder-only mini-GPT：embedding、causal block 和 LM head。",
    phase: "transformer",
    tags: ["GPT", "decoder-only", "LM head"],
    file: "13_gpt_from_scratch.md",
  },
  {
    no: "14",
    title: "GPT 的训练数据、目标数据和损失函数",
    summary: "显微镜式检查 x/y 错位、逐 token cross entropy 和 causal mask。",
    phase: "transformer",
    tags: ["next-token", "loss", "causal"],
    file: "14_gpt_data_target_loss.md",
  },
  {
    no: "15",
    title: "小型 GPT 预训练工程化",
    summary: "训练、验证、保存 checkpoint，并用 temperature、top-k、top-p 生成文本。",
    phase: "transformer",
    tags: ["pretrain", "checkpoint", "sampling"],
    file: "15_mini_gpt_pretraining.md",
  },
  {
    no: "16",
    title: "SFT：让模型学会按指令回答",
    summary: "用 tiny instruction 数据解释 prompt/response 模板和 response-only loss。",
    phase: "post-training",
    tags: ["SFT", "instruction", "mask"],
    file: "16_sft_instruction_tuning.md",
  },
  {
    no: "17",
    title: "PPO：经典 RLHF 的核心优化器",
    summary: "用 bandit 实验讲清 ratio clipping、reward 和 KL 约束。",
    phase: "post-training",
    tags: ["PPO", "RLHF", "policy"],
    file: "17_ppo_rlhf.md",
  },
  {
    no: "18",
    title: "DPO：直接偏好优化",
    summary: "从 chosen/rejected 偏好样本出发，理解不显式训练 reward model 的偏好优化。",
    phase: "post-training",
    tags: ["DPO", "preference", "KL"],
    file: "18_dpo_preference_optimization.md",
  },
  {
    no: "19",
    title: "GRPO：面向可验证任务的组内相对优化",
    summary: "用 group sampling 和 rule reward 理解相对 advantage 的计算方式。",
    phase: "post-training",
    tags: ["GRPO", "RLVR", "group"],
    file: "19_grpo_verifiable_tasks.md",
  },
  {
    no: "20",
    title: "Capstone：Mini LLM Pipeline",
    summary: "把 mini-GPT 预训练、loss 检查、SFT、PPO、DPO、GRPO 串成学习流水线。",
    phase: "post-training",
    tags: ["capstone", "pipeline", "LLM"],
    file: "20_capstone_mini_llm_pipeline.md",
  },
];

const repoBase = "https://github.com/fengjikui/llm-tutor/blob/main/tutorials";
const list = document.querySelector("#chapter-list");
const tabs = document.querySelectorAll(".phase-tab");

function renderChapters(phase = "all") {
  const selected = phase === "all" ? chapters : chapters.filter((chapter) => chapter.phase === phase);
  list.innerHTML = selected
    .map(
      (chapter) => `
        <a class="chapter-row" href="${repoBase}/${chapter.file}">
          <span class="chapter-no">${chapter.no}</span>
          <span class="chapter-main">
            <h3>${chapter.title}</h3>
            <p>${chapter.summary}</p>
          </span>
          <span class="chapter-meta">
            ${chapter.tags.map((tag) => `<span>${tag}</span>`).join("")}
          </span>
        </a>
      `
    )
    .join("");
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("is-active"));
    tab.classList.add("is-active");
    renderChapters(tab.dataset.phase);
  });
});

renderChapters();

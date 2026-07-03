# 实验真实性重设计

这份文档记录 2026-07-03 对 `llm_tutor` 全部实验的一次重新审视。结论很明确：当前课程结构是好的，但实验层级需要重新校准。后续不能再让 tiny 数据承担“真实效果展示”的任务；tiny 实验只用于解释公式和机制，真正的章节主实验必须使用公开数据集、真实训练目标、可解释指标和可保存产物。

## 总结

当前实验可以分成三类：

| 类型 | 当前代表 | 处理方式 |
|---|---|---|
| 真实主实验 | Breast Cancer Wisconsin、Fashion-MNIST | 保留，并补充完整数据运行命令和指标预期 |
| 机制显微镜 | 手写梯度下降、Self-Attention shape trace、GPT loss inspect | 保留，但明确它们只解释机制 |
| toy 替身实验 | toy translation、tiny story corpus、tiny SFT、PPO/DPO/GRPO bandit | 保留为显微镜，同时新增真实主实验 |

后续每个章节都采用双轨设计：

1. `smoke`：默认参数小样本、短 epoch，用来确认代码能跑通。
2. `real`：真实公开数据、标准 train/val/test、明确指标，用来让读者看到模型确实解决了一个现实任务。

对于后训练章节，再加一条 `gpu` 轨道：使用小型开源模型加 LoRA/QLoRA，在 4090 级别显卡上跑真实 SFT、PPO、DPO、GRPO。

更准确地说，每章以后按这三个名字组织：

- `Microscope Lab`：机制显微镜。可以 tiny，但只能解释公式、shape、mask、loss，不能宣称真实效果。
- `Real Lab`：章节主实验。必须是真实公开数据、真实训练目标、真实 heldout 指标。
- `GPU Lab`：资源更重的真实实验。主要服务翻译、GPT 预训练和后训练章节。

当前仓库实情也要诚实记录：仓库没有留存真实训练产物，未看到 `runs/`、`checkpoints/`、`metrics.jsonl`、`summary.json` 或模型权重；只有 Fashion-MNIST 数据缓存和 Capstone 报告。所以下一轮实现不能只改文档，还要真正跑出可复盘的 artifacts。

## 统一验收标准

每个训练型实验至少要满足这些条件：

- 写清楚数据集来源、下载方式、许可证或数据卡入口。
- 写清楚训练目标，例如二分类交叉熵、next-token prediction、response-only SFT loss、DPO pairwise loss、GRPO rule reward。
- 至少报告一个 loss 和一个任务指标，例如 accuracy、F1、BLEU、perplexity、preference accuracy、solve rate、reward mean。
- 支持 `--output-dir`，保存 `config.json`、`metrics.jsonl`、`summary.json`、`stdout.log`。
- 对模型训练类任务保存 checkpoint 或 LoRA adapter，便于后续生成和复盘。
- 文档里必须区分“这个实验证明什么”和“这个实验不证明什么”。
- 每章要有三种命令：快速 smoke、真实 CPU/本地运行、推荐 GPU 运行。

## 成熟参考来源

这些参考可以作为实现时的设计来源，但不能直接照搬旧接口：

| 方向 | 参考 | 采用方式 |
|---|---|---|
| PyTorch 基础流程 | [PyTorch Quickstart: FashionMNIST](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | 复用完整训练工作流和数据加载思路 |
| 图像数据 | [TorchVision datasets](https://docs.pytorch.org/vision/stable/datasets.html) | MNIST、Fashion-MNIST、CIFAR-10 作为视觉章节主数据 |
| 从零翻译 | [PyTorch Transformer translation Colab](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/8cdd9a659f7d22e15eb4a689206e4b6b/translation_transformer.ipynb) | 参考 Multi30k/Transformer from scratch 任务形态，避免绑定脆弱旧 `torchtext` 入口 |
| 翻译微调 | [Hugging Face translation task guide](https://huggingface.co/docs/transformers/en/tasks/translation) | 作为预训练模型 fine-tuning 的成熟补充实验 |
| 真实翻译数据 | [OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) | 从中筛选短句，构造英法或英德翻译训练集 |
| 文本分类 | [AG News](https://huggingface.co/datasets/sh0416/ag_news)、[IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | RNN/LSTM/GRU/Transformer encoder 章节主数据 |
| 语言建模 | [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)、[nanoGPT](https://github.com/karpathy/nanoGPT) | Tiny Shakespeare 作为低门槛从零预训练，WikiText-2 作为更真实语料 |
| 后训练工具 | [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)、[PPOTrainer](https://huggingface.co/docs/trl/en/ppo_trainer)、[DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)、[GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer) | 先手写 loss 显微镜，再接入 TRL 真实小模型实验 |
| 指令数据 | [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)、[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | SFT 的真实 instruction 数据候选 |
| 偏好数据 | [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)、[UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | DPO 的真实 chosen/rejected 数据候选 |
| 可验证推理数据 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K) | GRPO 的真实数学/verifier 数据候选 |

当前依赖还没有 `datasets`、`transformers`、`trl`、`peft`、`sacrebleu`。实现时建议分层加入：

- 基础真实实验依赖：`datasets`、`sacrebleu`。
- 语言模型和后训练依赖：`transformers`、`accelerate`、`peft`、`trl`。
- GPU 轨道可以放进可选 extra，避免初学者第一步安装环境过重。

## 逐章实验重设计

### 01. 机器学习到底在学什么

当前状态：`load_breast_cancer()` 是真实离线数据，适合作为第一章。它足够小，不需要网络，也能让读者看到 accuracy、precision、recall、F1。

保留：

- Breast Cancer Wisconsin 二分类。
- PyTorch 线性分类器。
- train/val/test 指标。

升级：

- 增加一段“为什么这个数据集适合作为第一站”：特征是实数表格，标签是二分类，模型输出是一个概率。
- 后续可加 Adult Income 或 Bank Marketing 作为扩展，但第一章不需要变重。

### 02. 损失函数、梯度下降和优化器

当前状态：一维线性回归是机制显微镜，不应该替换。

保留：

- 手动 MSE。
- 手动梯度。
- PyTorch autograd 和 optimizer 对照。

升级：

- 增加一个真实回归扩展：California Housing。
- 目标是预测房价中位数，指标是 MSE/MAE。
- 正文强调：手写直线帮助理解梯度，不代表真实建模复杂度。

### 03. PyTorch 快速生存指南

当前状态：这一章是训练循环工具章，不需要单独追求更复杂数据。

保留：

- `Dataset`、`DataLoader`、`nn.Module`、`loss.backward()`、`optimizer.step()`。
- 从前两章复用表格分类。

升级：

- 增加“一个训练脚本的最低保存要求”：参数、指标、stdout、summary。
- 在正文中把 `--output-dir` 作为标准实验接口讲清楚。

### 04. 神经网络基础

当前状态：使用 Breast Cancer 能说明非线性，但感受不够强。

真实主实验：

- Fashion-MNIST MLP 分类。
- 对比线性模型、单隐藏层 MLP、两层 MLP。
- 指标：train loss、val accuracy、test accuracy。

保留显微镜：

- Breast Cancer 上的线性模型和小 MLP，用于解释 logits、激活函数和过拟合。

预期效果：

- 读者能看到 MLP 比线性模型更能拟合非线性模式。
- 文档不承诺某个固定分数，只给合理区间和影响因素。

### 05. 训练神经网络的基本功

当前状态：如果只在 Breast Cancer 上比较优化器，差异太小。

真实主实验：

- Fashion-MNIST 或 CIFAR-10 subset。
- 比较 SGD、Adam、学习率、dropout、weight decay、gradient clipping。
- 指标：训练曲线、验证集 accuracy、过拟合 gap。

保留：

- 小数据快速 smoke。

升级重点：

- 让读者看到学习率过大导致 loss 震荡，dropout/weight decay 改变 train-val gap。

### 06. CNN 为什么适合图像

当前状态：Fashion-MNIST 是真实数据，但默认只取 2048/512/512，效果展示偏弱。

真实主实验：

- 第一层：MNIST 手写数字识别，让读者快速看到 CNN 的威力。
- 第二层：Fashion-MNIST 全量或较大子集，展示更现实、更难的图像分类。
- 第三层：CIFAR-10 作为可选挑战，说明彩色图像更难，需要更深模型和数据增强。
- 关键对照：同一数据上跑 MLP 和 CNN，说明卷积的局部感受野、参数共享不是装饰，而是改变了模型对图像的归纳偏置。

训练目标：

- 10 类图像分类，交叉熵损失。

指标：

- train loss、val accuracy、test accuracy、混淆矩阵。
- 保存预测样例：正确样例和错误样例各若干张。
- 保存 per-class accuracy，帮助读者理解模型具体错在哪里。

预期体验：

- MNIST 应该给读者一个很直观的“模型真的学会看数字”的体验。
- Fashion-MNIST 用来说明真实图像分类并不总是轻松到接近满分。

### 07. RNN：让模型读序列

当前状态：名字分类是内部 tiny 数据，适合解释字符序列，但不适合做章节主实验。

真实主实验：

- AG News 新闻主题分类。
- 输入新闻标题/摘要，输出 World、Sports、Business、Sci/Tech。
- 模型：Embedding + vanilla RNN + classifier。

保留显微镜：

- 名字分类只用于讲字符输入、padding、hidden state。

指标：

- accuracy、macro F1、混淆矩阵。

为什么选 AG News：

- 数据量真实，类别清楚，文本较短，CPU 上也能用子集训练出可观察效果。

### 08. LSTM 和 GRU：门控记忆

当前状态：继续用名字分类无法体现门控记忆的价值。

真实主实验：

- AG News 或 IMDB。
- 同一数据、同一 embedding，比较 RNN、GRU、LSTM。

训练目标：

- 文本分类交叉熵。

指标：

- 收敛速度、验证集 accuracy、macro F1、梯度范数。

预期体验：

- 在较长文本上，GRU/LSTM 通常比普通 RNN 更稳定。
- 文档要说清楚：这不是证明 LSTM 永远更强，而是在当前任务上观察门控结构的作用。

### 09. Seq2Seq：从分类到生成

当前状态：toy English-French pairs 只有几十句，只能说明数据格式，不能说明翻译能力。

真实主实验：

- OPUS Books 英法或英德短句翻译。
- 对原始句对做过滤：最大源/目标长度、最小 token 频次、固定 train/val/test。
- 模型：RNN encoder-decoder，不带 attention。

训练目标：

- teacher forcing 下的目标 token 交叉熵。

指标：

- token accuracy、validation loss、BLEU/SacreBLEU、sample translations。

保留显微镜：

- toy translation 用于解释 BOS/EOS、padding、teacher forcing。
- 如果真实翻译训练太慢，可以增加一个“可验证序列转换”过渡实验，例如日期格式转换或字符串 reverse/copy。它仍不是翻译主实验，但比随手写 30 句 toy translation 更容易稳定观察生成模型学到的规则。

验收：

- 真实运行后必须能输出若干可读翻译样例，即使质量朴素。
- 文档要强调：无 attention 的 Seq2Seq 会在长句上明显吃力。

### 10. Attention：让 decoder 不只看一个向量

当前状态：同样受 toy translation 限制。

真实主实验：

- 复用第 09 章 OPUS Books 数据切分。
- 在 RNN Seq2Seq 上加入 Bahdanau 或 Luong attention。

训练目标：

- 目标 token 交叉熵。

指标：

- 与第 09 章对比 validation loss、BLEU、长句样例表现。
- 保存 attention heatmap。

预期体验：

- 读者能看到 attention 对较长句子的改进，至少能从热力图理解“decoder 每一步看源句哪里”。

### 11. Self-Attention 从零实现

当前状态：shape trace 是必要显微镜。

保留：

- Q/K/V、scaled dot-product、mask、多头拆分和拼回。

新增真实主实验：

- Transformer encoder 做 AG News 分类。
- 与第 07/08 章 RNN/GRU/LSTM 做对照。

指标：

- accuracy、macro F1、训练吞吐或每 epoch 时间。

预期体验：

- 读者不仅知道 self-attention 的 shape，还能看到它作为文本分类器如何工作。

### 12. Transformer Block

当前状态：block inspection 能解释残差、LayerNorm、FFN，但还没有真实任务承接。

真实主实验：

- 小型 Transformer encoder-decoder 做 OPUS Books 翻译。
- 或先做 Transformer encoder AG News 分类，再进入翻译。

训练目标：

- 翻译任务的目标 token 交叉熵。

指标：

- validation loss、BLEU、sample translations。

实现提醒：

- 可以参考 PyTorch 旧翻译教程的任务设计，但实现数据入口时优先用 Hugging Face `datasets` 和 `sacrebleu`，减少 `torchtext` 版本问题。

### 13. GPT 从零实现

当前状态：`TINY_STORY_CORPUS` 太小，只适合作为 loss 流程演示。

真实主实验：

- Tiny Shakespeare 字符级 GPT。
- WikiText-2 子词级或词级 GPT 作为进阶。
- 增加 `--text-file` 入口，让读者可以用自己的 `.txt` 或 `.md` 文件训练一个领域小模型。

训练目标：

- next-token prediction。

指标：

- train loss、val loss、perplexity、生成样例。

保留显微镜：

- `TINY_STORY_CORPUS` 只用于单元测试和极快 smoke。

预期体验：

- 读者能看到模型从胡乱字符逐渐生成类似训练语料风格的文本。
- 正文要说明：这不是现代 LLM 规模的预训练，只是 decoder-only LM 的最小真实闭环。

### 14. GPT 的训练数据、目标数据和损失函数

当前状态：这一章是机制章节，保留。

升级：

- 用 Tiny Shakespeare 或 WikiText 抽一小段真实文本演示：
  - `x = tokens[:-1]`
  - `y = tokens[1:]`
  - causal mask
  - per-token cross entropy

验收：

- 读者能手动检查一个 batch 的 input、target、logits、loss shape。

### 15. 小型 GPT 预训练工程化

当前状态：训练 tiny corpus 不能展示预训练效果。

真实主实验：

- Tiny Shakespeare char-level GPT，CPU 可跑。
- WikiText-2 GPT，建议 GPU 或较小配置。

训练目标：

- next-token prediction。

指标：

- val loss、perplexity、tokens/sec、checkpoint、生成样例。

新增要求：

- `generate_with_mini_gpt` 必须能够加载第 15 章 checkpoint。
- `summary.json` 写入最终样例，方便网站展示。

### 16. SFT：让模型学会按指令回答

当前状态：6 条 tiny instruction 只能说明 response-only mask，不能说明 SFT 的真实效果。

真实主实验：

- 数据：Dolly 15k、Alpaca、或一个许可证清楚的中文/英文指令数据子集。
- 模型：小型 pretrained causal LM，例如 Qwen 0.5B 级别模型；CPU smoke 可用 tiny GPT-2 只测试流程。
- 方法：LoRA/QLoRA + response-only label mask。

训练目标：

- prompt token mask 掉，只对 response token 做 next-token cross entropy。

指标：

- heldout SFT loss。
- 格式遵循率。
- 固定 eval prompts 的 before/after 对比。

边界说明：

- tiny GPT-2 smoke 不能证明 SFT 有真实能力提升。
- 真实效果展示需要 pretrained 模型和数千条以上指令样本，4090 更合适。
- 这一章要拆成 `Mask Microscope` 和 `Instruction Real Lab`：前者解释 response-only label mask，后者才展示指令遵循变化。

### 17. PPO：经典 RLHF 的核心优化器

当前状态：bandit 版本可以解释 ratio、clipping、KL，但不是语言模型 PPO。

真实主实验：

- 数据：IMDB prompt 或短文本续写 prompt。
- Policy：小型 GPT-2/Qwen 0.5B 级别模型。
- Reward：预训练 sentiment classifier 或规则 reward。
- 目标：让模型生成更正向的短文本，同时用 KL 限制偏离 reference model。

训练目标：

- PPO clipped objective + value loss + entropy/KL 监控。

指标：

- reward mean、KL、clip fraction、entropy、sample generations。

成熟性判断：

- 情感控制 PPO 是成熟教学案例，但比 SFT/DPO 更容易不稳定。
- 实现时必须给出保守默认超参、小 batch、短生成长度和 early stop。

保留显微镜：

- bandit PPO 继续解释 PPO 公式，不再作为真实效果实验。
- 文档标题和命令名应显式标注 `bandit` 或 `microscope`，避免读者误以为这就是语言模型 PPO。

### 18. DPO：直接偏好优化

当前状态：bandit 偏好实验只说明 DPO loss 方向。

真实主实验：

- 数据：HH-RLHF 或 UltraFeedback binarized 子集。
- 模型：SFT 后的小型 causal LM 或同系列 pretrained 模型。
- 方法：TRL DPOTrainer 或手写 DPO loss 的小规模版本。

训练目标：

- 提高 chosen response 相对 rejected response 的 log-prob margin，同时参考 reference model。

指标：

- preference accuracy。
- DPO loss。
- chosen/rejected margin。
- 固定 prompts 的生成对比。

预期体验：

- DPO 比 PPO 更适合作为第一条真实偏好优化实验，因为它不需要在线采样和 reward model。
- 后训练真实化建议先实现 SFT 和 DPO，再实现 PPO。PPO 的 rollout、value head、reward model 和 KL 控制更容易引入训练不稳定。

### 19. GRPO：面向可验证任务的组内相对优化

当前状态：bandit GRPO 只能解释 group-relative advantage。

真实主实验：

- 数据：GSM8K 子集或 DeepMath-103K 子集。
- 任务：生成数学答案，规则 verifier 从输出中抽取最终答案并判分。
- 模型：0.5B 到 1.5B 级别小模型，推荐 4090。

训练目标：

- 同一 prompt 采样多条回答，按组内 reward 标准化优势更新 policy。

指标：

- solve rate。
- reward mean。
- format correctness。
- response length。
- KL 或 reference divergence。

成熟性判断：

- 真实 GRPO 是最吃硬件和工程细节的一章。
- 必须保留 arithmetic toy 显微镜，同时把 GSM8K/DeepMath 放在 GPU real track。

### 20. Capstone：Mini LLM Pipeline

当前状态：Capstone 是学习流水线，不能暗示当前 tiny 阶段已经形成生产级 LLM 后训练。

重设计：

- `Teaching Pipeline`：
  - 保留当前 mini GPT、tiny SFT、bandit PPO/DPO/GRPO。
  - 目标是让读者统一理解数据、目标、loss/reward 和参数更新。
- `CPU Real Pipeline`：
  - MNIST/Fashion-MNIST。
  - AG News。
  - OPUS Books subset 翻译。
  - Tiny Shakespeare GPT。
- `GPU LLM Pipeline`：
  - SFT on instruction dataset。
  - DPO on preference dataset。
  - PPO sentiment control。
  - GRPO on GSM8K subset。

报告要求：

- 每个阶段都写入数据、目标、损失函数、指标、checkpoint 路径、失败样例。
- 明确哪些阶段共享 checkpoint，哪些只是并列教学实验。
- 如果标题继续叫 Mini LLM Pipeline，就必须至少提供一条连续链路：pretrain checkpoint -> SFT checkpoint -> DPO adapter。否则标题应改成 Teaching Pipeline。

## 实施阶段

### Phase 1: 视觉和文本分类真实化

目标：

- 增加 MNIST loader。
- Fashion-MNIST 支持 `--full-train` 或 `--train-limit none`。
- 增加 AG News 数据管道。
- RNN/GRU/LSTM/Transformer encoder 都能在 AG News 上训练和评估。

验收：

- CNN 有 MNIST 和 Fashion-MNIST 两条真实命令。
- RNN 章节不再以名字分类作为主实验。
- 所有训练脚本保存 artifacts。

### Phase 2: 翻译任务真实化

目标：

- 增加 OPUS Books 数据处理。
- 实现 RNN Seq2Seq、Attention Seq2Seq、Transformer translation 的同数据对比。
- 引入 `sacrebleu`。

验收：

- 每个模型都能保存 sample translations。
- 至少一个真实运行能在 heldout split 上输出可读翻译。

### Phase 3: GPT 预训练真实化

目标：

- Tiny Shakespeare 下载/缓存/切分。
- WikiText-2 数据入口。
- `train_mini_gpt` 区分 `--dataset tiny_story|tiny_shakespeare|wikitext2`。

验收：

- checkpoint 可被生成脚本加载。
- summary 中包含训练前后或不同 checkpoint 的生成样例。

### Phase 4: 后训练真实化

目标：

- 保留手写 SFT/PPO/DPO/GRPO loss 显微镜。
- 新增 TRL real track：
  - SFT instruction tuning。
  - DPO preference optimization。
  - PPO sentiment steering。
  - GRPO math verifier。

验收：

- CPU smoke 只承诺流程可跑。
- GPU real track 给出 4090 建议配置、显存预估、checkpoint/adapter 保存路径、评价脚本。
- 实现顺序建议是 SFT -> DPO -> GRPO -> PPO。PPO 最后做，因为它最容易训练不稳定，也最容易因为 reward/KL 设置不当让读者得到错误直觉。

## 硬件判断

| 实验 | CPU 是否足够 | 4090 是否值得 |
|---|---|---|
| Breast Cancer / California Housing | 足够 | 不需要 |
| MNIST / Fashion-MNIST | 足够，完整训练会慢一些 | 可加速，但非必须 |
| CIFAR-10 | 小模型 CPU 可跑，体验一般 | 值得 |
| AG News RNN/GRU/LSTM | 子集 CPU 可跑 | 完整对比更舒服 |
| OPUS Books RNN/Attention/Transformer 翻译 | 小子集 CPU 可跑 | 真实 BLEU 对比建议 GPU |
| Tiny Shakespeare GPT | CPU 可跑 | GPU 可更快观察效果 |
| WikiText-2 GPT | CPU 可 smoke | 推荐 GPU |
| SFT/DPO/PPO/GRPO 小模型 | CPU 只适合 tiny 流程 | 推荐 4090 |

所以，当前不需要马上租服务器来做设计审查和 Phase 1。等进入 Phase 4，或者要把翻译和 GPT 的真实训练跑出漂亮效果时，4090 会明显提升体验。

## 下一步执行顺序

建议先落 Phase 1：

1. 新增 MNIST/Fashion-MNIST 全量运行接口。
2. 把第 06 章改成 MNIST 入门 + Fashion-MNIST 真实分类。
3. 新增 AG News 数据管道。
4. 把第 07/08/11 章的主实验迁到 AG News。
5. 跑 smoke 和至少一次真实小规模训练，保存 artifacts。

之后再进入翻译。翻译是课程体验的关键节点，应该单独花一轮把数据处理、BLEU、sample translations、attention heatmap 打磨好。

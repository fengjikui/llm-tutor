# 从梯度下降到 GPT

这是一套面向大语言模型学习路线的动手教程。目标不是把机器学习史完整讲一遍，而是让读者用尽量短的路径建立必要直觉，并通过 PyTorch 实验真正理解：

- 模型如何通过损失函数和梯度下降学习参数；
- 神经网络为什么能表达复杂模式；
- CNN、RNN、Seq2Seq、Attention、Transformer 如何一步步演进；
- GPT 的结构、训练数据、目标数据和损失函数如何设计；
- SFT、PPO、DPO、GRPO 分别解决什么问题，以及如何写出最小可运行实验。

课程会采用“概念解释 + 代码实现 + 实验观察 + 常见坑”的形式。前半部分服务于建立地基，后半部分重点放在 Transformer、GPT 和 LLM post-training。

完整大纲见 [COURSE_OUTLINE.md](COURSE_OUTLINE.md)。

## 本地运行

本项目使用 `uv` 管理环境。系统 Python 可以是更新版本，但项目依赖按 Python 3.12 安装：

```bash
uv sync --python 3.12
```

如果已经有 `uv.lock`，后续复现实验可以使用：

```bash
uv sync --locked
```

当前 smoke test：

```bash
bash scripts/smoke_test.sh
```

第一次运行 CNN 相关实验时，Fashion-MNIST 会下载到 `data/vision`；之后会复用本地缓存。

## 硬件说明

默认实验按本地 CPU 可跑设计。脚本会在可用时优先使用 CUDA；没有 NVIDIA GPU 时会自动使用 CPU。当前版本没有强依赖 4090 这类显卡。

如果只是学习教程主线、跑 smoke test、训练 mini-GPT 的 tiny 配置，普通笔记本即可。显卡更适合后续扩展实验：更大图像数据集、更长 mini-GPT 训练、更大的 instruction 数据、更真实的 PPO/DPO/GRPO 或 Hugging Face/TRL 实验。

单独运行实验：

```bash
uv run python -m llm_tutor.experiments.manual_gradient_descent
uv run python -m llm_tutor.experiments.train_linear_classifier --epochs 20
uv run python -m llm_tutor.experiments.train_neural_network_basics --epochs 30
```

教程正文从 [tutorials/README.md](tutorials/README.md) 开始阅读。

## 专栏网站

仓库里已经加入一个个人网站专栏第一版，入口在 [site/index.html](site/index.html)。
它把 20 章教程重新组织成 4 个阶段，并把精选章节、实验命令、代码入口和参考审美放在同一个页面里。

本地直接打开 HTML 即可预览；如果希望用本地服务：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000/site/`。

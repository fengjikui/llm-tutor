---
title: "22. 多模态小系列：从 ViT、CLIP、ClipCap 到现代 VLM/VLA"
status: "已完成"
summary: "理解图像如何变成 token，CLIP 如何做图文对齐，ClipCap 如何把 CLIP 图像表示映射成语言模型 prefix，并一路过渡到 Qwen3-VL、Qwen3.5-Omni、Qwen-VLA 和 encoder-free multimodal。"
---

# 22. 多模态小系列：从 ViT、CLIP、ClipCap 到现代 VLM/VLA

## 本章学习契约

- 新增概念：ViT patch token、图文对比学习、ClipCap/prefix mapping、captioning loss、Q-Former、visual projector、visual instruction tuning、动态分辨率、VLA、encoder-free multimodal。
- 本章要解决：读者看到现代多模态模型时，知道“图片怎么进入 LLM”。
- 本章不验证：不训练 VLM，不跑图像问答 benchmark。
- 看完重点：多模态模型不是把图片塞进 tokenizer，而是先把图片编码成视觉 token，再用投影/交叉注意力/对齐训练接到语言空间。

如果前 21 章讲的是“文本 token 进入 Transformer”，多模态主线就是：

```text
image
-> patches / vision encoder
-> visual tokens
-> projector / cross-attention / Q-Former
-> language model token space
-> text generation or contrastive retrieval
```

先把术语写正式：

| 缩写 | 正式含义 | 本章怎么用 |
|---|---|---|
| ViT | Vision Transformer | 把图片切成 patch token 的视觉骨架 |
| CLIP | Contrastive Language-Image Pre-training | 图文对齐模型 |
| ClipCap | CLIP Prefix for Image Captioning | 把 CLIP 图像表示映射成语言模型 prefix |
| VLM | Vision-Language Model | 看图/视频并输出文字 |
| MLLM | Multimodal Large Language Model | 更广义的多模态大模型 |
| VLA | Vision-Language-Action Model | 看见世界、理解指令并输出动作 |

## ViT：把图片切成 patch，再当序列处理

CNN 看图像时用卷积核滑动。ViT 的做法更接近 Transformer：把图片切成固定大小 patch，每个 patch 展平成向量，再投影成 token。

最小 patchify 示意：

```python
import torch

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    # images: [batch, channels, height, width]
    b, c, h, w = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(b, -1, c * patch_size * patch_size)
```

ViT 让“图片”变成“视觉 token 序列”。后续 CLIP、BLIP、LLaVA、Qwen-VL 这类模型都离不开这个直觉。

## CLIP：图片和文字进入同一个语义空间

CLIP 的核心训练目标很直接：一批图片和一批文字描述放在一起，模型要判断哪张图配哪句话。

```text
image_encoder(image) -> image_embedding
text_encoder(text)   -> text_embedding
相似度矩阵: image_embedding @ text_embedding.T
目标: 对角线匹配最高
```

最小 loss 形状：

```python
import torch
from torch.nn import functional as F

image_emb = F.normalize(image_emb, dim=-1)
text_emb = F.normalize(text_emb, dim=-1)
logits = image_emb @ text_emb.T
labels = torch.arange(logits.shape[0], device=logits.device)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

CLIP 的意义在于：它让文字可以指挥视觉分类。比如不训练一个猫狗分类器，只写 prompts：

```text
a photo of a cat
a photo of a dog
```

然后看图片 embedding 更接近哪个文本 embedding。

## ClipCap：把 CLIP 图像表示变成语言模型 prefix

CLIP 解决的是“图和文是否对齐”。ClipCap 往前走了一步：既然 CLIP 的图像向量已经有很强的语义信息，那能不能把它变成语言模型的 prefix，让 GPT-2 这类 decoder 继续生成 caption？

```text
image
-> CLIP image encoder
-> mapping network
-> prefix embeddings
-> GPT-2 / decoder LM
-> caption tokens
```

这在现代 VLM 发展里很值得作为“奠基节点”理解。它不是今天最强的 VLM 架构，但它把两个关键思想连起来了：

1. CLIP 提供图文对齐过的视觉语义表示。
2. 语言模型不需要从零学语言，只要学会如何接收视觉 prefix。

最小结构示意：

```python
import torch
from torch import nn

clip_image_embedding = clip_image_encoder(image)  # [batch, clip_dim]
prefix = mapper(clip_image_embedding)             # [batch, prefix_len * lm_dim]
prefix = prefix.view(batch, prefix_len, lm_dim)
inputs_embeds = torch.cat([prefix, text_embeddings], dim=1)
caption_logits = language_model(inputs_embeds=inputs_embeds)
```

训练目标回到我们熟悉的 next-token prediction：

```text
输入: image + "a photo of"
目标: "a dog running on the beach"
loss: 逐 token cross entropy
```

你说它是现代视觉语言模型的一个奠基技术，我会这样校准：

- 如果说最底层奠基，ViT 和 CLIP 更基础。
- 如果说“把 CLIP 视觉表示接到生成式语言模型上”，ClipCap 是非常关键、非常教学友好的早期方案。
- 如果说今天的通用 VLM，BLIP-2、Flamingo、LLaVA、Qwen-VL 这类模型在 ClipCap 的方向上继续发展：不只用一个全局 embedding，而是使用视觉 token、Q-Former、cross-attention 或 projector 来保留更多细节。

CoCa 可以作为旁支补充：它把 CLIP 风格的 contrastive loss 和 captioning loss 结合在同一个图文基础模型里。但本教程主线现在按 “ViT -> CLIP -> ClipCap -> BLIP/LLaVA/Qwen-VL” 来讲。

## BLIP/BLIP-2：用桥接模块降低训练成本

端到端训练一个大视觉模型加大语言模型非常贵。BLIP-2 的思路是冻结已有 image encoder 和 LLM，中间训练一个轻量 Q-Former 来桥接视觉和语言。

```text
frozen image encoder
-> visual features
-> Q-Former queries
-> language model
```

这给后来的多模态模型提供了一个常见模式：不要每次都从零训练所有模块，而是用一个相对小的连接器把强视觉 encoder 和强语言模型接起来。

## LLaVA：把视觉 token 投到 LLM 的词向量空间

LLaVA 风格的结构可以用一句话概括：

```text
CLIP ViT 提取视觉特征 -> MLP projector -> LLM embedding space -> instruction tuning
```

最小 projector 示意：

```python
from torch import nn

projector = nn.Sequential(
    nn.Linear(vision_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim),
)

visual_tokens_for_llm = projector(clip_visual_tokens)
```

这些 visual tokens 会被拼到文本 token 前后，LLM 在 causal attention 中一起处理。训练时通常分两步：

1. 对齐阶段：让 projector 学会把视觉特征放进语言模型能理解的空间。
2. 指令微调：用图文问答数据让模型学会按用户问题回答。

## 现代 VLM：Qwen3-VL、Qwen3.5-Omni 和动态多模态

现代多模态模型不只看普通照片，还要看截图、表格、PDF、长图、视频和 UI。这里会出现几个新组件：

- 动态分辨率：不同大小图片变成不同数量视觉 token，避免统一 resize 损失细节。
- Window Attention：高分辨率视觉 token 太多时，用局部窗口降低开销。
- MRoPE：把 RoPE 扩展到文字、图像二维坐标、视频时间轴。
- OCR/文档能力：模型需要保留局部细节和布局，而不是只提取全局语义。
- grounding：输出 bounding box、point 或区域定位。

Qwen2.5-VL、Qwen3-VL、Qwen3.5-Omni 这类模型把这些问题系统化了：图片、视频、音频不再只是“一个额外输入”，而是带空间、时间、分辨率结构的 token 序列。

可以按这个阶梯理解：

| 阶段 | 代表模型 | 关键变化 |
|---|---|---|
| 图文对齐 | CLIP | 图像和文字进入共享语义空间 |
| 图生文 | ClipCap | CLIP 图像 embedding 通过 mapper 变成 LM prefix |
| 视觉接入 LLM | BLIP-2、LLaVA | Q-Former 或 projector 把视觉 token 接到语言模型 |
| 长上下文 VLM | Qwen3-VL | 交错文本/图像/视频上下文、增强 MRoPE、DeepStack 多层视觉特征 |
| 全模态交互 | Qwen3.5-Omni | 文本、视觉、音频、视频共同进入 omni 模型，支持更长音视频理解和语音交互 |
| 具身动作 | Qwen-VLA | 在 VLM 之上接动作/轨迹生成，面向机器人操作、导航、轨迹预测 |

这里的重点不是背模型名，而是看能力边界如何移动：

```text
CLIP:    图文相似吗？
ClipCap: 看图能说一句话吗？
LLaVA:   看图能按指令回答吗？
Qwen3-VL: 图、视频、长文档能混在一个上下文里推理吗？
Qwen-VLA: 看见环境后能输出动作吗？
```

## Encoder-Free Multimodal：Google Gemma 4 这条新分支

大多数 VLM 都有单独视觉 encoder：

```text
image -> vision encoder -> connector -> LLM
```

Google 在 Gemma 4 12B 上公开强调了一条 encoder-free multimodal 路线：不再使用单独的多模态 encoder，视觉和音频输入直接进入 LLM backbone。这个方向的动机很清楚：减少外接模块和部署复杂度，让多模态能力更像语言模型的原生输入能力。

教学上可以这样理解：

| 传统 VLM | Encoder-free multimodal |
|---|---|
| 视觉 encoder 先提特征 | 输入侧直接把多模态表示送入 LLM backbone |
| connector/projector 很关键 | backbone 自身承担更多跨模态建模 |
| 视觉能力受 vision tower 影响很大 | 训练配方和统一 backbone 更重要 |

这不代表视觉 encoder 路线马上过时。Qwen3-VL/Qwen3.5-Omni、PaliGemma、LLaVA 这些路线仍然非常重要。更准确的判断是：现代 VLM 正在从“外接视觉塔”分化出“原生多模态 backbone”和“VLA 动作模型”两条新方向。

## 看 VLM 代码时的五个问题

1. 视觉 encoder 是 ViT、ConvNet，还是混合结构？
2. 图片 token 如何进入语言模型：MLP projector、Q-Former、cross-attention、DeepStack，还是原生统一 backbone？
3. 训练目标是什么：contrastive、ClipCap captioning、VQA、instruction tuning、OCR、grounding、action prediction？
4. 位置信息如何表示：固定 resize 后的位置、dynamic resolution、MRoPE、absolute time encoding？
5. 如果是 VLA，动作空间是什么：离散 action token、连续轨迹、还是 DiT/flow-matching action decoder？

## 参考

- [An Image is Worth 16x16 Words: Vision Transformer](https://arxiv.org/abs/2010.11929)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
- [Official ClipCap implementation](https://github.com/rmokady/CLIP_prefix_caption)
- [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)
- [BLIP](https://arxiv.org/abs/2201.12086)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [Qwen2-VL](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)
- [Qwen-VLA Technical Report](https://arxiv.org/abs/2605.30280)
- [Qwen-VLA official repository](https://github.com/QwenLM/Qwen-VLA)
- [Google Gemma 4 12B announcement](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/)
- [GPT-4V System Card PDF](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

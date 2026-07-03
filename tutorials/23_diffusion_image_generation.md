---
title: "23. Diffusion 生图：从加噪去噪到 Stable Diffusion 和 DiT"
status: "已完成"
summary: "用最少数学理解扩散模型：前向加噪、反向去噪、噪声预测、latent diffusion、classifier-free guidance、Stable Diffusion 与 Diffusion Transformer。"
---

# 23. Diffusion 生图：从加噪去噪到 Stable Diffusion 和 DiT

## 本章学习契约

- 新增概念：DDPM、forward noising、reverse denoising、noise prediction、latent diffusion、text conditioning、classifier-free guidance、DiT。
- 本章要解决：读者知道生图模型为什么不是 GPT 式从左到右生成，而是从噪声一步步去噪。
- 本章不验证：不训练生图模型，不跑 GPU 采样实验。
- 看完重点：Stable Diffusion 这类系统由 VAE、text encoder、denoiser、scheduler 组成，不只是一个大 U-Net。

语言模型生成文本时是自回归的：

```text
token_1 -> token_2 -> token_3 -> ...
```

Diffusion 生图模型的直觉完全不同：

```text
纯噪声图
-> 去掉一点噪声
-> 再去掉一点噪声
-> ...
-> 清晰图像
```

## DDPM：训练模型预测噪声

DDPM 有两个过程：

| 过程 | 是否学习 | 做什么 |
|---|---|---|
| forward process | 不学习 | 给真实图像逐步加噪 |
| reverse process | 学习 | 从噪声里一步步还原图像 |

训练时，我们随机选一个时间步 `t`，把干净图像 `x0` 加噪得到 `xt`，然后让模型预测加进去的噪声 `epsilon`。

最小形状示意：

```python
import torch
from torch.nn import functional as F

def add_noise(x0, noise, alpha_bar_t):
    return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

x0 = torch.randn(8, 3, 64, 64)
noise = torch.randn_like(x0)
xt = add_noise(x0, noise, alpha_bar_t=torch.tensor(0.3))
pred_noise = denoiser(xt, timestep=t)
loss = F.mse_loss(pred_noise, noise)
```

这和 GPT 的 next-token loss 很不一样：

- GPT 预测下一个离散 token。
- Diffusion 通常预测连续噪声、干净样本或 velocity。

## Sampling：从噪声走回图像

采样时先从纯噪声开始：

```text
x_T ~ Normal(0, I)
```

然后按 scheduler 的规则从 `T` 走到 `0`：

```python
x = torch.randn(batch, channels, height, width)
for t in reversed(timesteps):
    pred_noise = denoiser(x, t, condition)
    x = scheduler_step(x, pred_noise, t)
```

不同 scheduler 的区别在于：每一步怎么根据模型预测更新 `x`。这就是为什么同一个模型可以搭配 DDIM、Euler、DPM-Solver 等不同采样器。

## Latent Diffusion：不在像素空间里去噪

高分辨率图片的像素空间很大。Latent Diffusion 的关键想法是：先用 VAE 把图像压到 latent 空间，在更小的 latent 上做 diffusion，最后再 decode 回图片。

```text
image
-> VAE encoder
-> latent
-> diffusion denoising in latent space
-> VAE decoder
-> image
```

Stable Diffusion 属于 text-to-image latent diffusion。它的核心组件通常是：

| 组件 | 作用 |
|---|---|
| Text Encoder | 把 prompt 编成文本条件 |
| VAE Encoder/Decoder | 在图像和 latent 之间转换 |
| Denoiser U-Net / Transformer | 在 latent 空间预测噪声 |
| Scheduler | 控制采样时间步和更新公式 |

## Prompt 如何影响图像：Cross-Attention

Text-to-image 模型需要让图像 latent “看见”文本。常见方式是 cross-attention：

```text
query: image latent tokens
key/value: text tokens
```

也就是说，去噪网络在每一步不仅看当前 noisy latent，还会读 prompt 的文本表示。prompt 不是最后贴标签，而是参与每一步去噪。

## Classifier-Free Guidance：让 prompt 更有力

CFG 的直觉是：同一步去噪做两次预测，一次带 prompt，一次不带 prompt，然后把差值放大。

```python
eps_uncond = denoiser(x, t, empty_prompt)
eps_cond = denoiser(x, t, prompt)
eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

`guidance_scale` 太小，图像可能不听 prompt；太大，图像可能过饱和、变形或失去多样性。

## DiT：把 U-Net 换成 Transformer

Stable Diffusion 早期主干多是 U-Net。DiT 的方向是：在 latent patch 上使用 Transformer 作为 denoiser。

```text
latent image
-> patch tokens
-> Transformer blocks with timestep/class/text conditioning
-> predicted noise
```

这和 ViT、多模态、LLM 主线又接上了：图片 latent 也可以变成 token 序列，Transformer 不只用于文本。

## 看生图模型代码时的五个问题

1. 模型在像素空间还是 latent 空间扩散？
2. denoiser 是 U-Net、Transformer，还是混合结构？
3. 模型预测的是 noise、x0，还是 velocity？
4. 文本条件通过 cross-attention、AdaLN，还是其他条件注入方式进入？
5. scheduler 和 guidance scale 如何影响速度、质量和稳定性？

## 和 LLM 的关系

Diffusion 和 LLM 的目标不同，但工程问题很像：

| LLM | Diffusion |
|---|---|
| token embedding | image/latent patch embedding |
| causal self-attention | spatial/latent attention |
| next-token loss | noise prediction loss |
| sampler: top-k/top-p/temperature | sampler: DDIM/Euler/DPM-Solver |
| prompt conditioning | text encoder + cross-attention |

如果读者已经理解 Transformer 和 token，这一章的关键就是把“token 序列”泛化到“图像 patch/latent token 序列”。

## 参考

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Stable Diffusion pipelines in Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/en/index)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Official DiT repository](https://github.com/facebookresearch/dit)

---
title: "05. 训练神经网络的基本功"
status: "已完成"
summary: "比较学习率、优化器、weight decay 和 Dropout，理解训练策略如何影响同一个神经网络。"
---

# 05. 训练神经网络的基本功

## 本章学习契约

- 新增概念：训练稳定性、学习率、batch size、正则化、过拟合和欠拟合。
- 实验要验证：同一个模型结构下，训练配置会显著影响 loss 曲线和泛化表现。
- 实验不验证：它不是调参秘籍，也不能保证某组超参数在所有任务上最好。
- 跑完重点看：train/val 曲线是否一起下降，是否出现训练集好但验证集差，以及不同配置的最终测试指标。

到目前为止，我们已经知道神经网络的基本训练闭环：

```text
前向传播 -> 计算 loss -> 反向传播 -> 优化器更新参数
```

但真实训练时，模型结构只是故事的一半。另一半是训练策略：学习率怎么选、优化器用什么、要不要正则化、模型是否已经过拟合。

这一章不引入新任务，而是继续使用前面的表格分类数据。这样做的好处是：数据和主体网络结构不变，我们更容易观察训练策略本身的影响。少数配置会加入训练期正则化层，例如 Dropout；这会改变训练时的模型行为，但不是换一个完全不同的任务。

## 学习率

学习率决定每次参数更新走多远：

```text
参数 = 参数 - learning_rate * 梯度
```

学习率太小，loss 下降很慢；学习率太大，训练可能震荡，甚至越训越差。

不要把学习率理解成“越大越快”。它更像步子大小：步子太小走得慢，步子太大可能跨过最低点。

## 优化器

SGD 会沿着当前梯度方向更新参数。加上 momentum 后，它会带一点“惯性”，让更新方向不至于每个 batch 都剧烈摇摆。

Adam 会为不同参数自适应调整更新幅度。很多深度学习任务里，Adam 是一个非常好用的默认选择，尤其适合快速跑通实验。

这不表示 Adam 永远更好。优化器选择要结合任务、学习率、batch size 和训练时长一起看。

初学时可以先把三者这样区分：

| 优化器 | 直觉 | 你可能看到的现象 |
|---|---|---|
| SGD | 完全按当前梯度走 | 简单、稳定，但可能慢 |
| SGD + momentum | 带一点历史方向 | 比普通 SGD 少一点抖动 |
| Adam | 给不同参数自适应步长 | 常常更快跑通，但需要注意泛化和学习率 |

本章实验不是为了宣布谁最好，而是让你第一次看到“训练策略会改变曲线形状”。

## Weight Decay

Weight decay 会惩罚过大的参数，让模型不要把某些权重推得太极端。它是一种常见正则化方式。

直觉上，它在提醒模型：

```text
如果两个解在训练集上差不多好，优先选择参数更温和的那个。
```

## Dropout

Dropout 会在训练时随机把一部分隐藏单元置零。它迫使网络不要过度依赖某几个神经元。PyTorch 的 Dropout 还会对保留下来的激活做缩放，让整体激活幅度在训练和评估之间更接近。

注意：Dropout 只在 `model.train()` 阶段生效；到了 `model.eval()` 阶段，它会被关闭。这就是上一章强调训练模式和评估模式的原因。

Weight decay 和 Dropout 都叫正则化，但它们作用的位置不同：

| 方法 | 作用位置 | 直觉 |
|---|---|---|
| Weight decay | loss/参数更新 | 惩罚过大的权重 |
| Dropout | 训练期前向传播 | 随机屏蔽隐藏单元，减少互相依赖 |

后面训练语言模型时，你还会看到 dropout 出现在 embedding、attention 或 MLP 附近；weight decay 则通常由优化器配置控制。

## 运行实验

```bash
uv run python -m llm_tutor.experiments.compare_training_strategies --epochs 12
```

脚本会比较几种配置：

- `adam_lr_1e-3`
- `adam_lr_1e-2`
- `sgd_momentum_lr_1e-2`
- `adam_weight_decay`
- `adam_dropout`
- `adam_weight_decay_dropout`

脚本先输出 validation summary：

```text
validation summary
strategy                       val_loss  val_acc  malignant_val_f1
adam_lr_1e-3                   ...
adam_lr_1e-2                   ...
sgd_momentum_lr_1e-2           ...
adam_weight_decay              ...
adam_dropout                   ...
adam_weight_decay_dropout      ...
```

然后按 validation loss 选择一个配置，最后只对这个配置报告一次 test 指标。

## 怎么看结果

比较策略时先看 `val_loss`，它比训练 loss 更能反映模型有没有泛化。不要用 test set 来挑配置；test set 应该留到最后，只用于报告一次最终效果。

如果某个配置训练 loss 很低，但验证 loss 不好，说明它可能过拟合。  
如果某个配置 loss 一直降不下去，优先怀疑学习率、优化器或输入预处理。

注意，单次实验只能提供线索，不能证明严格因果。比如 `adam_weight_decay_dropout` 表现更好时，你不能立刻断言“一定是 Dropout 起作用”，因为它同时改变了 weight decay 和 Dropout。脚本把 `adam_weight_decay`、`adam_dropout` 也单独列出来，就是为了帮助你观察组合配置和单独配置的差异。

## 这个实验能说明什么

- 同一个模型，训练策略不同，结果也会不同。
- Adam 通常更容易快速跑通，但不是理论上永远最优。
- 正则化的价值不一定在小数据、短训练里立刻显现，但它是后续大模型训练的重要基础。
- 模型选择应该主要依赖 validation set，test set 留给最终报告。

## 这个实验不能证明什么

- 它不能证明某个优化器在所有任务上最好。
- 它不能证明 Dropout 一定提高指标。
- 它不能证明测试集最高的配置就适合真实业务。
- 它不能从一次小实验里严格分离所有因素的因果影响。

## 常见失败

- 指标波动：小数据集上很正常，可以固定随机种子并重复多次。
- Dropout 配置短期效果变差：正则化会让训练更难一些，但可能改善泛化。
- SGD 看起来慢：可以增加 epoch，或尝试不同学习率。

上一章：[04. 神经网络基础](04_neural_network_basics.md)  
下一章：[06. CNN 为什么适合图像](06_cnn_image_classification.md)

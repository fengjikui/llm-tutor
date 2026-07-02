from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapstoneStage:
    name: str
    goal: str
    command: tuple[str, ...]


def build_capstone_stages(*, checkpoint_path: Path) -> list[CapstoneStage]:
    checkpoint = str(checkpoint_path)
    return [
        CapstoneStage(
            name="pretrain",
            goal="训练一个 tiny causal LM，并保存 checkpoint。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.train_mini_gpt",
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--block-size",
                "32",
                "--embed-dim",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--sample-len",
                "20",
                "--checkpoint-path",
                checkpoint,
            ),
        ),
        CapstoneStage(
            name="generate",
            goal="从 checkpoint 加载 mini-GPT，并用采样参数生成文本。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.generate_with_mini_gpt",
                "--checkpoint-path",
                checkpoint,
                "--max-new-tokens",
                "20",
                "--top-k",
                "5",
                "--top-p",
                "0.9",
            ),
        ),
        CapstoneStage(
            name="inspect-loss",
            goal="检查 GPT 的 x/y、逐 token loss 和 causal mask。",
            command=("uv", "run", "python", "-m", "llm_tutor.experiments.inspect_gpt_loss"),
        ),
        CapstoneStage(
            name="sft",
            goal="用 response-only label mask 做 tiny SFT。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.train_sft",
                "--epochs",
                "2",
                "--batch-size",
                "3",
                "--embed-dim",
                "32",
                "--num-layers",
                "1",
            ),
        ),
        CapstoneStage(
            name="ppo",
            goal="用 bandit 实验观察 PPO ratio、clipping 和 KL。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.train_ppo_bandit",
                "--epochs",
                "3",
            ),
        ),
        CapstoneStage(
            name="dpo",
            goal="用 chosen/rejected 偏好对训练 tiny DPO policy。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.train_dpo_bandit",
                "--epochs",
                "3",
            ),
        ),
        CapstoneStage(
            name="grpo",
            goal="用 group sampling 和规则 reward 训练 tiny GRPO policy。",
            command=(
                "uv",
                "run",
                "python",
                "-m",
                "llm_tutor.experiments.train_grpo_bandit",
                "--epochs",
                "3",
                "--group-size",
                "4",
            ),
        ),
    ]


def format_stage_command(stage: CapstoneStage) -> str:
    return shlex.join(stage.command)

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from llm_tutor.capstone.pipeline import build_capstone_stages
from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="第 20 章：运行或预览 Mini LLM Pipeline")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行 pipeline；默认只打印命令。",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    add_artifact_args(parser)
    args = parser.parse_args(argv)

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="run_capstone_pipeline",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    if args.checkpoint_path is not None:
        stages = build_capstone_stages(checkpoint_path=args.checkpoint_path)
        _run_or_print(stages, execute=args.execute, artifacts=artifacts)
        return

    if not args.execute:
        stages = build_capstone_stages(checkpoint_path=Path("checkpoints/capstone_mini_gpt.pt"))
        _run_or_print(stages, execute=False, artifacts=artifacts)
        return

    if artifacts.enabled:
        checkpoint_path = artifacts.path("capstone_mini_gpt.pt")
        stages = build_capstone_stages(checkpoint_path=checkpoint_path)
        _run_or_print(stages, execute=True, artifacts=artifacts)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "mini_gpt.pt"
        stages = build_capstone_stages(checkpoint_path=checkpoint_path)
        _run_or_print(stages, execute=True, artifacts=artifacts)


def _run_or_print(stages, *, execute: bool, artifacts: ExperimentArtifacts) -> None:
    mode = "execute" if execute else "dry-run"
    print(f"capstone_mode={mode} stages={len(stages)}")
    records = []
    for index, stage in enumerate(stages, start=1):
        command = stage.command
        stage_dir = None
        if artifacts.enabled:
            stage_dir = artifacts.path(f"{index:02d}_{stage.name.replace('-', '_')}")
            if _supports_output_dir(command):
                command = (*command, "--output-dir", str(stage_dir))

        print(f"\n[{index:02d}] {stage.name}")
        print(f"goal={stage.goal}")
        print(f"command={shlex.join(command)}")
        record = {
            "index": index,
            "name": stage.name,
            "goal": stage.goal,
            "command": list(command),
            "output_dir": stage_dir,
            "exit_code": None,
        }
        if execute:
            if artifacts.enabled:
                assert stage_dir is not None
                stage_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, text=True, capture_output=True, check=False)
                stage_dir.joinpath("stdout.log").write_text(result.stdout, encoding="utf-8")
                stage_dir.joinpath("stderr.log").write_text(result.stderr, encoding="utf-8")
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
                record["exit_code"] = result.returncode
                records.append(record)
                if result.returncode != 0:
                    artifacts.write_summary({"mode": mode, "stages": records})
                    result.check_returncode()
            else:
                subprocess.run(command, check=True)
                record["exit_code"] = 0
                records.append(record)
        else:
            records.append(record)
    artifacts.write_summary({"mode": mode, "stages": records})


def _supports_output_dir(command: tuple[str, ...]) -> bool:
    supported_modules = {
        "llm_tutor.experiments.train_mini_gpt",
        "llm_tutor.experiments.generate_with_mini_gpt",
        "llm_tutor.experiments.train_sft",
        "llm_tutor.experiments.train_ppo_bandit",
        "llm_tutor.experiments.train_dpo_bandit",
        "llm_tutor.experiments.train_grpo_bandit",
    }
    return any(item in supported_modules for item in command)


if __name__ == "__main__":
    main()

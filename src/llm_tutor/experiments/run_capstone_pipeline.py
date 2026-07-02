from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from llm_tutor.capstone.pipeline import build_capstone_stages, format_stage_command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="第 20 章：运行或预览 Mini LLM Pipeline")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行 pipeline；默认只打印命令。",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.checkpoint_path is not None:
        stages = build_capstone_stages(checkpoint_path=args.checkpoint_path)
        _run_or_print(stages, execute=args.execute)
        return

    if not args.execute:
        stages = build_capstone_stages(checkpoint_path=Path("checkpoints/capstone_mini_gpt.pt"))
        _run_or_print(stages, execute=False)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "mini_gpt.pt"
        stages = build_capstone_stages(checkpoint_path=checkpoint_path)
        _run_or_print(stages, execute=True)


def _run_or_print(stages, *, execute: bool) -> None:
    mode = "execute" if execute else "dry-run"
    print(f"capstone_mode={mode} stages={len(stages)}")
    for index, stage in enumerate(stages, start=1):
        print(f"\n[{index:02d}] {stage.name}")
        print(f"goal={stage.goal}")
        print(f"command={format_stage_command(stage)}")
        if execute:
            subprocess.run(stage.command, check=True)


if __name__ == "__main__":
    main()

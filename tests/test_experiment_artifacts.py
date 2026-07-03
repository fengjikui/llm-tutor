from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from llm_tutor.experiments.artifacts import ExperimentArtifacts
from llm_tutor.experiments.run_capstone_pipeline import main as run_capstone_main


def test_artifact_writer_records_jsonl_and_stdout(tmp_path: Path) -> None:
    artifacts = ExperimentArtifacts.create(
        tmp_path,
        experiment_name="unit_test",
        config={"epochs": 1, "path": tmp_path / "model.pt"},
    )

    with artifacts.capture_stdout():
        print("hello artifact")
        artifacts.append_metric({"epoch": 1, "loss": 0.5})
        artifacts.write_summary({"ok": True})

    config = json.loads(tmp_path.joinpath("config.json").read_text(encoding="utf-8"))
    metric = json.loads(tmp_path.joinpath("metrics.jsonl").read_text(encoding="utf-8"))
    summary = json.loads(tmp_path.joinpath("summary.json").read_text(encoding="utf-8"))

    assert config["experiment"] == "unit_test"
    assert config["config"]["path"].endswith("model.pt")
    assert metric == {"epoch": 1, "loss": 0.5}
    assert summary == {"ok": True}
    assert "hello artifact" in tmp_path.joinpath("stdout.log").read_text(encoding="utf-8")


def test_ppo_cli_writes_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_tutor.experiments.train_ppo_bandit",
            "--epochs",
            "1",
            "--ppo-epochs",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert "artifacts_dir=" in result.stdout
    assert tmp_path.joinpath("config.json").exists()
    assert tmp_path.joinpath("stdout.log").exists()
    metric = json.loads(tmp_path.joinpath("metrics.jsonl").read_text(encoding="utf-8"))
    summary = json.loads(tmp_path.joinpath("summary.json").read_text(encoding="utf-8"))
    assert metric["phase"] == "ppo"
    assert "final_policy" in summary


def test_capstone_dry_run_writes_stage_summary(tmp_path: Path) -> None:
    run_capstone_main(["--output-dir", str(tmp_path)])

    summary = json.loads(tmp_path.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "dry-run"
    assert len(summary["stages"]) == 7
    assert summary["stages"][0]["name"] == "pretrain"

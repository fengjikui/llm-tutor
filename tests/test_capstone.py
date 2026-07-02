from pathlib import Path

from llm_tutor.capstone.pipeline import build_capstone_stages, format_stage_command
from llm_tutor.experiments import run_capstone_pipeline


def test_capstone_pipeline_includes_all_post_training_stages() -> None:
    stages = build_capstone_stages(checkpoint_path=Path("/tmp/mini_gpt.pt"))
    names = [stage.name for stage in stages]

    assert names == ["pretrain", "generate", "inspect-loss", "sft", "ppo", "dpo", "grpo"]


def test_capstone_pipeline_threads_checkpoint_path() -> None:
    stages = build_capstone_stages(checkpoint_path=Path("/tmp/mini_gpt.pt"))
    commands = [format_stage_command(stage) for stage in stages]

    assert any("--checkpoint-path /tmp/mini_gpt.pt" in command for command in commands)
    assert "llm_tutor.experiments.train_mini_gpt" in commands[0]
    assert "llm_tutor.experiments.generate_with_mini_gpt" in commands[1]


def test_format_stage_command_is_readable() -> None:
    stage = build_capstone_stages(checkpoint_path=Path("/tmp/mini_gpt.pt"))[0]

    assert format_stage_command(stage).startswith("uv run python -m")


def test_capstone_dry_run_does_not_execute_subprocess(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(
        run_capstone_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(args),
    )

    run_capstone_pipeline.main([])

    output = capsys.readouterr().out
    assert "capstone_mode=dry-run" in output
    assert not calls


def test_capstone_execute_uses_same_temporary_checkpoint(monkeypatch) -> None:
    calls = []

    class FakeTemporaryDirectory:
        def __enter__(self):
            return "/tmp/fake-capstone"

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        run_capstone_pipeline.tempfile,
        "TemporaryDirectory",
        FakeTemporaryDirectory,
    )
    monkeypatch.setattr(
        run_capstone_pipeline.subprocess,
        "run",
        lambda command, check: calls.append(tuple(command)),
    )

    run_capstone_pipeline.main(["--execute"])

    assert len(calls) == 7
    joined = [" ".join(command) for command in calls]
    assert any("--checkpoint-path /tmp/fake-capstone/mini_gpt.pt" in command for command in joined)
    assert "/tmp/fake-capstone/mini_gpt.pt" in joined[0]
    assert "/tmp/fake-capstone/mini_gpt.pt" in joined[1]


def test_capstone_execute_with_custom_checkpoint_skips_temporary_directory(monkeypatch) -> None:
    calls = []

    def fail_if_called():
        raise AssertionError("TemporaryDirectory should not be used with --checkpoint-path")

    monkeypatch.setattr(run_capstone_pipeline.tempfile, "TemporaryDirectory", fail_if_called)
    monkeypatch.setattr(
        run_capstone_pipeline.subprocess,
        "run",
        lambda command, check: calls.append(tuple(command)),
    )

    run_capstone_pipeline.main(["--execute", "--checkpoint-path", "/tmp/custom.pt"])

    assert len(calls) == 7
    assert "/tmp/custom.pt" in " ".join(calls[0])
    assert "/tmp/custom.pt" in " ".join(calls[1])

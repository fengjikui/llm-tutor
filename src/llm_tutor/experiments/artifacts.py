from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class ExperimentArtifacts:
    """Small opt-in artifact writer for tutorial experiments."""

    def __init__(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir

    @classmethod
    def create(
        cls,
        output_dir: Path | None,
        *,
        experiment_name: str,
        config: Mapping[str, Any],
    ) -> ExperimentArtifacts:
        if output_dir is None:
            return cls(run_dir=None)

        run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = cls(run_dir=run_dir)
        writer.write_config(
            {
                "experiment": experiment_name,
                "created_at": datetime.now(UTC).isoformat(),
                "command": sys.argv,
                "config": dict(config),
            }
        )
        return writer

    @property
    def enabled(self) -> bool:
        return self.run_dir is not None

    def path(self, name: str) -> Path:
        if self.run_dir is None:
            raise RuntimeError("Artifact path requested, but artifact writing is disabled.")
        return self.run_dir / name

    @contextmanager
    def capture_stdout(self):
        if self.run_dir is None:
            with nullcontext():
                yield
            return

        stdout_path = self.path("stdout.log")
        with stdout_path.open("w", encoding="utf-8") as log_file:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, log_file)
            try:
                yield
            finally:
                sys.stdout = original_stdout

    def write_config(self, config: Mapping[str, Any]) -> None:
        self.write_json("config.json", config)

    def append_metric(self, row: Mapping[str, Any]) -> None:
        if self.run_dir is None:
            return
        with self.path("metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True))
            file.write("\n")

    def append_metrics(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.append_metric(row)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self.write_json("summary.json", summary)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        if self.run_dir is None:
            return
        self.path(name).write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_text(self, name: str, text: str) -> None:
        if self.run_dir is None:
            return
        self.path(name).write_text(text, encoding="utf-8")


def add_artifact_args(parser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for this run's artifacts. "
            "When set, writes config.json, metrics.jsonl, summary.json, and stdout.log."
        ),
    )


def args_to_config(args: Any) -> dict[str, Any]:
    return vars(args)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(item) for item in value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)

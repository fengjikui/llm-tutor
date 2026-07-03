from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_site_build_renders_lab_entry() -> None:
    subprocess.run([sys.executable, "scripts/build_site.py"], check=True)

    chapter = Path("site/chapters/06_cnn_image_classification.html").read_text(
        encoding="utf-8"
    )

    assert "本章实践入口" in chapter
    assert "src/llm_tutor/experiments/train_cnn.py" in chapter
    assert "uv run python -m llm_tutor.experiments.train_cnn" in chapter

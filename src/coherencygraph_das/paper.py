from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import resolve


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"paper build failed: {' '.join(command)}\n{completed.stdout}\n{completed.stderr}"
        )


def build_paper(cfg: dict) -> Path:
    root = Path(cfg["_root"])
    source = root / "manuscript" / "cageo_submission"
    build = source / "output"
    build.mkdir(parents=True, exist_ok=True)
    _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=output", "main.tex"], source)
    _run(["bibtex", "output/main"], source)
    for _ in range(2):
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=output", "main.tex"], source)
    for stem in ["supplement", "cover_letter"]:
        for _ in range(2):
            _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=output", f"{stem}.tex"], source)
    destination = resolve(cfg, cfg["paths"]["output_pdf"])
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "main.pdf": "CoherencyGraph_DAS_Manuscript.pdf",
        "supplement.pdf": "CoherencyGraph_DAS_Supplement.pdf",
        "cover_letter.pdf": "CoherencyGraph_DAS_Cover_Letter.pdf",
    }
    for source_name, destination_name in names.items():
        shutil.copy2(build / source_name, destination / destination_name)
    return destination

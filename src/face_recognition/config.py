# src/face_recognition/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Paths:
    """Project paths (code is in repo, data/models/outputs live in Drive workspace by default)."""
    project_dir: Path
    src_dir: Path
    workspace_dir: Path
    data_dir: Path
    models_dir: Path
    outputs_dir: Path


def _env_path(key: str, default: str) -> Path:
    """Read a path from env var if exists, otherwise use default."""
    val = os.environ.get(key, default)
    return Path(val).expanduser().resolve()


def get_paths() -> Paths:
    """
    Central place for all paths.
    - PROJECT_DIR: where the repo is cloned in Colab (/content/face-recognition-10p)
    - WORKSPACE_DIR: persistent Drive storage (or any local folder if you set env vars)
    """
    project_dir = _env_path("FR_PROJECT_DIR", "/content/face-recognition-10p")
    src_dir = project_dir / "src"

    workspace_dir = _env_path("FR_WORKSPACE_DIR", "/content/drive/MyDrive/face-recognition-10p-workspace")
    data_dir = workspace_dir / "data"
    models_dir = workspace_dir / "models"
    outputs_dir = workspace_dir / "outputs"

    return Paths(
        project_dir=project_dir,
        src_dir=src_dir,
        workspace_dir=workspace_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
    )


def get_device() -> str:
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

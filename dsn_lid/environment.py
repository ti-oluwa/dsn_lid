"""Runtime environment: where files live, Hugging Face login, device, seeding."""

from __future__ import annotations

import dataclasses
import os
import random
from pathlib import Path

import numpy
import torch
from huggingface_hub import get_token, login, snapshot_download

from .config import DATASET_REPOSITORY

KAGGLE_WORKING_DIRECTORY = Path("/kaggle/working")
KAGGLE_TEMP_DIRECTORY = Path("/kaggle/temp")


def is_running_on_kaggle() -> bool:
    """Whether the code is running inside a Kaggle notebook."""
    return KAGGLE_WORKING_DIRECTORY.exists()


@dataclasses.dataclass(frozen=True)
class Workspace:
    """Folders used by a run.

    Large, disposable files (dataset, checkpoints) go to scratch space so they are
    not saved as Kaggle notebook output; deliverables go to the output directory.
    """

    data_directory: Path
    """Downloaded training and validation data."""

    checkpoint_directory: Path
    """Training checkpoints and best-weights files."""

    export_directory: Path
    """The Hugging Face model folder that gets uploaded."""

    output_directory: Path
    """Experiment logs and the submission zip."""

    @classmethod
    def detect(cls) -> Workspace:
        """Build a workspace for Kaggle (when detected) or for the current folder."""
        if is_running_on_kaggle():
            output_directory = KAGGLE_WORKING_DIRECTORY
            scratch_directory = (
                KAGGLE_TEMP_DIRECTORY
                if KAGGLE_TEMP_DIRECTORY.is_dir()
                else output_directory
            )
        else:
            output_directory = scratch_directory = Path.cwd() / "workspace"

        workspace = cls(
            data_directory=scratch_directory / "data",
            checkpoint_directory=output_directory / "checkpoints",
            export_directory=output_directory / "export",
            output_directory=output_directory,
        )
        for directory in (
            workspace.data_directory,
            workspace.checkpoint_directory,
            workspace.output_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return workspace


def resolve_hub_token() -> str | None:
    """Find a Hugging Face token in the environment, Kaggle secrets or Colab secrets."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa  # not on Kaggle, or the secret is not attached to this notebook
        pass
    try:
        from google.colab import userdata

        return userdata.get("HF_TOKEN")
    except Exception:  # not on Colab, or the secret is missing
        return None


def login_to_hub() -> bool:
    """Log in to Hugging Face with whatever credentials are available.

    :returns: True if a token was found or one is already cached on this machine.
    """
    token = resolve_hub_token()
    if token:
        login(token=token)
        return True
    return get_token() is not None


def select_device() -> torch.device:
    """Return the CUDA device when available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def download_dataset(data_directory: Path) -> Path:
    """Download the competition dataset (resumes if already present).

    :param data_directory: Folder to download into.
    :returns: The folder containing train.csv and validation.csv.
    """
    snapshot_download(
        repo_id=DATASET_REPOSITORY, repo_type="dataset", local_dir=str(data_directory)
    )
    return data_directory

"""Publishing the model to Hugging Face and building the Codabench submission file."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

COMMIT_HASH_LENGTH = 40
UPLOAD_PATTERNS = [
    "*.safetensors",
    "config.json",
    "preprocessor_config.json",
    "README.md",
    "LICENSE*",
]


def push_to_hub(
    export_directory: Path, repository_id: str, allow_existing: bool = False
) -> str:
    """Upload the export folder to a public repository and return the commit hash.

    :param export_directory: Folder produced by `export_model` and `write_model_card`.
    :param repository_id: `username/model-name`.
    :param allow_existing: Permit pushing to a repository that already exists. Leave False so a
        repository that was already submitted is never modified.
    :raises FileExistsError: If the repository exists and `allow_existing` is False.
    """
    api = HfApi()
    if not allow_existing and api.repo_exists(repo_id=repository_id):
        raise FileExistsError(
            f"{repository_id} already exists; use a new name for each submission"
        )

    api.create_repo(repo_id=repository_id, private=False, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(export_directory),
        repo_id=repository_id,
        allow_patterns=UPLOAD_PATTERNS,
    )
    commit_hash = commit.oid
    if len(commit_hash) != COMMIT_HASH_LENGTH:
        raise ValueError(
            f"expected a {COMMIT_HASH_LENGTH}-character commit hash, got {commit_hash!r}"
        )
    return commit_hash


def write_submission_zip(
    repository_id: str, commit_hash: str, output_directory: Path
) -> Path:
    """Create submission.zip containing submission.json for the Codabench Participate page.

    :returns: Path of the zip file.
    """
    submission = {"repo": repository_id, "revision": commit_hash}
    zip_path = output_directory / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("submission.json", json.dumps(submission))
    return zip_path

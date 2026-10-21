"""Exporting the model as a Hugging Face folder, its model card, and pre-submission checks."""

from __future__ import annotations

import copy
import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    PreTrainedModel,
    SequenceFeatureExtractor,
)

from .config import (
    ID_TO_LABEL,
    LABELS,
    MAX_PARAMETERS,
    MAX_REPOSITORY_BYTES,
    SAMPLE_RATE,
)
from .modeling import count_parameters

ALLOWED_FILE_NAMES = {
    "config.json",
    "preprocessor_config.json",
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
}
FORBIDDEN_SUFFIXES = (".py", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
ALLOWED_DTYPES = {"F16", "F32"}


def export_model(
    model: PreTrainedModel,
    feature_extractor: SequenceFeatureExtractor,
    export_directory: Path,
) -> None:
    """Write an fp16 safetensors copy of the model and its preprocessor to an empty folder.

    The model passed in is left untouched.
    """
    shutil.rmtree(export_directory, ignore_errors=True)
    export_directory.mkdir(parents=True)
    half_precision_model = copy.deepcopy(model).cpu().half().eval()
    half_precision_model.save_pretrained(export_directory, safe_serialization=True)
    feature_extractor.save_pretrained(export_directory)


@dataclasses.dataclass(frozen=True)
class ModelCardDetails:
    """Facts that go into the README of the submitted model."""

    base_model: str
    """Pretrained encoder the model starts from."""

    num_layers: int
    """Encoder layers kept."""

    total_steps: int
    """Optimizer steps of the training run."""

    weighted_f1: float
    """Validation language-weighted F1 of the exported model."""

    per_class_f1: dict[str, float]
    """Validation F1 of each label."""

    bias_calibrated: bool
    """Whether per-class offsets were added to the classifier bias."""

    parameters_millions: float
    """Total parameters, in millions."""

    repository_megabytes: float
    """Size of the exported folder."""

    seconds_per_clip: float
    """Measured average inference time per clip."""

    peak_memory_gb: float
    """Measured peak memory at inference."""

    benchmark_device: str
    """Device the timing and memory were measured on."""

    external_datasets: tuple[str, ...] = ()
    """Every dataset used besides the competition data; must be listed in the card."""


def build_model_card(details: ModelCardDetails) -> str:
    """Render the README.md for the submitted model."""
    class_rows = "\n".join(
        f"| {label} | {score:.3f} |" for label, score in details.per_class_f1.items()
    )
    external = "\n".join(f"- {name}" for name in details.external_datasets) or "None."
    calibration = (
        " Per-class offsets tuned on the validation set were added to the classifier bias."
        if details.bias_calibrated
        else ""
    )
    return f"""---
license: cc-by-nc-4.0
language: [{", ".join(LABELS[:-1])}]
pipeline_tag: audio-classification
base_model: {details.base_model}
---
# Nigerian language identification

Identifies which of 12 Nigerian languages is spoken in a short clip, or `other` for anything else
(English, Nigerian Pidgin, other languages). Labels: {" ".join(LABELS)}.

## Model

`{details.base_model}` cut down to its first {details.num_layers} transformer layers, followed by a
learned weighted sum over the layer outputs, mean pooling and a linear classifier. Weights are fp16.

Training used {details.total_steps} optimizer steps on random 4-8 second crops, class-balanced sampling,
label smoothing, and augmentation with telephone band-limiting, mu-law coding, noise, reverb and speed
changes.{calibration}

## Data

- Training data: Data-Science-Nigeria/dsn-2026-lid-train, used under the licences and attribution terms
  in its ATTRIBUTION.md.
- External datasets:
{external}
- None of the prohibited corpora (Hypa-Voices, omnilingual-asr-corpus, african-speech-ipa) were used.

## Results

Speaker-disjoint validation, language-weighted F1: **{details.weighted_f1:.4f}**

| Label | F1 |
|---|---|
{class_rows}

Local measurement on {details.benchmark_device}: {details.parameters_millions:.1f}M parameters,
{details.repository_megabytes:.0f} MB repository, {details.seconds_per_clip:.3f} s per clip,
{details.peak_memory_gb:.2f} GB peak memory.

## Limitations

Annang, Ibibio and Efik are closely related and are the most likely to be confused with each other.
Nigerian Pidgin is close to English, which makes `other` harder. The base model is released for
non-commercial use (CC BY-NC 4.0), so this model is too.
"""


def write_model_card(export_directory: Path, details: ModelCardDetails) -> None:
    """Write README.md and a LICENSE note into the export folder."""
    (export_directory / "README.md").write_text(
        build_model_card(details), encoding="utf-8"
    )
    (export_directory / "LICENSE").write_text(
        f"Derived from {details.base_model} (CC BY-NC 4.0) and fine-tuned on "
        "Data-Science-Nigeria/dsn-2026-lid-train. See README.md and the dataset's ATTRIBUTION.md.\n",
        encoding="utf-8",
    )


class PreflightReport(NamedTuple):
    """Result of checking an export folder against the submission rules."""

    problems: list[str]
    """Rule violations; empty when the folder is fine to submit."""

    num_parameters: int
    """Parameter count of the loaded model."""

    repository_bytes: int
    """Total size of the files in the folder."""

    @property
    def passed(self) -> bool:
        """True when no problems were found."""
        return not self.problems


def check_export_folder(export_directory: Path) -> PreflightReport:
    """Verify an export folder against the submission rules.

    Checks file types, label order, weight dtypes, size and parameter limits, that the
    folder loads with the standard Auto classes, and that a forward pass returns 13 logits.
    """
    problems: list[str] = []
    files = [path for path in export_directory.iterdir() if path.is_file()]
    repository_bytes = sum(path.stat().st_size for path in files)

    for path in files:
        is_weights = path.name == "model.safetensors" or (
            path.name.startswith("model-") and path.suffix == ".safetensors"
        )
        if path.suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file: {path.name}")
        elif not is_weights and path.name not in ALLOWED_FILE_NAMES:
            problems.append(f"unexpected file: {path.name}")
    if any(path.is_dir() for path in export_directory.iterdir()):
        problems.append("the folder contains sub-directories")

    config = AutoModelForAudioClassification.from_pretrained(export_directory).config
    if [config.id2label[index] for index in range(len(ID_TO_LABEL))] != list(
        LABELS
    ) or len(config.id2label) != len(LABELS):
        problems.append("id2label does not match the required 13 labels in order")
    if getattr(config, "auto_map", None):
        problems.append("config.json contains auto_map")

    for weights_path in export_directory.glob("*.safetensors"):
        with safe_open(str(weights_path), framework="pt") as weights_file:
            dtypes = {
                weights_file.get_slice(name).get_dtype() for name in weights_file.keys()
            }
        if not dtypes <= ALLOWED_DTYPES:
            problems.append(
                f"{weights_path.name} has unsupported dtypes: {sorted(dtypes - ALLOWED_DTYPES)}"
            )

    model = (
        AutoModelForAudioClassification.from_pretrained(export_directory).float().eval()
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(export_directory)
    num_parameters = count_parameters(model)
    if num_parameters > MAX_PARAMETERS:
        problems.append(
            f"{num_parameters / 1e6:.1f}M parameters exceeds the 350M limit"
        )
    if repository_bytes > MAX_REPOSITORY_BYTES:
        problems.append(f"{repository_bytes / 1e9:.2f} GB exceeds the 1.5 GB limit")
    if feature_extractor.sampling_rate != SAMPLE_RATE:
        problems.append(
            f"preprocessor sampling rate is {feature_extractor.sampling_rate}, expected {SAMPLE_RATE}"
        )

    probe = feature_extractor(
        [torch.randn(SAMPLE_RATE * 2).numpy()],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    with torch.no_grad():
        logits = model(
            input_values=probe["input_values"], attention_mask=probe["attention_mask"]
        ).logits
    if logits.shape != (1, len(LABELS)):
        problems.append(f"forward pass returned logits of shape {tuple(logits.shape)}")

    return PreflightReport(
        problems=problems,
        num_parameters=num_parameters,
        repository_bytes=repository_bytes,
    )


def run_official_checker(checker_script: Path, export_directory: Path) -> int:
    """Run the starting kit's check_model_folder.py on the export folder.

    Assumes the script takes the model folder as its only argument; adjust if the kit differs.

    :returns: The checker's exit code.
    """
    completed = subprocess.run(
        [sys.executable, str(checker_script), str(export_directory)], check=False
    )
    return completed.returncode

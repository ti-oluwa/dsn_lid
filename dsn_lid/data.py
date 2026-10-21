"""Manifests, audio loading, the training dataset and batching."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy
import pandas
import soundfile
import torch
import torchaudio.functional as audio_functional
from transformers import SequenceFeatureExtractor

from .augment import Augmenter
from .config import (
    HIGH_RESOURCE_LABELS,
    LABEL_TO_ID,
    NUM_LABELS,
    SAMPLE_RATE,
    TrainingConfig,
)

REQUIRED_COLUMNS = ("id", "path", "label", "duration")


def load_manifest(data_directory: Path, split: str) -> pandas.DataFrame:
    """Read CSV manifests or extract embedded audio from Parquet shards.

    :param data_directory: Folder containing the downloaded dataset.
    :param split: Either "train" or "validation".
    :returns: The manifest with an integer `label_index` column and an absolute `audio_path` column.
    :raises ValueError: If a required column is missing or a label is not one of the 13 classes.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    csv_path = data_directory / f"{split}.csv"
    parquet_paths = sorted(
        set(data_directory.glob(f"{split}-*.parquet"))
        | set(data_directory.glob(f"data/{split}-*.parquet"))
    )
    if csv_path.exists():
        manifest = pandas.read_csv(csv_path)
    elif parquet_paths:
        manifest_parts = []
        audio_directory = data_directory / "audio" / split
        audio_directory.mkdir(parents=True, exist_ok=True)
        for shard_index, parquet_path in enumerate(parquet_paths):
            shard = pandas.read_parquet(parquet_path)
            if "audio" not in shard.columns:
                raise ValueError(f"{parquet_path.name} is missing the audio column")

            audio_paths = []
            for row_index, audio in enumerate(shard.pop("audio")):
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not isinstance(audio_bytes, (bytes, bytearray, memoryview)):
                    raise TypeError(
                        f"{parquet_path.name} row {row_index} has no embedded audio bytes"
                    )
                audio_path = audio_directory / f"{shard_index:05d}-{row_index:06d}.flac"
                if not audio_path.exists():
                    audio_path.write_bytes(audio_bytes)
                audio_paths.append(str(audio_path.relative_to(data_directory)))

            shard["path"] = audio_paths
            manifest_parts.append(shard)
        manifest = pandas.concat(manifest_parts, ignore_index=True)
    else:
        raise FileNotFoundError(
            f"No {split}.csv or {split}-*.parquet files found in {data_directory}"
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in manifest.columns
    ]
    if missing_columns:
        raise ValueError(f"{split} manifest is missing columns: {missing_columns}")

    manifest["label_index"] = manifest["label"].map(LABEL_TO_ID)
    if manifest["label_index"].isna().any():
        unknown_labels = sorted(
            manifest.loc[manifest["label_index"].isna(), "label"].unique()
        )
        raise ValueError(f"{split} manifest has unknown labels: {unknown_labels}")

    manifest["label_index"] = manifest["label_index"].astype(int)
    manifest["audio_path"] = manifest["path"].map(
        lambda path: str(path if Path(path).is_absolute() else data_directory / path)
    )
    return manifest


def audit_manifests(
    train_manifest: pandas.DataFrame, validation_manifest: pandas.DataFrame
) -> None:
    """Print the dataset statistics that matter before training.

    Covers class balance, source mix, what `other` contains, speaker leakage between
    the splits, label_id consistency and missing audio files.
    """
    print(train_manifest.head(3).to_string(), "\n")
    summary = train_manifest.groupby("label").agg(
        clips=("id", "count"),
        hours=("duration", lambda seconds: seconds.sum() / 3600),
        speakers=("speaker_id", "nunique"),
    )
    print(
        summary.reindex(
            list(HIGH_RESOURCE_LABELS)
            + [l for l in summary.index if l not in HIGH_RESOURCE_LABELS]
        ),
        "\n",
    )
    print(
        train_manifest.groupby(["label", "source"]).size().unstack(fill_value=0), "\n"
    )
    print("`other` by source:")
    print(
        train_manifest.loc[train_manifest["label"] == "other", "source"].value_counts(),
        "\n",
    )
    print("validation clips per class:")
    print(validation_manifest["label"].value_counts(), "\n")
    print("duration (s):")
    print(train_manifest["duration"].describe(), "\n")
    shared_speakers = set(train_manifest["speaker_id"].dropna()) & set(
        validation_manifest["speaker_id"].dropna()
    )
    print(f"speakers present in both splits: {len(shared_speakers)}")
    if "label_id" in train_manifest.columns:
        consistent = (train_manifest["label_id"] == train_manifest["label_index"]).all()
        print(f"label_id column matches the required label order: {consistent}")

    for name, manifest in (
        ("train", train_manifest),
        ("validation", validation_manifest),
    ):
        missing_files = sum(not Path(path).exists() for path in manifest["audio_path"])
        print(
            f"{name}: {len(manifest)} clips, {manifest['duration'].sum() / 3600:.1f} h, {missing_files} missing files"
        )


def load_waveform(
    path: str | Path,
    max_seconds: float | None = None,
    crop_seconds: float | None = None,
) -> torch.Tensor:
    """Read a clip as a mono 16 kHz float waveform.

    :param path: Audio file to read.
    :param max_seconds: Keep only the first this-many seconds (ignored when `crop_seconds` is set).
    :param crop_seconds: Read a random window of this length; shorter clips are returned whole.
    """
    info = soundfile.info(str(path))
    start_frame, stop_frame = 0, info.frames
    if crop_seconds is not None:
        crop_frames = int(crop_seconds * info.samplerate)
        if info.frames > crop_frames:
            start_frame = random.randint(0, info.frames - crop_frames)
            stop_frame = start_frame + crop_frames
    elif max_seconds is not None:
        stop_frame = min(info.frames, int(max_seconds * info.samplerate))

    samples, _ = soundfile.read(
        str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(samples.mean(axis=1))
    if info.samplerate != SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform, orig_freq=info.samplerate, new_freq=SAMPLE_RATE
        )
    return waveform


class Batch(NamedTuple):
    """One padded batch ready for the model."""

    input_values: torch.Tensor
    """Normalised waveforms, shape (batch, samples)."""

    attention_mask: torch.Tensor
    """1 for real samples and 0 for padding, same shape as `input_values`."""

    labels: torch.Tensor
    """Class indices, shape (batch,)."""


def extract_features(
    feature_extractor: SequenceFeatureExtractor,
    waveforms: Sequence[numpy.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise and pad waveforms into model inputs.

    :returns: `(input_values, attention_mask)`.
    """
    features = feature_extractor(
        list(waveforms),
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return features["input_values"], features["attention_mask"]


class TrainingDataset(torch.utils.data.Dataset):
    """Random, augmented crops of the training clips."""

    def __init__(
        self,
        manifest: pandas.DataFrame,
        crop_seconds_range: tuple[float, float],
        augmenter: Augmenter,
    ) -> None:
        """
        :param manifest: Training manifest with `audio_path` and `label_index` columns.
        :param crop_seconds_range: Range of the random crop length in seconds.
        :param augmenter: Applied to every crop.
        """
        self.audio_paths = manifest["audio_path"].tolist()
        self.label_indices = manifest["label_index"].to_numpy()
        self.crop_seconds_range = crop_seconds_range
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> tuple[numpy.ndarray, int]:
        crop_seconds = random.uniform(*self.crop_seconds_range)
        waveform = load_waveform(self.audio_paths[index], crop_seconds=crop_seconds)
        return self.augmenter(waveform).numpy(), int(self.label_indices[index])


class WaveformCollator:
    """Turns a list of (waveform, label) pairs into a padded `Batch`."""

    def __init__(self, feature_extractor: SequenceFeatureExtractor) -> None:
        """
        :param feature_extractor: Normalises and pads the waveforms.
        """
        self.feature_extractor = feature_extractor

    def __call__(self, examples: Sequence[tuple[numpy.ndarray, int]]) -> Batch:
        waveforms, labels = zip(*examples)
        input_values, attention_mask = extract_features(
            self.feature_extractor, waveforms
        )
        return Batch(input_values, attention_mask, torch.tensor(labels))


def class_balanced_weights(label_indices: numpy.ndarray, power: float) -> torch.Tensor:
    """Per-clip sampling weights proportional to 1 / class_count ** power."""
    class_counts = numpy.bincount(label_indices, minlength=NUM_LABELS).astype(float)
    class_weights = 1.0 / numpy.maximum(class_counts, 1.0) ** power
    return torch.as_tensor(class_weights[label_indices], dtype=torch.double)


def build_training_loader(
    manifest: pandas.DataFrame,
    feature_extractor: SequenceFeatureExtractor,
    config: TrainingConfig,
    augmenter: Augmenter,
    num_batches: int,
) -> torch.utils.data.DataLoader:
    """Create a loader that yields exactly `num_batches` class-balanced batches.

    :param manifest: Training manifest.
    :param feature_extractor: Used to pad and normalise each batch.
    :param config: Provides batch size, crop range, sampling power and worker count.
    :param augmenter: Applied to every crop.
    :param num_batches: Number of batches the loader should produce before stopping.
    """
    dataset = TrainingDataset(
        manifest=manifest,
        crop_seconds_range=config.crop_seconds_range,
        augmenter=augmenter,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_balanced_weights(
            manifest["label_index"].to_numpy(), power=config.sampling_power
        ),
        num_samples=num_batches * config.batch_size,
        replacement=True,
    )
    use_workers = config.num_workers > 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=WaveformCollator(feature_extractor),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
    )

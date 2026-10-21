"""Inference over a manifest and validation scoring."""

from __future__ import annotations

from typing import NamedTuple

import numpy
import pandas
import torch
from transformers import PreTrainedModel, SequenceFeatureExtractor

from .config import NUM_LABELS
from .data import extract_features, load_waveform
from .metrics import language_weighted_f1


@torch.no_grad()
def predict_logits(
    model: PreTrainedModel,
    feature_extractor: SequenceFeatureExtractor,
    manifest: pandas.DataFrame,
    device: torch.device,
    batch_size: int = 8,
    max_seconds: float = 30.0,
) -> numpy.ndarray:
    """Run the model over every clip in a manifest.

    Clips are batched by length to minimise padding. Uses fp16 autocast on CUDA.

    :param model: Classifier, already on `device`.
    :param feature_extractor: Normalises and pads the waveforms.
    :param manifest: Needs `audio_path` and `duration` columns.
    :param device: Device the model is on.
    :param batch_size: Clips per forward pass.
    :param max_seconds: Clips longer than this are truncated.
    :returns: Logits in manifest order, shape (clips, 13).
    """
    model.eval()
    audio_paths = manifest["audio_path"].tolist()
    shortest_first = numpy.argsort(manifest["duration"].to_numpy())
    logits = numpy.zeros((len(manifest), NUM_LABELS), dtype=numpy.float32)
    for start in range(0, len(shortest_first), batch_size):
        positions = shortest_first[start : start + batch_size]
        waveforms = [
            load_waveform(audio_paths[position], max_seconds=max_seconds).numpy()
            for position in positions
        ]
        input_values, attention_mask = extract_features(feature_extractor, waveforms)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            batch_logits = model(
                input_values=input_values.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
        logits[positions] = batch_logits.float().cpu().numpy()
    return logits


class EvaluationResult(NamedTuple):
    """Scores and raw outputs from one validation pass."""

    weighted_f1: float
    """Language-weighted F1 over the manifest."""

    logits: numpy.ndarray
    """Model outputs in manifest order, shape (clips, 13)."""


def evaluate(
    model: PreTrainedModel,
    feature_extractor: SequenceFeatureExtractor,
    manifest: pandas.DataFrame,
    device: torch.device,
    batch_size: int = 8,
    max_seconds: float = 30.0,
) -> EvaluationResult:
    """Predict a labelled manifest and score it. Arguments are those of `predict_logits`."""
    logits = predict_logits(
        model=model,
        feature_extractor=feature_extractor,
        manifest=manifest,
        device=device,
        batch_size=batch_size,
        max_seconds=max_seconds,
    )
    weighted_f1 = language_weighted_f1(
        manifest["label_index"].to_numpy(), logits.argmax(axis=1)
    )
    return EvaluationResult(weighted_f1=weighted_f1, logits=logits)

"""Competition metrics: language-weighted F1, efficiency, final score, and bias calibration."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy

from .config import (
    HIGH_RESOURCE_LABELS,
    LABELS,
    MAX_PARAMETERS,
    MAX_PEAK_MEMORY_GB,
    MAX_REPOSITORY_BYTES,
    MAX_SECONDS_PER_CLIP,
    MIN_RANKED_F1,
    MIN_SECONDS_PER_CLIP,
    NUM_LABELS,
)

RAW_CLASS_SHARES = numpy.array([
    4.3 if label in HIGH_RESOURCE_LABELS else 8.7 for label in LABELS
])
CLASS_WEIGHTS: numpy.ndarray = RAW_CLASS_SHARES / RAW_CLASS_SHARES.sum()
"""Fixed share of each class in the language-weighted F1, ordered like `LABELS`."""


def confusion_counts(
    true_indices: numpy.ndarray, predicted_indices: numpy.ndarray
) -> numpy.ndarray:
    """Confusion matrix of counts: rows are true classes, columns are predicted classes."""
    flat_pairs = true_indices * NUM_LABELS + predicted_indices
    return numpy.bincount(flat_pairs, minlength=NUM_LABELS**2).reshape(
        NUM_LABELS, NUM_LABELS
    )


def per_class_f1(
    true_indices: numpy.ndarray, predicted_indices: numpy.ndarray
) -> numpy.ndarray:
    """F1 score of each class, 0 for a class with no true and no predicted clips."""
    counts = confusion_counts(true_indices, predicted_indices)
    true_positives = numpy.diag(counts).astype(float)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    return numpy.divide(
        2 * true_positives,
        denominator,
        out=numpy.zeros(NUM_LABELS),
        where=denominator > 0,
    )


def language_weighted_f1(
    true_indices: numpy.ndarray, predicted_indices: numpy.ndarray
) -> float:
    """The competition's accuracy metric: per-class F1 combined with the fixed class shares."""
    return float((per_class_f1(true_indices, predicted_indices) * CLASS_WEIGHTS).sum())


class Efficiency(NamedTuple):
    """The three efficiency components, each between 0 and 1."""

    size: float
    """Mean of the parameter-count score and the repository-size score."""

    memory: float
    """Score for peak inference memory."""

    speed: float
    """Score for average seconds per clip on a log scale."""

    @property
    def overall(self) -> float:
        """Mean of the three components."""
        return (self.size + self.memory + self.speed) / 3


def clip_to_unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))


def compute_efficiency(
    num_parameters: int,
    repository_bytes: float,
    peak_memory_gb: float,
    seconds_per_clip: float,
) -> Efficiency:
    """Score a model against the competition limits.

    :param num_parameters: Parameter count of the whole model.
    :param repository_bytes: Total size of the files that would be uploaded.
    :param peak_memory_gb: Peak of GPU and RAM usage at inference.
    :param seconds_per_clip: Average time for the full pipeline on one clip.
    """
    parameter_score = 1 - num_parameters / MAX_PARAMETERS
    repository_score = 1 - repository_bytes / MAX_REPOSITORY_BYTES
    speed_score = 1 - math.log(seconds_per_clip / MIN_SECONDS_PER_CLIP) / math.log(
        MAX_SECONDS_PER_CLIP / MIN_SECONDS_PER_CLIP
    )
    return Efficiency(
        size=clip_to_unit_interval((parameter_score + repository_score) / 2),
        memory=clip_to_unit_interval(1 - peak_memory_gb / MAX_PEAK_MEMORY_GB),
        speed=clip_to_unit_interval(speed_score),
    )


def estimate_final_score(weighted_f1: float, efficiency: Efficiency) -> float:
    """Final score: language-weighted F1 scaled by 0.5 to 1.0 according to efficiency."""
    return weighted_f1 * (0.5 + 0.5 * efficiency.overall)


class BiasTuningResult(NamedTuple):
    """Outcome of tuning per-class logit offsets."""

    bias: numpy.ndarray
    """Offset to add to each class logit, ordered like `LABELS`."""

    f1_before: float
    """Language-weighted F1 without offsets, on the data used for tuning."""

    f1_after: float
    """Language-weighted F1 with offsets, on the data used for tuning."""


def tune_class_bias(
    logits: numpy.ndarray,
    true_indices: numpy.ndarray,
    grid: numpy.ndarray | None = None,
    rounds: int = 3,
) -> BiasTuningResult:
    """Coordinate ascent over per-class logit offsets to maximise the language-weighted F1.

    :param logits: Model outputs, shape (clips, 13).
    :param true_indices: Class index of each clip.
    :param grid: Candidate offset values tried for each class.
    :param rounds: Passes over all classes.
    """
    grid = numpy.linspace(-1.0, 1.0, 21) if grid is None else grid
    bias = numpy.zeros(NUM_LABELS)
    f1_before = best_f1 = language_weighted_f1(true_indices, logits.argmax(axis=1))
    for _ in range(rounds):
        for class_index in range(NUM_LABELS):
            for candidate in grid:
                trial = bias.copy()
                trial[class_index] = candidate
                trial_f1 = language_weighted_f1(
                    true_indices, (logits + trial).argmax(axis=1)
                )
                if trial_f1 > best_f1 + 1e-4:
                    best_f1, bias = trial_f1, trial
    return BiasTuningResult(bias=bias, f1_before=f1_before, f1_after=best_f1)


def held_out_bias_gain(
    logits: numpy.ndarray,
    true_indices: numpy.ndarray,
    grid: numpy.ndarray | None = None,
    repeats: int = 3,
    seed: int = 0,
) -> float:
    """Average F1 gain from offsets tuned on one random half of the data and scored on the other.

    A gain near zero or negative means the offsets only fit the tuning data.
    """
    generator = numpy.random.default_rng(seed)
    gains = []
    for _ in range(repeats):
        first_half, second_half = numpy.array_split(
            generator.permutation(len(true_indices)), 2
        )
        for tuning_half, scoring_half in (
            (first_half, second_half),
            (second_half, first_half),
        ):
            bias = tune_class_bias(
                logits[tuning_half], true_indices[tuning_half], grid=grid
            ).bias
            scoring_logits, scoring_truth = (
                logits[scoring_half],
                true_indices[scoring_half],
            )
            gains.append(
                language_weighted_f1(
                    scoring_truth, (scoring_logits + bias).argmax(axis=1)
                )
                - language_weighted_f1(scoring_truth, scoring_logits.argmax(axis=1))
            )
    return float(numpy.mean(gains))


def suggest_next_steps(
    weighted_f1: float, per_class_scores: dict[str, float]
) -> list[str]:
    """Plain-language hints based on the validation results.

    :param weighted_f1: Language-weighted F1 on validation.
    :param per_class_scores: F1 of each label.
    """
    hints = []
    if weighted_f1 < MIN_RANKED_F1:
        hints.append(
            f"F1 is below {MIN_RANKED_F1}, so a submission would not be ranked. "
            "Check labels, normalisation, attention mask, learning rate and data loading before tuning."
        )

    weakest = sorted(per_class_scores, key=per_class_scores.get)[:3]
    if set(weakest) & {"anw", "efi", "ibb"}:
        hints.append(
            "Annang, Ibibio or Efik are among the weakest classes: train longer or check for "
            "speaker and source leakage."
        )
    if per_class_scores.get("other", 1.0) < 0.7:
        hints.append(
            "`other` scores low: inspect what it contains and rebalance before adding data."
        )
    if weighted_f1 > 0.85:
        hints.append("Accuracy is strong; if inference is slow, try fewer layers.")
    return hints

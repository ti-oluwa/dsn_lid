"""Constants and configuration objects shared by every stage of the pipeline."""

from __future__ import annotations

import dataclasses

LABELS: tuple[str, ...] = (
    "yor",
    "ibo",
    "hau",
    "ego",
    "anw",
    "efi",
    "ibb",
    "idu",
    "igl",
    "nup",
    "tiv",
    "urh",
    "other",
)
"""Class labels in the exact order required by `id2label` in the submitted config."""

NUM_LABELS: int = len(LABELS)
LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = dict(enumerate(LABELS))

HIGH_RESOURCE_LABELS: tuple[str, ...] = ("yor", "ibo", "hau")
"""Languages that carry half the weight of the others in the language-weighted F1."""

SAMPLE_RATE: int = 16_000
DATASET_REPOSITORY: str = "Data-Science-Nigeria/dsn-2026-lid-train"

MAX_PARAMETERS: int = 350_000_000
MAX_REPOSITORY_BYTES: float = 1.5e9
MAX_PEAK_MEMORY_GB: float = 4.0
MIN_SECONDS_PER_CLIP: float = 0.01
MAX_SECONDS_PER_CLIP: float = 0.5
MIN_RANKED_F1: float = 0.60


@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    """Probabilities and ranges for the training-time waveform augmentations."""

    speed_probability: float = 0.30
    """Chance of resampling the clip to change speed and pitch."""

    speed_factors: tuple[float, ...] = (0.9, 0.95, 1.05, 1.1)
    """Speed multipliers to pick from when speed perturbation is applied."""

    reverb_probability: float = 0.15
    """Chance of convolving the clip with a synthetic room impulse response."""

    reverb_decay_seconds: tuple[float, float] = (0.1, 0.5)
    """Range of the reverberation decay time in seconds."""

    telephone_probability: float = 0.30
    """Chance of band-limiting the clip to 300-3400 Hz at an 8 kHz sample rate."""

    mu_law_probability: float = 0.30
    """Chance of mu-law companding on a clip that was already band-limited."""

    noise_probability: float = 0.40
    """Chance of mixing in white or pink noise."""

    noise_snr_db: tuple[float, float] = (5.0, 20.0)
    """Range of the signal-to-noise ratio, in dB, for added noise."""


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Backbone and architecture settings."""

    base_model: str = "facebook/mms-300m"
    """Hugging Face id or local folder of the pretrained encoder."""

    num_layers: int = 12
    """Transformer layers kept from the encoder; the rest are discarded."""

    mask_time_probability: float = 0.05
    """Fraction of time steps masked by the built-in SpecAugment."""

    gradient_checkpointing: bool = False
    """Trade compute for memory; enable when training runs out of GPU memory."""


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Optimisation, sampling and evaluation settings."""

    total_steps: int = 2500
    """Optimizer steps in the run."""

    evaluation_interval_steps: int = 500
    """Steps between validation passes and checkpoints."""

    log_interval_steps: int = 50
    """Steps between progress lines."""

    batch_size: int = 16
    """Crops per forward pass."""

    gradient_accumulation_steps: int = 2
    """Forward passes accumulated into one optimizer step."""

    crop_seconds_range: tuple[float, float] = (4.0, 8.0)
    """Range of the random crop length, in seconds, taken from each training clip."""

    encoder_learning_rate: float = 4e-5
    """Learning rate for the pretrained encoder layers."""

    head_learning_rate: float = 1e-3
    """Learning rate for the layer weights, projector and classifier."""

    weight_decay: float = 0.01
    """AdamW weight decay."""

    warmup_fraction: float = 0.10
    """Fraction of steps spent warming the learning rate up linearly."""

    label_smoothing: float = 0.1
    """Label smoothing for the cross-entropy loss."""

    sampling_power: float = 1.0
    """Class sampling weight is 1 / count ** power: 1.0 balances classes fully, 0.5 is inverse-sqrt."""

    max_gradient_norm: float = 1.0
    """Gradient clipping threshold."""

    num_workers: int = 4
    """Data loading worker processes."""

    seed: int = 42
    """Seed for Python, NumPy and PyTorch."""

    max_evaluation_seconds: float = 30.0
    """Validation clips longer than this are truncated."""

    evaluation_batch_size: int = 8
    """Clips per forward pass during validation."""

    @property
    def clips_per_step(self) -> int:
        """Number of crops that contribute to one optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps

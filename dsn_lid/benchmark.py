"""Inference speed and memory measurement, run in a fresh process for realistic numbers.

Command line: `python -m dsn_lid.benchmark MODEL_DIRECTORY --device cuda`
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from .config import SAMPLE_RATE

CLIP_SECONDS = (3, 8, 15, 30)
CLIPS_PER_LENGTH = 10
WARMUP_CLIPS = 5


class BenchmarkResult(NamedTuple):
    """Inference cost of an exported model on one device."""

    device: str
    """Either "cuda" or "cpu"."""

    seconds_per_clip: float
    """Median over passes of the mean time to process one clip, feature extraction included."""

    peak_gpu_gb: float
    """Peak CUDA memory allocated during inference; 0 on CPU."""

    peak_ram_gb: float
    """Peak resident memory of the process."""

    @property
    def peak_memory_gb(self) -> float:
        """The larger of the GPU and RAM peaks, which is what the competition scores."""
        return max(self.peak_gpu_gb, self.peak_ram_gb)


def peak_ram_gb() -> float:
    """Peak resident memory of this process in gigabytes."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024) / 1e9
    except ImportError:  # Windows has no `resource` module
        import psutil

        return psutil.Process().memory_info().rss / 1e9


def measure(
    model_directory: Path, device_name: str, passes: int = 3
) -> BenchmarkResult:
    """Time the full pipeline (feature extraction and model) one clip at a time.

    :param model_directory: Folder produced by `export_model`.
    :param device_name: "cuda" or "cpu".
    :param passes: Number of timed passes over the clips; the median is reported.
    """
    use_cuda = device_name == "cuda"
    device = torch.device(device_name)
    model = AutoModelForAudioClassification.from_pretrained(model_directory)
    model = model.to(
        device=device, dtype=torch.float16 if use_cuda else torch.float32
    ).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_directory)
    generator = numpy.random.default_rng(0)
    clips = [
        generator.standard_normal(SAMPLE_RATE * seconds).astype("float32")
        for seconds in CLIP_SECONDS
        for _ in range(CLIPS_PER_LENGTH)
    ]

    def classify(waveform: numpy.ndarray) -> torch.Tensor:
        features = feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = features["input_values"].to(device=device, dtype=model.dtype)
        with torch.no_grad():
            return model(
                input_values=input_values,
                attention_mask=features["attention_mask"].to(device),
            ).logits

    for waveform in clips[:WARMUP_CLIPS]:
        classify(waveform)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    pass_means = []
    for _ in range(passes):
        durations = []
        for waveform in clips:
            start = time.perf_counter()
            classify(waveform).cpu()
            durations.append(time.perf_counter() - start)
        pass_means.append(statistics.fmean(durations))
    return BenchmarkResult(
        device=device_name,
        seconds_per_clip=statistics.median(pass_means),
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0,
        peak_ram_gb=peak_ram_gb(),
    )


def measure_in_subprocess(model_directory: Path, device_name: str) -> BenchmarkResult:
    """Run `measure` in a new Python process so the notebook's own memory is not counted."""
    package_parent = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dsn_lid.benchmark",
            str(model_directory),
            "--device",
            device_name,
        ],
        capture_output=True,
        text=True,
        cwd=package_parent,
        check=True,
    )
    return BenchmarkResult(**json.loads(completed.stdout.strip().splitlines()[-1]))


def main() -> None:
    """Command-line entry point: print the result as one JSON line."""
    parser = argparse.ArgumentParser(
        description="Measure inference speed and memory of an exported model."
    )
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    arguments = parser.parse_args()
    result = measure(arguments.model_directory, arguments.device)
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    main()

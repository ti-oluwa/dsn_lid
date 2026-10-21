"""Fine-tuning loop with mixed precision, checkpoint resume and validation-based model selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import pandas
import torch
from transformers import (
    PreTrainedModel,
    SequenceFeatureExtractor,
    get_cosine_schedule_with_warmup,
)

from .augment import Augmenter
from .config import TrainingConfig
from .data import build_training_loader
from .evaluation import evaluate
from .modeling import split_trainable_parameters


class TrainingResult(NamedTuple):
    """What a training run produced."""

    best_f1: float | None
    """Best validation language-weighted F1, or None when no validation data was given."""

    history: list[tuple[int, float]]
    """Validation F1 recorded at each evaluation, as `(step, f1)`."""

    best_weights_path: Path
    """State dict of the best model (the final model when no validation data was given)."""


class Trainer:
    """Fine-tunes a classifier on class-balanced, augmented crops."""

    def __init__(
        self,
        model: PreTrainedModel,
        feature_extractor: SequenceFeatureExtractor,
        config: TrainingConfig,
        augmenter: Augmenter,
        device: torch.device,
        checkpoint_directory: Path,
        run_name: str,
    ) -> None:
        """
        :param model: Classifier to train; moved to `device`.
        :param feature_extractor: Normalises and pads waveforms.
        :param config: Optimisation and evaluation settings.
        :param augmenter: Applied to every training crop.
        :param device: Training device. fp16 autocast is used on CUDA only.
        :param checkpoint_directory: Where checkpoints and best weights are written.
        :param run_name: Prefix for the files written by this run.
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.config = config
        self.augmenter = augmenter
        self.device = device
        self.use_mixed_precision = device.type == "cuda"
        self.last_checkpoint_path = checkpoint_directory / f"{run_name}_last.pt"
        self.best_weights_path = checkpoint_directory / f"{run_name}_best.pt"

    def build_optimizer(self) -> torch.optim.AdamW:
        """AdamW with a lower learning rate for the pretrained encoder than for the new head."""
        encoder_parameters, head_parameters = split_trainable_parameters(self.model)
        return torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.config.encoder_learning_rate},
                {"params": head_parameters, "lr": self.config.head_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )

    def fit(
        self,
        train_manifest: pandas.DataFrame,
        validation_manifest: pandas.DataFrame | None = None,
        resume: bool = True,
    ) -> TrainingResult:
        """Train for `config.total_steps` optimizer steps.

        With a validation manifest, the weights with the best language-weighted F1 are kept.
        Without one, the final weights are kept.

        :param train_manifest: Training clips with `audio_path` and `label_index` columns.
        :param validation_manifest: Clips used to pick the best checkpoint, or None.
        :param resume: Continue from the last checkpoint of this run if one exists.
        """
        config = self.config
        optimizer = self.build_optimizer()
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.warmup_fraction * config.total_steps),
            num_training_steps=config.total_steps,
        )
        grad_scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_mixed_precision
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        completed_steps, best_f1, history = 0, None, []
        if resume and self.last_checkpoint_path.exists():
            checkpoint = torch.load(
                self.last_checkpoint_path, map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            grad_scaler.load_state_dict(checkpoint["grad_scaler"])
            completed_steps = checkpoint["completed_steps"]
            best_f1, history = checkpoint["best_f1"], checkpoint["history"]
            print(f"Resumed from step {completed_steps}")

        remaining_steps = config.total_steps - completed_steps
        if remaining_steps > 0:
            batches = iter(
                build_training_loader(
                    manifest=train_manifest,
                    feature_extractor=self.feature_extractor,
                    config=config,
                    augmenter=self.augmenter,
                    num_batches=remaining_steps * config.gradient_accumulation_steps,
                )
            )
            self.model.train()
            window_loss, window_start = 0.0, time.time()
            for step in range(completed_steps + 1, config.total_steps + 1):
                window_loss += self.run_optimizer_step(
                    batches, optimizer, grad_scaler, criterion
                )
                scheduler.step()

                if step % config.log_interval_steps == 0:
                    seconds_per_step = (
                        time.time() - window_start
                    ) / config.log_interval_steps
                    print(
                        f"step {step}/{config.total_steps}  loss {window_loss / config.log_interval_steps:.4f}"
                        f"  lr {scheduler.get_last_lr()[0]:.2e}  {seconds_per_step:.2f}s/step",
                        flush=True,
                    )
                    window_loss, window_start = 0.0, time.time()

                is_checkpoint_step = (
                    step % config.evaluation_interval_steps == 0
                    or step == config.total_steps
                )
                if not is_checkpoint_step:
                    continue
                if validation_manifest is not None:
                    result = evaluate(
                        model=self.model,
                        feature_extractor=self.feature_extractor,
                        manifest=validation_manifest,
                        device=self.device,
                        batch_size=config.evaluation_batch_size,
                        max_seconds=config.max_evaluation_seconds,
                    )
                    self.model.train()
                    history.append((step, result.weighted_f1))
                    if best_f1 is None or result.weighted_f1 > best_f1:
                        best_f1 = result.weighted_f1
                        torch.save(self.model.state_dict(), self.best_weights_path)
                    print(
                        f">>> step {step}  validation F1 {result.weighted_f1:.4f}  (best {best_f1:.4f})",
                        flush=True,
                    )
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "grad_scaler": grad_scaler.state_dict(),
                        "completed_steps": step,
                        "best_f1": best_f1,
                        "history": history,
                    },
                    self.last_checkpoint_path,
                )

        if validation_manifest is None:
            torch.save(self.model.state_dict(), self.best_weights_path)
        return TrainingResult(
            best_f1=best_f1, history=history, best_weights_path=self.best_weights_path
        )

    def run_optimizer_step(
        self,
        batches,
        optimizer: torch.optim.Optimizer,
        grad_scaler: torch.amp.GradScaler,
        criterion: torch.nn.Module,
    ) -> float:
        """Accumulate gradients over several batches and take one optimizer step.

        :param batches: Iterator of `Batch` objects.
        :returns: Mean loss over the accumulated batches.
        """
        accumulation = self.config.gradient_accumulation_steps
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accumulation):
            batch = next(batches)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_mixed_precision,
            ):
                logits = self.model(
                    input_values=batch.input_values.to(self.device),
                    attention_mask=batch.attention_mask.to(self.device),
                ).logits
            loss = (
                criterion(logits.float(), batch.labels.to(self.device)) / accumulation
            )
            grad_scaler.scale(loss).backward()
            step_loss += loss.item()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_gradient_norm
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        return step_loss

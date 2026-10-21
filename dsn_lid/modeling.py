"""Feature extractor and classification model construction."""

from __future__ import annotations

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    PreTrainedModel,
    SequenceFeatureExtractor,
)

from .config import ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS, SAMPLE_RATE, ModelConfig

HEAD_PARAMETER_PREFIXES = ("projector", "classifier", "layer_weights")
"""Name prefixes of the parameters that are trained from scratch."""


def build_feature_extractor(base_model: str) -> SequenceFeatureExtractor:
    """Load the base model's feature extractor with 16 kHz input, normalisation and attention masks."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model)
    feature_extractor.sampling_rate = SAMPLE_RATE
    feature_extractor.do_normalize = True
    feature_extractor.return_attention_mask = True
    return feature_extractor


def initialise_missing_mask_embedding(
    model: PreTrainedModel, missing_keys: list[str]
) -> None:
    """Initialise the SpecAugment mask embedding when the checkpoint does not contain it.

    Without this the parameter holds uninitialised memory and the first training forward pass can return NaN.

    :param model: Freshly loaded classifier.
    :param missing_keys: Parameter names the checkpoint lacked, from `output_loading_info`.
    """
    mask_embedding = getattr(model.base_model, "masked_spec_embed", None)
    if mask_embedding is None:
        return
    missing_mask_embedding = any(
        key.endswith("masked_spec_embed") for key in missing_keys
    )
    if missing_mask_embedding or not torch.isfinite(mask_embedding).all():
        torch.nn.init.uniform_(mask_embedding)


def build_model(config: ModelConfig) -> PreTrainedModel:
    """Load the pretrained encoder, keep `config.num_layers` layers and attach a 13-way head.

    The CNN feature encoder is frozen.
    """
    model, loading_info = AutoModelForAudioClassification.from_pretrained(
        config.base_model,
        num_hidden_layers=config.num_layers,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        use_weighted_layer_sum=True,
        # Layerdrop must stay off: when a layer is skipped, newer transformers versions return fewer
        # hidden states than the weighted layer sum has weights for, and the forward pass crashes.
        layerdrop=0.0,
        mask_time_prob=config.mask_time_probability,
        output_loading_info=True,
    )
    initialise_missing_mask_embedding(model, missing_keys=loading_info["missing_keys"])
    model.freeze_feature_encoder()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def split_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Separate trainable parameters into pretrained-encoder and newly added head groups.

    :returns: `(encoder_parameters, head_parameters)`.
    """
    encoder_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = name.startswith(HEAD_PARAMETER_PREFIXES)
        (head_parameters if is_head else encoder_parameters).append(parameter)
    return encoder_parameters, head_parameters

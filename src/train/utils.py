import logging
import os
import torch
import re

from transformers import Adafactor
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

import torch.optim as optim


def construct_optimizer(trainable_parameters, training_config):
    """


    Args:
        trainable_parameters:
        training_config:

    Returns:

    """
    weight_decay = training_config.weight_decay
    if weight_decay is None:
        weight_decay = 0.0

    if training_config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            trainable_parameters,
            lr=training_config.lr,
            weight_decay=weight_decay,
            eps=1e-6,
        )

    elif training_config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            trainable_parameters,
            lr=training_config.lr,
            weight_decay=weight_decay,
        )

    elif training_config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_parameters,
            lr=training_config.lr,
            weight_decay=weight_decay,
            eps=1e-8,
        )

    elif training_config.optimizer.lower() == "adafactor":
        optimizer = Adafactor(
            trainable_parameters,
            lr=training_config.lr,
            weight_decay=weight_decay,
            decay_rate=0,
            relative_step=False,
        )

    else:
        raise ValueError(f"Optimizer {training_config.optimizer} not implemented yet ")

    return optimizer


def construct_scheduler(optimizer, training_config):
    num_warmup_steps = training_config.num_batches * training_config.warmup_ratio

    if training_config.scheduler == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, training_config.num_batches
        )

    elif training_config.scheduler == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer)

    elif training_config.scheduler == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, training_config.num_batches
        )

    elif training_config.scheduler == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, training_config.num_batches
        )

    else:
        raise ValueError(f"scheduler {training_config.scheduler} not implemented")

import logging
import os
import torch
import copy
import re

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from src.utils.utils import getValueFromKey_matchingRegex

from src.model.ia3 import modify_withIA3
from src.model.EncoderDecoderWrapper import EncoderDecoderWrapper
from src.model.utils import *

MODEL_ARCHITECTURE = {
    ".*t5.*": "encoder_decoder",
}


def construct_model(model_config, device):
    """
    Args:
        model_name:
        peft_method:
        max_seq_len:
        device

    Returns:
        model:
        transformer:
        tokenizer:
    """
    model_architecture = getValueFromKey_matchingRegex(
        MODEL_ARCHITECTURE, model_config.pretrained_model
    )

    if model_architecture == "encoder_decoder":
        transformer = AutoModelForSeq2SeqLM.from_pretrained(
            model_config.pretrained_model
        )
        architecture_wrapper = EncoderDecoderWrapper
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.pretrained_model,
            model_max_length=model_config.max_seq_len,
        )
    else:
        raise NotImplementedError
    
    # Load PEFT method
    if model_config.peft_method is not None:
        if model_config.peft_method == "ia3":
            transformer, trainableParameter_regex = modify_withIA3(
                transformer, model_config
            )
        else:
            raise ValueError(f"Invalid PEFT method {model_config.peft_method}")
    else:
        trainableParameter_regex = ".*"

    # Load wrapper for computing loss and doing inference
    model = architecture_wrapper(transformer, tokenizer, model_config).to(device)

    return model, trainableParameter_regex


def loadCheckpoint_intoModel(model_config, checkpoint, model):
    """_summary_

    Args:
        checkpoint (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    modelParameters_names = set(checkpoint.keys())
    modelStateDict_keys = set(model.state_dict().keys())

    if not modelParameters_names.issubset(modelStateDict_keys):
        import ipdb
        ipdb.set_trace()

    # These embed_tokens in T5 are in the model architecture, but are tied to the
    # embedding matrix, and so should not be shared
    if "transformer.decoder.embed_tokens.weight" in modelStateDict_keys:
        assert "transformer.decoder.embed_tokens.weight" not in modelParameters_names
        assert "transformer.encoder.embed_tokens.weight" not in modelParameters_names

    # Must tie the encoder and decoder embeddings to the shared weight if the shared weight is a parameter.
    if "transformer.shared.weight" in checkpoint:
        checkpoint["transformer.decoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]
        checkpoint["transformer.encoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]

    model.load_state_dict(checkpoint, strict=False)

    if "transformer.shared.weight" in checkpoint:
        del checkpoint["transformer.decoder.embed_tokens.weight"]
        del checkpoint["transformer.encoder.embed_tokens.weight"]

    return model


def load_model(model_config, device):
    """

    Args:
        model_config:
        device:

    Returns:

    """

    model, trainableParameter_regex = construct_model(
        model_config,
        device=device,
    )

    if model_config.filepath_to_load_model is not None:
        model = loadCheckpoint_intoModel(
            model_config, torch.load(model_config.filepath_to_load_model), model
        )

    return model, trainableParameter_regex

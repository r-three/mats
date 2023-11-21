import argparse
from tqdm import tqdm
import torch
import os
import logging
import numpy as np
import collections
import re
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from src.train.TrainingConfig import TrainingConfig
from src.model.ModelConfig import ModelConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig

from src.data.batches import *
from src.data.dataset_readers import get_datasetReader, getDatasets_inMixture
from src.data.PytorchDataset import PytorchDataset

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.utils.model_utils import *

from src.utils.utils import *
from src.eval.utils import *


def normalize_activations(stored_activations, num_encoderTokens, num_decoderTokens):
    normalized_activations = {}
    for module_name, parameter in stored_activations.items():
        if re.fullmatch(".*encoder.*|.*decoder.*EncDecAttention.(k|v).*", module_name):
            num_tokens = num_encoderTokens
        else:
            num_tokens = num_decoderTokens
        normalized_activations[module_name] = parameter / num_tokens
    return normalized_activations


def get_gramMatrixDir(modelCheckpoint_fp):
    exp_dir = os.path.dirname(modelCheckpoint_fp).replace("/checkpoints", "")
    checkpoint_filename = os.path.basename(modelCheckpoint_fp)
    activations_dir = os.path.join(exp_dir, "gram_matrix")
    safe_makedirs(activations_dir)
    return activations_dir, checkpoint_filename


def getInputActivationGramMatrix_fp(
    modelCheckpoint_fp, split, use_trueFisher
):
    gramMatrix_dir, checkpoint_filename = get_gramMatrixDir(modelCheckpoint_fp)

    if use_trueFisher:
        newCheckpoint_fp = f"_true_gram_matrix_input_activations_{split}"
    else:
        newCheckpoint_fp = f"_gram_matrix_input_activations_{split}"

    gramMatrixInputActivations_fp = os.path.join(
        gramMatrix_dir,
        checkpoint_filename.replace(".pt", newCheckpoint_fp + ".pt"),
    )

    return gramMatrixInputActivations_fp


def getOutputActivationGradientGramMatrix_fp(
    modelCheckpoint_fp, split, use_trueFisher
):
    activations_dir, checkpoint_filename = get_gramMatrixDir(modelCheckpoint_fp)

    if use_trueFisher:
        newCheckpoint_fp = f"_true_gram_matrix_output_activation_gradients_{split}"
    else:
        newCheckpoint_fp = f"_gram_matrix_output_activation_gradients_{split}"

    gramMatrixOutputActivationGradients_fp = os.path.join(
        activations_dir,
        checkpoint_filename.replace(
            ".pt",
            f"{newCheckpoint_fp}.pt",
        ),
    )

    return gramMatrixOutputActivationGradients_fp


def save_activations(
    saved_activations,
    module_name,
    module,
    input,
    output,
) -> None:
    """PyTorch Forward hook to save inputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    saved_activations[module_name] = input[0].float()


def save_gradients(
    saved_gradients,
    module_name,
    module,
    grad_input,
    grad_output,
) -> None:
    """PyTorch Backward hook to save inputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    saved_gradients[module_name] = grad_output[0].float()


def computeGramMatrix_forLinearLayer(module_name, matrix, mask):
    # [batch_size * num_tokens, input_dim]
    masked_activations = (
        matrix.flatten(0, 1) * mask.flatten(0, 1).to(matrix.device)[:, None]
    )

    return torch.matmul(masked_activations.T, masked_activations)


def update_gramMatrix(module_name, gram_matrix, runningSum_gramMatrix):
    if module_name not in runningSum_gramMatrix:
        runningSum_gramMatrix[module_name] = gram_matrix
    else:
        runningSum_gramMatrix[module_name] += gram_matrix

    return runningSum_gramMatrix


def saveGramMatrix_perModel(
    device,
    world_size,
    model_config,
    evaluation_config,
    checkpoint_descriptor,
    split,
    use_trueFisher,
):
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    modelCheckpoint_fp = get_modelCheckpointFilepath(
        model_config.pretrained_model,
        checkpoint_descriptor,
        evaluationDataset_config.instruction_format,
        dataset,
    )
    model, _ = loadModel_fromCheckpointfp(model_config, modelCheckpoint_fp, device)
    model.eval()

    dataset_reader, _ = get_datasetReader(
        dataset_mixture=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders={},
    )
    metrics = dataset_reader.get_datasetMetrics()

    pytorch_dataset = PytorchDataset(
        dataset_reader.get_dataset(
            split,
            evaluationDataset_config.template_idx,
            use_answerChoices=use_trueFisher,
            num_samples=-1,
        ),
        evaluationDataset_config,
        model.tokenize_fn,
        device=device,
    )

    iterator = getSingleEpoch_OfBatches(
        pytorch_dataset,
        batch_size=1,
        world_size=world_size,
        device=device,
    )

    stored_inputActivations = {}
    stored_outputActivationsGradients = {}

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(
                partial(
                    save_activations,
                    stored_inputActivations,
                    module_name,
                )
            )
            # output activation for lm_head is vocab_size, which is too large to store
            if "lm_head" not in module_name:
                module.register_full_backward_hook(
                    partial(
                        save_gradients,
                        stored_outputActivationsGradients,
                        module_name,
                    )
                )

    inputActivations_gramMatrix = {}
    outputActivationGradient_gramMatrix = {}

    total_numEncoderTokens = 0
    total_numDecoderTokens = 0
    num_samples = 0

    for idx, batch in tqdm(enumerate(iterator)):
        # When computing the true fisher, we have to sample the label from the predicted
        # distribution
        if use_trueFisher:
            batch = sample_label(model, batch, evaluation_config, metrics)

        # Fisher is the gradient of the log likelihood, so the log prob of the training lbl
        loss, _ = model(batch)
        log_prob = -loss
        log_prob.backward()

        num_encoderTokens = None
        num_decoderTokens = None

        with torch.no_grad():
            for module_name, activations in stored_inputActivations.items():
                if re.fullmatch(
                    ".*encoder.*|.*decoder.*EncDecAttention.(k|v).*", module_name
                ):
                    key = "input_mask"
                    num_encoderTokens = torch.sum(batch[key])
                else:
                    key = "target_mask"
                    num_decoderTokens = torch.sum(batch[key])

                mask = batch[key]

                inputActivations_gramMatrix = update_gramMatrix(
                    module_name,
                    computeGramMatrix_forLinearLayer(module_name, activations, mask),
                    inputActivations_gramMatrix,
                )

                if module_name != "transformer.lm_head":
                    gradients = stored_outputActivationsGradients[module_name]

                    outputActivationGradient_gramMatrix = update_gramMatrix(
                        module_name,
                        computeGramMatrix_forLinearLayer(module_name, gradients, mask),
                        outputActivationGradient_gramMatrix,
                    )

        assert num_encoderTokens is not None
        assert num_decoderTokens is not None

        model.zero_grad()
        total_numEncoderTokens += num_encoderTokens
        total_numDecoderTokens += num_decoderTokens
        num_samples += 1

        if num_samples >= 1000:
            break

    with torch.no_grad():
        inputActivations_gramMatrix = normalize_activations(
            inputActivations_gramMatrix, total_numEncoderTokens, total_numDecoderTokens
        )
        outputActivationGradient_gramMatrix = normalize_activations(
            outputActivationGradient_gramMatrix,
            total_numEncoderTokens,
            total_numDecoderTokens,
        )
    print(
        f"{evaluationDataset_config.dataset} has {num_samples} samples with {total_numEncoderTokens} encoder tokens and {total_numDecoderTokens} decoder tokens"
    )
    torch.save(
        detach_metadata(inputActivations_gramMatrix),
        open(
            getInputActivationGramMatrix_fp(
                modelCheckpoint_fp, split, use_trueFisher
            ),
            "wb",
        ),
    )
    torch.save(
        detach_metadata(outputActivationGradient_gramMatrix),
        open(
            getOutputActivationGradientGramMatrix_fp(
                modelCheckpoint_fp, split, use_trueFisher
            ),
            "wb",
        ),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser.add_argument("-d", "--dataset_mixture", type=str, required=True)
    parser.add_argument("--checkpoint_descriptor", type=str, default="full_model")
    parser.add_argument(
        "-s", "--split", type=str, choices=["train", "validation"], required=True
    )
    parser.add_argument("--use_true_fisher", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config, evaluationDataset_config, evaluation_config = getInference_configs(
        args
    )

    for dataset in getDatasets_inMixture(args.dataset_mixture):
        newEvaluation_config = get_newEvaluationConfig(
            evaluation_config, {"dataset": dataset}, {}
        )
        saveGramMatrix_perModel(
            device,
            1,
            model_config,
            newEvaluation_config,
            args.checkpoint_descriptor,
            args.split,
            args.use_true_fisher,
        )

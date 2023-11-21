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

from src.data.batches import getSingleEpoch_OfBatches
from src.data.dataset_readers import get_datasetReader, getDatasets_inMixture
from src.data.PytorchDataset import PytorchDataset

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.utils.model_utils import *

from src.utils.utils import *
from src.eval.utils import *

from src.merging.save_metadata.save_gram_matrix import get_gramMatrixDir


def compute_blockwiseFisher(model, trainableParameter_regex):
    perExample_fisher = {}
    model_parameters = dict((key, value) for key, value in model.named_parameters())
    for parameter_name, parameter in model_parameters.items():
        if (
            re.fullmatch(trainableParameter_regex, parameter_name)
            and parameter.requires_grad
        ):
            flattened_grad = torch.flatten(parameter.grad)
            outer_product = torch.outer(flattened_grad, flattened_grad)

            if parameter_name not in perExample_fisher:
                perExample_fisher[parameter_name] = outer_product.detach()
            else:
                perExample_fisher[parameter_name] += outer_product.detach()
    return perExample_fisher


def compute_diagonalFisher(model, trainableParameter_regex):
    perExample_fisher = {}
    for parameter_name, parameter in model.named_parameters():
        if (
            re.fullmatch(trainableParameter_regex, parameter_name)
            and parameter.requires_grad
        ):
            if parameter_name not in perExample_fisher:
                perExample_fisher[parameter_name] = torch.zeros_like(parameter.data)
            perExample_fisher[parameter_name] += torch.square(parameter.grad)
    return perExample_fisher

def get_fisherArgumentString(
    fisher_approximation,
    split,
    use_trueFisher,
):
    if use_trueFisher:
        fisher_estimate = "true"
    else:
        fisher_estimate = "empirical"

    fisherArguments_string = f"{fisher_approximation}_{fisher_estimate}_fisher_{split}"
    return fisherArguments_string


def get_fisherFP(
    modelCheckpoint_fp,
    split,
    use_trueFisher,
    fisher_approximation,
):
    fisher_dir, checkpoint_filename = get_gramMatrixDir(modelCheckpoint_fp)

    fisher_fp = os.path.join(
        fisher_dir,
        checkpoint_filename.replace(
            ".pt",
            f"_{get_fisherArgumentString(fisher_approximation, split, use_trueFisher)}.pt",
        ),
    )

    return fisher_fp


def getFisher_perModel(
    device,
    world_size,
    model_config,
    evaluation_config,
    checkpoint_descriptor,
    split,
    use_trueFisher,
    fisher_approximation
):
    assert fisher_approximation in ["diagonal", "blockwise"]
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    modelCheckpoint_fp = get_modelCheckpointFilepath(
        model_config.pretrained_model,
        checkpoint_descriptor,
        evaluationDataset_config.instruction_format,
        dataset,
    )
    model, trainableParameter_regex = loadModel_fromCheckpointfp(
        model_config, modelCheckpoint_fp, device
    )

    model.eval()

    dataset_reader, _ = get_datasetReader(
        dataset_mixture=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders={},
    )
    metrics = dataset_reader.get_datasetMetrics()

    # answer_choices are added if we are computing the true fisher since we have to
    # score each answer_choice and sample from that distribution
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

    stored_fisher = {}

    def update_storedFisher(perExample_fisher):
        for parameter_name, value in perExample_fisher.items():
            if parameter_name not in stored_fisher:
                stored_fisher[parameter_name] = value
            else:
                stored_fisher[parameter_name] += value

    num_samples = 0

    for idx, batch in tqdm(enumerate(iterator)):
        # When computing the true fisher, we have to sample the label from the predicted
        # distribution
        if use_trueFisher:
            batch = sample_label(model, batch, evaluation_config, metrics)

        loss, _ = model(batch)
        # Fisher is the gradient of the log likelihood (which is the negative loss of the log prob )
        log_prob = -loss
        log_prob.backward()

        # Compute the per-example Fisher and update the total Fisher
        with torch.no_grad():
            if fisher_approximation == "diagonal":
                perExample_fisher = compute_diagonalFisher(
                    model, trainableParameter_regex
                )
            elif fisher_approximation == "blockwise":
                perExample_fisher = compute_blockwiseFisher(
                    model, trainableParameter_regex
                )
            else:
                raise NotImplementedError
            update_storedFisher(perExample_fisher)
        num_samples += 1
        model.zero_grad()

        if num_samples >= 1000:
            break

    with torch.no_grad():
        stored_fisher = normalize_metadata(stored_fisher, num_samples)

    print(f"{evaluationDataset_config.dataset} has {num_samples} samples")

    fisher_fp = get_fisherFP(
        modelCheckpoint_fp,
        split,
        use_trueFisher,
        fisher_approximation
    )

    torch.save(detach_metadata(stored_fisher), fisher_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addFisherArguments_toParser(parser)
    parser.add_argument("-d", "--dataset_mixture", type=str, required=True)
    parser.add_argument("--checkpoint_descriptor", type=str, default="full_model")
    args = parser.parse_args()

    model_config, evaluationDataset_config, evaluation_config = getInference_configs(
        args
    )

    assert evaluation_config.use_bfloat16_during_eval is None
    assert evaluation_config.sample_tokens is None

    if args.fisher_approximation == "blockwise":
        assert "ia3" in args.checkpoint_descriptor, "Blockwise only works for IA3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in getDatasets_inMixture(args.dataset_mixture):
        newEvaluation_config = get_newEvaluationConfig(
            evaluation_config,
            {"dataset": dataset},
            {},
        )
        getFisher_perModel(
            device,
            1,
            model_config,
            newEvaluation_config,
            args.checkpoint_descriptor,
            args.split,
            args.use_true_fisher,
            args.fisher_approximation,
        )

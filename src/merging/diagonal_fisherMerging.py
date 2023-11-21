import argparse
import logging
import os
import torch
import copy
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel

from src.train.TrainingConfig import TrainingConfig
from src.model.ModelConfig import ModelConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig

from src.inference import *

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.save_metadata.save_fisher import *
from src.model.utils import *


def diagonal_fisherMerging(
    model_lambda,
    loaded_checkpoints,
    loaded_fisher,
):
    """

    Args:
        model_lambda:
        loaded_checkpoints: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        loaded_fisher:

    Returns:

    """

    logging.info(
        f"Using Diagonal Fisher Merging to merge {len(loaded_checkpoints)} checkpoints"
    )
    start = time.time()

    # Assume model_lambda must be 1 unless we are scaling different checkpoints by different amounts, in which case we use lambda and (1 - lambda), which only holds for 2 checkpoints
    if model_lambda != 1.0:
        assert len(loaded_checkpoints) == 2

    checkpoints_andFisherMatrices = {}
    for checkpoint_fp, checkpoint in loaded_checkpoints.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andFisherMatrices[dataset] = {"checkpoint": checkpoint}

    for activation_fp, fisher in loaded_fisher.items():
        dataset = getDataset_fromCheckpointFilepath(activation_fp)
        checkpoints_andFisherMatrices[dataset].update({"fisher": fisher})

    listOf_fisherWeightedCheckpoints = []
    listOf_fishers = []

    for dataset, checkpoint_andFisherMatrix in checkpoints_andFisherMatrices.items():
        checkpoint = checkpoint_andFisherMatrix["checkpoint"]
        fisher = checkpoint_andFisherMatrix["fisher"]

        fisher = set_minimum(fisher, 1e-8)

        # Must scale checkpoint
        if model_lambda != 1.0:
            if len(listOf_fishers) == 0:
                fisher = scale(fisher, model_lambda)

            else:
                assert len(listOf_fishers) == 1
                fisher = scale(fisher, (1 - model_lambda))

        fisherWeighted_checkpoint = pairwiseMap_modelParameters(
            checkpoint, fisher, lambda x, y: x * y
        )

        listOf_fisherWeightedCheckpoints.append(fisherWeighted_checkpoint)
        listOf_fishers.append(fisher)

    summed_model = divide(
        scale_andSum(listOf_fisherWeightedCheckpoints, 1),
        scale_andSum(listOf_fishers, 1),
    )
    end = time.time()
    diff_time = end - start

    return summed_model, diff_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addFisherArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)
    parser.add_argument("--model_lambda", default=1.0)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.fisher_approximation = "diagonal"

    model_config, evaluationDataset_config, evaluation_config = getInference_configs(
        args
    )

    checkpoints_fp, loaded_checkpoints = loadCheckpoints_toMerge(
        model_config.pretrained_model,
        args.checkpoint_descriptor,
        evaluationDataset_config.instruction_format,
        args.dataset_mixture_to_merge,
        torch.device("cpu"),
    )
    loaded_diagonalFishers = {}
    for checkpoint_fp in checkpoints_fp:
        diagonal_fisher = torch.load(
            get_fisherFP(
                checkpoint_fp,
                args.split,
                args.use_true_fisher,
                "diagonal",
            ),
            torch.device("cpu"),
        )

        loaded_diagonalFishers[checkpoint_fp] = diagonal_fisher

    if args.time:
        total_diffTime = []
        for i in range(10):
            _, diff_time = diagonal_fisherMerging(
                model_lambda=1.0,
                loaded_checkpoints=loaded_checkpoints,
                loaded_fisher=loaded_diagonalFishers,
            )
            total_diffTime.append(diff_time)
        total_diffTime = np.asarray(total_diffTime)
        print("Mean: ", np.mean(total_diffTime))
        print("Std: ", np.std(total_diffTime))
        print(total_diffTime)
    else:
        list_modelLambda = get_listModelLambda(args.model_lambda)
        for model_lambda in list_modelLambda:
            merged_model, _ = diagonal_fisherMerging(
                model_lambda=model_lambda,
                loaded_checkpoints=loaded_checkpoints,
                loaded_fisher=loaded_diagonalFishers,
            )

            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join(
                    "fisher_merging",
                    get_fisherArgumentString(
                        args.fisher_approximation,
                        args.split,
                        args.use_true_fisher,
                    ),
                    f"model_lambda_{model_lambda}",
                ),
            )
            safe_makedirs(experiment_dir)
            mergedCheckpoint_fp = os.path.join(experiment_dir, "merged_model.pt")
            print(mergedCheckpoint_fp)
            torch.save(merged_model, mergedCheckpoint_fp)
            new_modelConfig = get_newModelConfig(
                model_config, {"filepath_to_load_model": mergedCheckpoint_fp}
            )

            inference_datasetMixture = args.inference_dataset_mixture
            if inference_datasetMixture is None:
                inference_datasetMixture = args.dataset_mixture_to_merge
            new_evaluationConfig = get_newEvaluationConfig(
                evaluation_config, None, {"dataset_mixture": inference_datasetMixture}
            )

            if args.multiple_prompts:
                inference_fn = inference_withMultiplePrompts
            else:
                inference_fn = inference_withSinglePrompt

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inference_fn(
                device,
                1,
                None,
                new_modelConfig,
                new_evaluationConfig,
                experiment_dir,
                title=None,
                cached_singleDatasetReaders=None,
            )

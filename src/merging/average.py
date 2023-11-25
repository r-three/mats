import argparse
import logging
import os
import torch
import copy

from src.data.dataset_readers import get_datasetMixtureReader

from src.train.TrainingConfig import TrainingConfig

from src.model.ModelConfig import ModelConfig

from src.utils.utils import format_modelName, ParseKwargs

from src.model.load_model import load_model

from src.inference import *

from src.train.TrainingConfig import TrainingConfig
from src.model.ModelConfig import ModelConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.save_metadata.save_fisher import *
from src.model.utils import *
import time


def average(model_lambda, loaded_checkpoints):
    """

    Args:
        model_lambda:
        loaded_checkpoints: Dictionary mapping checkpoint filepaths to dictionary of parameters.

    Returns:

    """
    checkpoints = list(loaded_checkpoints.values())
    start = time.time()

    logging.info(
        f"Taking the average of {len(checkpoints)} checkpoints with lambda {model_lambda}"
    )

    if len(checkpoints) == 2:
        scaled_checkpoints = [
            scale(checkpoints[0], model_lambda),
            scale(checkpoints[1], (1 - model_lambda)),
        ]
        checkpoints = scaled_checkpoints
        scaling_factor = 1
    else:
        scaling_factor = model_lambda / len(checkpoints)

    # Divide by number of checkpoints to get the average.
    average_model = scale_andSum(checkpoints, scaling_factor)

    end = time.time()
    diff_time = end - start

    return average_model, diff_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)
    parser.add_argument("--model_lambda", default=1.0)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if args.time:
        total_diffTime = []
        for i in range(10):
            _, diff_time = average(1.0, loaded_checkpoints)
            total_diffTime.append(diff_time)
        total_diffTime = np.asarray(total_diffTime)
        print("Mean: ", np.mean(total_diffTime))
        print("Std: ", np.std(total_diffTime))
    else:
        list_modelLambda = get_listModelLambda(args.model_lambda)
        for model_lambda in list_modelLambda:
            merged_model, _ = average(model_lambda, loaded_checkpoints)

            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join("average", f"model_lambda_{model_lambda}"),
            )
            safe_makedirs(experiment_dir)
            mergedCheckpoint_fp = os.path.join(experiment_dir, "merged_model.pt")
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

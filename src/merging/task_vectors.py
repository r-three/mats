import argparse
import logging
import os
import torch
import copy

from src.train.TrainingConfig import TrainingConfig

from src.model.ModelConfig import ModelConfig
from src.model.utils import get_trainableParameters

from src.train.TrainingConfig import TrainingConfig
from src.model.ModelConfig import ModelConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig

from src.inference import *

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *

from src.model.utils import *


def get_taskVectors(task_models, pretrained_model):
    """

    Args:
        task_models:
        pretrained_model:

    Returns:

    """

    taskVector_models = map_forDictionaries(
        task_models, lambda checkpoint: subtract(checkpoint, pretrained_model)
    )
    return taskVector_models


def task_vectors(model_lambda, loadedModels_parameters, pretrained_parameters):
    """

    Args:
        model_lambda:
        loadedModels_parameters: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        pretrained_parameters: Dictionary of parameters.

    Returns:

    """
    assert pretrained_parameters is not None
    logging.info(
        f"Taking the task vectors of {len(loadedModels_parameters)} checkpoints with lambda {model_lambda}"
    )
    start = time.time()

    taskVector_models = get_taskVectors(loadedModels_parameters, pretrained_parameters)
    taskVector_models = list(taskVector_models.values())

    summed_model = scale_andSum(taskVector_models, model_lambda)
    final_model = add(summed_model, pretrained_parameters)
    end = time.time()
    diff_time = end - start

    return final_model, diff_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)
    parser.add_argument("--model_lambda", type=float)
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

    pretrained_model, trainableParameter_regex = load_model(
        model_config, torch.device("cpu")
    )
    pretrained_checkpoint, _ = get_trainableParameters(
        pretrained_model, trainableParameter_regex
    )

    if args.time:
        total_diffTime = []
        for i in range(10):
            _, diff_time = task_vectors(
                model_lambda=0.3,
                loadedModels_parameters=loaded_checkpoints,
                pretrained_parameters=pretrained_checkpoint,
            )
            total_diffTime.append(diff_time)
        total_diffTime = np.asarray(total_diffTime)
        print("Mean: ", np.mean(total_diffTime))
        print("Std: ", np.std(total_diffTime))
    else:
        list_modelLambda = get_listModelLambda(args.model_lambda)

        for model_lambda in list_modelLambda:
            merged_model, _ = task_vectors(
                model_lambda=model_lambda,
                loadedModels_parameters=loaded_checkpoints,
                pretrained_parameters=pretrained_checkpoint,
            )
            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join("task_vectors", f"model_lambda_{model_lambda}"),
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

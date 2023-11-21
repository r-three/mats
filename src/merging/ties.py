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
from src.merging.task_vectors import get_taskVectors


def get_flattenedTaskVectors(task_models, pretrained_model):
    """

    Args:
        task_models:
        pretrained_model:

    Returns:

    """

    taskVector_models = map_forDictionaries(
        task_models,
        lambda checkpoint: flatten_model(subtract(checkpoint, pretrained_model)),
    )
    return taskVector_models


def flatten_model(model):
    all_parameters = []
    for parameter_name, parameter_value in model.items():
        all_parameters.append(parameter_value.flatten())
    stacked_parameters = torch.cat(all_parameters, dim=0)
    return stacked_parameters


# From https://github.com/prateeky2806/ties-merging/tree/main/src
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks,
    reset_thresh=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


def aggregate(T, agg_type, dim=0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    return result


def tv_merging(tv_flat_checks):
    """Merging by creating and scaling Task Vectors"""
    all_checks = tv_flat_checks.clone()
    tv_merged_check = aggregate(all_checks, "sum")
    return tv_merged_check


def ties(model_lambda, loadedModels_parameters, pretrained_parameters):
    """

    Args:
        model_lambda:
        loadedModels_parameters: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        pretrained_parameters: Dictionary of parameters.

    Returns:

    """
    assert pretrained_parameters is not None
    logging.info(
        f"Taking the TIES of {len(loadedModels_parameters)} checkpoints with lambda {model_lambda}"
    )
    start = time.time()
    taskVector_models = get_flattenedTaskVectors(
        loadedModels_parameters, pretrained_parameters
    )
    taskVector_models = torch.stack(list(taskVector_models.values()), dim=0)
    merged_tv = ties_merging(taskVector_models, 20, "dis-mean")
    merged_model = flatten_model(pretrained_parameters) + model_lambda * merged_tv

    start_idx = 0
    final_model = {}
    for parameter_name, parameter_value in pretrained_parameters.items():
        parameter_size = parameter_value.numel()
        end_idx = start_idx + parameter_size
        final_model[parameter_name] = merged_model[start_idx:end_idx].reshape(
            parameter_value.shape
        )
        start_idx = end_idx
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
            _, diff_time = ties(
                model_lambda=1.0,
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
            merged_model, _ = ties(
                model_lambda=model_lambda,
                loadedModels_parameters=loaded_checkpoints,
                pretrained_parameters=pretrained_checkpoint,
            )

            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join("ties", f"model_lambda_{model_lambda}"),
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

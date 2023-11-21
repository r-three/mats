import argparse
import logging
import os
import torch
import json
import copy
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel


from src.train.TrainingConfig import TrainingConfig

from src.model.load_model import load_model
from src.model.ModelConfig import ModelConfig

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.evaluate import evaluate_singleTemplate, evaluate_multipleTemplates
from src.eval.utils import concatenate_scores, saveResult_acrossDatasets

from src.data.dataset_readers import getDatasets_inMixture
from src.data.DatasetConfig import DatasetConfig

from src.utils.distributed import is_nodeZero, is_distributedSetup
from src.utils.utils import *


def inference_withMultiplePrompts(
    device,
    world_size,
    port,
    model_config,
    evaluation_config,
    experiment_dir,
    title,
    cached_singleDatasetReaders,
):
    """

    Args:
        model_config:
        evaluation_config:
        experiment_dir
        cached_singleDatasetReaders:
        world_size:
        device:

    Returns:

    """
    if is_distributedSetup(world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=device, world_size=world_size)

    model, _ = load_model(model_config, device=device)

    if is_distributedSetup(world_size):
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    (
        multipleTemplate_scores,
        runs_dir,
        cached_singleDatasetReaders,
    ) = evaluate_multipleTemplates(
        model,
        evaluation_config,
        os.path.join(experiment_dir, "predictions", "inference"),
        cached_singleDatasetReaders,
        world_size,
        device,
    )

    inferenceScores_fp = os.path.join(experiment_dir, "inference_scores")

    if multipleTemplate_scores is not None:
        append_json(
            {"scores": multipleTemplate_scores, "runs_dir": runs_dir},
            inferenceScores_fp + ".json",
        )

        def getScore_fn(dataset_score):
            return (
                f"{round(dataset_score['median'] * 100, 1)} "
                f"({round(dataset_score['interquartile_range'] * 100, 1)})"
            )

        if evaluation_config.dataset_mixture is not None:
            datasets = getDatasets_inMixture(evaluation_config.dataset_mixture)
            saveAverage_acrossDatasets = True
        else:
            datasets = evaluation_config.get_datsetConfig().dataset
            saveAverage_acrossDatasets = False

        saveResult_acrossDatasets(
            datasets,
            multipleTemplate_scores,
            getScore_fn,
            inferenceScores_fp + ".txt",
            saveAverage_acrossDatasets=saveAverage_acrossDatasets,
            title=title,
        )

    return cached_singleDatasetReaders


def inference_withSinglePrompt(
    device,
    world_size,
    port,
    model_config,
    evaluation_config,
    experiment_dir,
    title,
    cached_singleDatasetReaders,
):
    """

    Args:
        model_config:
        evaluation_config:
        experiment_dir
        cached_singleDatasetReaders:
        world_size:
        device:

    Returns:

    """
    if is_distributedSetup(world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=device, world_size=world_size)

    model, _ = load_model(model_config, device=device)

    if is_distributedSetup(world_size):
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    (
        singleTemplate_scores,
        runs_dir,
        cached_singleDatasetReaders,
    ) = evaluate_singleTemplate(
        model,
        evaluation_config,
        os.path.join(experiment_dir, "predictions", "inference"),
        cached_singleDatasetReaders,
        world_size,
        device,
    )

    inferenceScores_fp = os.path.join(
        experiment_dir, f"{evaluation_config.get_datasetConfig().split}_scores"
    )

    if singleTemplate_scores is not None:
        append_json(
            {"scores": singleTemplate_scores, "runs_dir": runs_dir},
            inferenceScores_fp + ".json",
        )

        def getScore_fn(split, dataset_score):
            if split is None:
                return str(round(dataset_score["average"] * 100, 1))
            else:
                return str(round(dataset_score[split]["average"] * 100, 1))

        if evaluation_config.dataset_mixture is not None:
            datasets = getDatasets_inMixture(evaluation_config.dataset_mixture)
            saveAverage_acrossDatasets = True
        else:
            datasets = [evaluation_config.get_datasetConfig().dataset]
            saveAverage_acrossDatasets = False

        if evaluation_config.get_datasetConfig().split == "train_validation":
            saveResult_acrossDatasets(
                datasets,
                singleTemplate_scores,
                "train",
                lambda x: getScore_fn("train", x),
                inferenceScores_fp + ".txt",
                saveAverage_acrossDatasets=saveAverage_acrossDatasets,
                title=title,
            )
            saveResult_acrossDatasets(
                datasets,
                singleTemplate_scores,
                "validation",
                lambda x: getScore_fn("validation", x),
                inferenceScores_fp + ".txt",
                saveAverage_acrossDatasets=saveAverage_acrossDatasets,
                title=title,
            )
        else:
            saveResult_acrossDatasets(
                datasets,
                singleTemplate_scores,
                None,
                lambda x: getScore_fn(None, x),
                inferenceScores_fp + ".txt",
                saveAverage_acrossDatasets=saveAverage_acrossDatasets,
                title=title,
            )
    return cached_singleDatasetReaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_model", type=str)
    parser.add_argument("-e", "--experiment_dir", type=str)
    parser.add_argument("--checkpoint_idx", type=int)

    parser.add_argument("-c", "--config_filepaths", action="store", type=str, nargs="*")
    parser.add_argument(
        "-m", "--model_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-ed", "--evaluation_dataset_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-er", "--evaluation_run_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument("-i", "--inference_dataset_mixture", type=str)
    parser.add_argument("--multiple_prompts", action="store_true")
    parser.add_argument("-w", "--world_size", default=1, type=int)
    parser.add_argument("-p", "--port", default=12345, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.merged_model is not None:
        if "p3" in args.merged_model:
            args.config_filepaths = [
                "configs/evaluation_dataset/p3_validation.json",
                "configs/evaluation_run/individual_task.json",
                "configs/models/t5_large.json",
            ]
        else:
            args.config_filepaths = [
                "configs/evaluation_dataset/flan_validation.json",
                "configs/evaluation_run/individual_task.json",
                "configs/models/t5_large.json",
            ]
        evaluation_prefix = "evaluation"
        if "ia3" in args.merged_model:
            args.config_filepaths.append("configs/models/ia3.json")

        args.model_kwargs = {"filepath_to_load_model": args.merged_model}
    else:
        evaluation_prefix = "evaluation"

    if args.experiment_dir is not None:
        if args.checkpoint_idx is None:
            checkpoint_idx = get_bestCheckpointIdx(args.experiment_dir)
        else:
            checkpoint_idx = args.checkpoint_idx

        dataset = args.experiment_dir.split("/")[2]

        args.config_filepaths = [
            os.path.join(args.experiment_dir, "model_config.json"),
            "configs/evaluation_dataset/p3_validation.json",
            "configs/evaluation_run/individual_task.json",
        ]
        evaluation_prefix = "evaluation"

        args.model_kwargs = {
            "filepath_to_load_model": os.path.join(
                args.experiment_dir,
                "checkpoints",
                f"checkpoint_{checkpoint_idx}.pt",
            )
        }

        args.evaluation_dataset_kwargs = {"evaluation_dataset": dataset}

    if args.inference_dataset_mixture is not None:
        args.evaluation_run_kwargs.update(
            {"dataset_mixture": args.inference_dataset_mixture}
        )

    parsed_filepaths = parse_configFilepaths(
        args.config_filepaths,
        types=[
            "models",
            "evaluation_dataset",
            "evaluation_run",
        ],
    )

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference on the best model found during training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        config_filepaths=parsed_filepaths["models"], update_dict=args.model_kwargs
    )

    evaluationDataset_config = DatasetConfig(
        evaluation_prefix,
        config_filepaths=parsed_filepaths["evaluation_dataset"],
        update_dict=args.evaluation_dataset_kwargs,
    )
    evaluation_config = EvaluationConfig(
        evaluationDataset_config,
        config_filepaths=parsed_filepaths["evaluation_run"],
        update_dict=args.evaluation_run_kwargs,
    )

    filepath_forExperimentDir = model_config.filepath_to_load_model

    # Use pre-trained model
    if filepath_forExperimentDir is None:
        experiment_dir = "exp_out"
        experiment_dir = os.path.join(
            experiment_dir, evaluationDataset_config.get_experimentDir()
        )
        experiment_dir = os.path.join(experiment_dir, model_config.get_experimentDir())

    else:
        # If checkpoints is in filepath, then we assume we are doing inference on a particular checkpoint and the scores should be score with respect to this particular checkpoint
        if (
            "checkpoints" in filepath_forExperimentDir
            and evaluation_config.get_datasetConfig().split != "test"
        ):
            experiment_dir = os.path.dirname(os.path.dirname(filepath_forExperimentDir))
            checkpoint_name = os.path.basename(filepath_forExperimentDir).replace(
                ".pt", ""
            )
            experiment_dir = os.path.join(experiment_dir, checkpoint_name)
        # Else assume the model to do inference is a generic experiment directory where the score will be saved
        else:
            if "merging" in filepath_forExperimentDir:
                experiment_dir = os.path.dirname(filepath_forExperimentDir)
            else:
                experiment_dir = os.path.dirname(
                    os.path.dirname(filepath_forExperimentDir)
                )

    if args.multiple_prompts:
        inference_fn = inference_withMultiplePrompts
    else:
        inference_fn = inference_withSinglePrompt

    if is_distributedSetup(args.world_size):
        mp.spawn(
            inference_fn,
            args=(
                args.world_size,
                args.port,
                model_config,
                evaluation_config,
                experiment_dir,
                None,
            ),
            nprocs=args.world_size,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_fn(
            device,
            args.world_size,
            args.port,
            model_config,
            evaluation_config,
            experiment_dir,
            title=None,
            cached_singleDatasetReaders=None,
        )

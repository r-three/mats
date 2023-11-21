import os
import torch
import json


from src.model.ModelConfig import ModelConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig

from src.merging.utils.checkpoint_filepaths import (
    get_modelCheckpointFilepath,
    get_datasetMixtureCheckpointFilepaths,
)
from src.model.load_model import load_model
from src.utils.utils import *


def checkEqual_dictKeys(first_dict, second_dict):
    assert set(first_dict.keys()) == set(second_dict.keys())


def checkDisjoint_dictKeys(first_dict, second_dict):
    assert len(set(first_dict.keys()).intersection(set(second_dict.keys()))) == 0


def loadCheckpoints_toMerge(
    pretrained_model,
    checkpoint_descriptor,
    instruction_format,
    datasetMixture_toMerge,
    device,
):
    checkpoint_fps = get_datasetMixtureCheckpointFilepaths(
        pretrained_model,
        checkpoint_descriptor,
        instruction_format,
        datasetMixture_toMerge,
    )
    checkpoints = {}
    for checkpoint_fp in checkpoint_fps:
        checkpoints[checkpoint_fp] = torch.load(checkpoint_fp, device)

    return checkpoint_fps, checkpoints


def getMerging_experimentDir(
    instruction_format,
    dataset_mixture_to_merge,
    pretrained_model,
    checkpoint_descriptor,
    mergeFunction_name,
):
    dataset_mixture_to_merge_str = (
        dataset_mixture_to_merge
        if isinstance(dataset_mixture_to_merge, str)
        else "_".join(dataset_mixture_to_merge)
    )

    experiment_dir = os.path.join(
        "exp_out",
        "merging",
        instruction_format,
        dataset_mixture_to_merge_str,
        format_modelName(pretrained_model),
        checkpoint_descriptor,
        mergeFunction_name,
    )

    return experiment_dir


def getMergedCheckpoint_fp(
    instruction_format,
    dataset_mixture_to_merge,
    pretrained_model,
    checkpoint_descriptor,
    mergeFunction_name,
):
    dataset_mixture_to_merge_str = (
        dataset_mixture_to_merge
        if isinstance(dataset_mixture_to_merge, str)
        else "_".join(dataset_mixture_to_merge)
    )

    mergedCheckpoint_dir = os.path.join(
        "merged_checkpoints",
        "merging",
        instruction_format,
        dataset_mixture_to_merge_str,
        format_modelName(pretrained_model),
        checkpoint_descriptor,
    )
    safe_makedirs(mergedCheckpoint_dir)
    mergedCheckpoint_fp = os.path.join(mergedCheckpoint_dir, mergeFunction_name + ".pt")

    return mergedCheckpoint_fp


def loadModel_fromCheckpointfp(model_config, modelCheckpoint_fp, device):
    model_updateDict = model_config.get_key_values()
    model_updateDict.update({"filepath_to_load_model": modelCheckpoint_fp})
    new_modelConfig = ModelConfig(update_dict=model_updateDict)
    model, trainableParameter_regex = load_model(new_modelConfig, device=device)
    return model, trainableParameter_regex


def addInferenceConfigArguments_toParser(parser):
    parser.add_argument(
        "-c", "--config_filepaths", action="store", type=str, nargs="*", required=True
    )
    parser.add_argument(
        "-m", "--model_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-ed", "--evaluation_dataset_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-er", "--evaluation_run_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    return parser


def addMergingConfigArguments_toParser(parser):
    parser.add_argument("-d", "--dataset_mixture_to_merge", type=str, required=True)
    parser.add_argument("-i", "--inference_dataset_mixture", type=str)
    parser.add_argument("--checkpoint_descriptor", type=str, default="full_model")
    parser.add_argument("--multiple_prompts", action="store_true")
    return parser


def addFisherArguments_toParser(parser):
    parser.add_argument("--use_true_fisher", action="store_true")
    parser.add_argument(
        "-s", "--split", type=str, choices=["train", "validation"], required=True
    )
    parser.add_argument(
        "-f",
        "--fisher_approximation",
        type=str,
        choices=["blockwise", "diagonal"],
    )
    return parser


def getInference_configs(args):
    parsed_filepaths = parse_configFilepaths(
        args.config_filepaths,
        types=[
            "models",
            "evaluation_dataset",
            "evaluation_run",
        ],
    )

    model_config = ModelConfig(
        config_filepaths=parsed_filepaths["models"], update_dict=args.model_kwargs
    )

    evaluationDataset_config = DatasetConfig(
        "evaluation",
        config_filepaths=parsed_filepaths["evaluation_dataset"],
        update_dict=args.evaluation_dataset_kwargs,
    )
    evaluation_config = EvaluationConfig(
        evaluationDataset_config,
        config_filepaths=parsed_filepaths["evaluation_run"],
        update_dict=args.evaluation_run_kwargs,
    )

    return model_config, evaluationDataset_config, evaluation_config


def get_newModelConfig(model_config, update_dict):
    new_updateDict = model_config.get_key_values()
    new_updateDict.update(update_dict)
    new_modelConfig = ModelConfig(None, update_dict=new_updateDict)

    return new_modelConfig


def get_newEvaluationConfig(
    evaluation_config,
    dataset_updateDict,
    evaluation_updateDict,
):
    evaluationDataset_config = evaluation_config.get_datasetConfig()

    newDataset_updateDict = evaluationDataset_config.get_key_values()
    if dataset_updateDict is not None:
        newDataset_updateDict.update(dataset_updateDict)
    new_evaluationDatasetConfig = DatasetConfig(None, update_dict=newDataset_updateDict)

    newEvaluation_updateDict = evaluation_config.get_key_values()
    newEvaluation_updateDict.update(evaluation_updateDict)
    newEvaluation_config = EvaluationConfig(
        evaluationDataset_config=new_evaluationDatasetConfig,
        update_dict=newEvaluation_updateDict,
    )

    return newEvaluation_config


def get_listModelLambda(model_lambda):
    list_modelLambda = []
    if model_lambda == "None" or model_lambda == -1.0:
        for i in range(0, 11):
            list_modelLambda.append(i / 10.0)
    else:
        list_modelLambda.append(model_lambda)

    return list_modelLambda

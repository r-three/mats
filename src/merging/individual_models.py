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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in getDatasets_inMixture(args.dataset_mixture_to_merge):
        (
            model_config,
            evaluationDataset_config,
            evaluation_config,
        ) = getInference_configs(args)
        modelCheckpoint_fp = get_modelCheckpointFilepath(
            model_config.pretrained_model,
            args.checkpoint_descriptor,
            evaluationDataset_config.instruction_format,
            dataset,
        )

        new_modelConfig = get_newModelConfig(
            model_config, {"filepath_to_load_model": modelCheckpoint_fp}
        )

        new_evaluationConfig = get_newEvaluationConfig(
            evaluation_config, {"dataset": dataset}, {}
        )

        experiment_name = "baseline"

        experiment_dir = getMerging_experimentDir(
            evaluationDataset_config.instruction_format,
            args.dataset_mixture_to_merge,
            model_config.pretrained_model,
            args.checkpoint_descriptor,
            experiment_name,
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

import argparse
import logging
import os
import torch
import copy

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.utils.model_utils import *
from src.inference import *
from src.model.utils import *

from src.merging.save_metadata.save_gram_matrix import *


def regmean(model_lambda, loaded_checkpoints, loaded_inputActivations):
    """

    Args:
        model_lambda:
        forward_or_backward:
        loaded_checkpoints: Dictionary mapping checkpoint filepaths to dictionary of parameters.
        loaded_inputActivations:
    Returns:

    """

    logging.info(f"Using Reg Mean to merge {len(loaded_checkpoints)} checkpoints")
    start = time.time()

    checkpoints_andActivationGramMatrices = {}
    for checkpoint_fp, parameter in loaded_checkpoints.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andActivationGramMatrices[dataset] = {"checkpoint": parameter}

    for (
        checkpoint_fp,
        gram_matrix,
    ) in loaded_inputActivations.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andActivationGramMatrices[dataset].update(
            {"input_activations_gram_matrix": gram_matrix}
        )

    gramMatrixTimesWeights_acrossDatasets = []
    gramMatrix_acrossDatasets = []

    nonMergedWeights_acrossDatasets = []

    for (
        dataset,
        checkpoint_andGramMatrix,
    ) in checkpoints_andActivationGramMatrices.items():
        checkpoint = checkpoint_andGramMatrix["checkpoint"]

        gram_matrices = {}
        scaledGram_timesWeight_matrices = {}
        for module_name, gram_matrix in checkpoint_andGramMatrix[
            "input_activations_gram_matrix"
        ].items():
            parameter_name = module_name + ".weight"

            scaled_gramMatrix = scale_nonDiagonalElements(gram_matrix, model_lambda)
            gram_matrices[parameter_name] = scaled_gramMatrix

            # Transpose the weight since Pytorch stores the transposed weight by default
            scaledGram_timesWeight = torch.matmul(
                scaled_gramMatrix, checkpoint[parameter_name].T
            )
            scaledGram_timesWeight_matrices[parameter_name] = scaledGram_timesWeight

        nonMerged_weights = {}
        for param_name, parameter in checkpoint.items():
            if param_name not in scaledGram_timesWeight_matrices:
                nonMerged_weights[param_name] = parameter

        gramMatrix_acrossDatasets.append(gram_matrices)
        gramMatrixTimesWeights_acrossDatasets.append(scaledGram_timesWeight_matrices)

        # checkDisjoint_dictKeys(inputActivation_gramMatrices, nonMerged_weights)
        nonMergedWeights_acrossDatasets.append(nonMerged_weights)
    # Compute the final weights according to RegMean
    final_model = matmul(
        matrix_inverse(scale_andSum(gramMatrix_acrossDatasets, 1)),
        scale_andSum(gramMatrixTimesWeights_acrossDatasets, 1),
    )
    # Transpose back the parameters
    for parameter_name, parameter in final_model.items():
        final_model[parameter_name] = parameter.T

    final_nonGramModel = scale_andSum(
        nonMergedWeights_acrossDatasets, 1 / len(nonMergedWeights_acrossDatasets)
    )
    # Add the non-merged weights
    for parameter_name, parameter in final_nonGramModel.items():
        final_model[parameter_name] = parameter
    end = time.time()
    diff_time = end - start

    return final_model, diff_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)
    parser.add_argument("--use_true_fisher", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--sum_seq_len", action="store_true")

    parser.add_argument(
        "-s", "--split", type=str, choices=["train", "validation"], required=True
    )
    parser.add_argument("--model_lambda", type=float)
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

    loaded_inputActivations = {}
    for checkpoint_fp in checkpoints_fp:
        input_activations = torch.load(
            getInputActivationGramMatrix_fp(
                checkpoint_fp, args.split, args.use_true_fisher, args.sum_seq_len
            ),
            torch.device("cpu"),
        )
        loaded_inputActivations[checkpoint_fp] = input_activations

    if args.time:
        total_diffTime = []
        for i in range(10):
            _, diff_time = regmean(0.9, loaded_checkpoints, loaded_inputActivations)
            total_diffTime.append(diff_time)
        total_diffTime = np.asarray(total_diffTime)
        print("Mean: ", np.mean(total_diffTime))
        print("Std: ", np.std(total_diffTime))
    else:
        list_modelLambda = get_listModelLambda(args.model_lambda)

        for model_lambda in list_modelLambda:
            merged_model, _ = regmean(
                model_lambda, loaded_checkpoints, loaded_inputActivations
            )

            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join(
                    "regmean", f"{args.split}", f"model_lambda_{model_lambda}"
                ),
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
                title="full_evaluation",
                cached_singleDatasetReaders=None,
            )

import argparse
import logging
import os
import torch
import copy
from tqdm import tqdm

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.merging.utils.model_utils import *
from src.inference import *
from torch import from_numpy, tensor, zeros_like
from src.merging.save_metadata.save_fisher import *

from src.model.utils import *

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg


def cg_forward(
    sum_diagonalFisherMatrices,
    sum_diagonalFisherTimesWeight,
    init_model,
    all_parameterNames,
    num_iterations,
):
    final_model = {}
    training_log = {}

    for parameter_name in tqdm(all_parameterNames):
        training_log[parameter_name] = {}
        weight_shape = sum_diagonalFisherTimesWeight[parameter_name].shape

        def matrixVector_product(v):
            v_weight_torch = from_numpy(v).reshape(weight_shape).float()

            matrixVector = torch.mul(
                sum_diagonalFisherMatrices[parameter_name],
                v_weight_torch,
            )
            return matrixVector.flatten().cpu().numpy()

        b = sum_diagonalFisherTimesWeight[parameter_name].detach().flatten()

        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()), matvec=matrixVector_product
        )

        if init_model is not None:
            init_x0 = init_model[parameter_name].detach().cpu().numpy().flatten()
            x_final, exit_code = cg(A, b, x0=init_x0, maxiter=num_iterations)
            initial_error = np.linalg.norm(matrixVector_product(init_x0) - b.numpy())

        else:
            x_final, exit_code = cg(A, b, maxiter=num_iterations)

        final_error = np.linalg.norm(matrixVector_product(x_final) - b.numpy())

        # Reshape weight
        final_weight = torch.tensor(x_final).reshape(weight_shape)
        final_model[parameter_name] = final_weight

        training_log[parameter_name].update(
            {
                "exit_code": exit_code,
                "final_error": final_error.astype(float),
            }
        )

        if init_model is not None:
            training_log[parameter_name].update(
                {
                    "initial_error": initial_error.astype(float),
                }
            )

    return final_model, training_log


def conjugateGradient_diagonalFisher(
    num_iterations,
    initialization,
    loaded_checkpoints,
    loaded_diagonalFishers,
    pretrained_checkpoint,
):
    checkpoints_andActivationGramMatrices = {}
    for checkpoint_fp, parameter in loaded_checkpoints.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andActivationGramMatrices[dataset] = {"checkpoint": parameter}

    all_parameterNames = None
    for (
        checkpoint_fp,
        diagonal_fisher,
    ) in loaded_diagonalFishers.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andActivationGramMatrices[dataset].update(
            {"diagonal_fisher": diagonal_fisher}
        )
        all_parameterNames = diagonal_fisher.keys()

    diagonalFisher_acrossDatasets = []
    weights_acrossDatasets = []
    diagonalFisherTimesWeight_acrossDatasets = []
    nonMergedWeights_acrossDatasets = []

    for (
        dataset,
        checkpoint_andGramMatrix,
    ) in checkpoints_andActivationGramMatrices.items():
        checkpoint = checkpoint_andGramMatrix["checkpoint"]

        diagonalFisher_matrices = {}
        weights = {}
        for module_name, diagonal_fisher in checkpoint_andGramMatrix[
            "diagonal_fisher"
        ].items():
            parameter_name = module_name
            weights[parameter_name] = checkpoint[parameter_name]
            diagonalFisher_matrices[parameter_name] = diagonal_fisher

        diagonalFisher_acrossDatasets.append(diagonalFisher_matrices)
        weights_acrossDatasets.append(weights)
        diagonalFisher_timesWeight = element_wise_multiply(
            diagonalFisher_matrices, weights
        )
        diagonalFisherTimesWeight_acrossDatasets.append(diagonalFisher_timesWeight)

        # Get non merged weights
        nonMerged_weights = {}
        for parameter_name, parameter in checkpoint.items():
            if parameter_name not in diagonalFisher_matrices:
                nonMerged_weights[parameter_name] = parameter
        nonMergedWeights_acrossDatasets.append(nonMerged_weights)

    sum_diagonalFisherMatrices = scale_andSum(diagonalFisher_acrossDatasets, 1)
    average_weights = scale_andSum(
        weights_acrossDatasets, 1 / len(weights_acrossDatasets)
    )
    sum_diagonalFisherTimesWeight = scale_andSum(
        diagonalFisherTimesWeight_acrossDatasets, 1
    )

    if initialization == "average":
        init_model = average_weights
    elif initialization == "pretrained":
        init_model = pretrained_checkpoint
    else:
        if initialization is not None:
            init_model = {}
            # Transpose weights
            for parameter_name, parameter in torch.load(initialization).items():
                init_model[parameter_name] = parameter

        else:
            init_model = None

    final_model, training_log = cg_forward(
        sum_diagonalFisherMatrices,
        sum_diagonalFisherTimesWeight,
        init_model,
        all_parameterNames,
        num_iterations,
    )

    final_nonMergedModel = scale_andSum(
        nonMergedWeights_acrossDatasets, 1 / len(nonMergedWeights_acrossDatasets)
    )
    # Add the non-merged weights
    for parameter_name, parameter in final_nonMergedModel.items():
        final_model[parameter_name] = parameter

    return final_model, training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addFisherArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)

    parser.add_argument("--initialization")
    parser.add_argument("--model_lambda", type=float)
    parser.add_argument("--num_iterations", type=int, required=True)
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

    merged_model, training_log = conjugateGradient_diagonalFisher(
        args.num_iterations,
        args.initialization,
        loaded_checkpoints,
        loaded_diagonalFishers,
        pretrained_checkpoint,
    )

    experiment_name = os.path.join(
        "conjugate_gradients",
        get_fisherArgumentString(
            "diagonal", args.split, False
        ),
    )

    if args.initialization is not None:
        # If we use path, then just use the name between the second to last and last /
        if "exp_out" in args.initialization:
            if "conjugate_gradient" in args.initialization:
                initialization_string = args.initialization.split("/")[-2]
            else:
                initialization_string = "_".join(args.initialization.split("/")[6:-2])
        else:
            initialization_string = args.initialization
        experiment_name += f"_initialize_{initialization_string}"

    experiment_name += f"_iterations_{args.num_iterations}"

    experiment_dir = getMerging_experimentDir(
        evaluationDataset_config.instruction_format,
        args.dataset_mixture_to_merge,
        model_config.pretrained_model,
        args.checkpoint_descriptor,
        experiment_name,
    )
    safe_makedirs(experiment_dir)

    with open(
        os.path.join(experiment_dir, "log.json"),
        "w+",
    ) as f:
        f.write(json.dumps(training_log, indent=4) + "\n")

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

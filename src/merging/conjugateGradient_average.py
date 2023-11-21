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
    average_weights,
    init_model,
    all_parameterNames,
    num_iterations,
):
    final_model = {}
    training_log = {}

    for parameter_name in tqdm(all_parameterNames):
        training_log[parameter_name] = {}
        weight_shape = average_weights[parameter_name].shape

        def matrixVector_product(v):
            return v

        b = average_weights[parameter_name].detach().flatten()

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


def conjugateGradient_average(
    num_iterations,
    initialization,
    loaded_checkpoints,
    pretrained_checkpoint,
):
    checkpoints_andActivationGramMatrices = {}
    for checkpoint_fp, parameter in loaded_checkpoints.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andActivationGramMatrices[dataset] = {"checkpoint": parameter}

    weights_acrossDatasets = []

    all_parameterNames = None
    for (
        dataset,
        checkpoint_andGramMatrix,
    ) in checkpoints_andActivationGramMatrices.items():
        weights = checkpoint_andGramMatrix["checkpoint"]
        weights_acrossDatasets.append(weights)
        all_parameterNames = weights.keys()

    average_weights = scale_andSum(
        weights_acrossDatasets, 1 / len(weights_acrossDatasets)
    )

    if initialization == "average":
        init_model = average_weights
    elif initialization == "pretrained":
        init_model = pretrained_checkpoint
    else:
        if initialization is not None:
            init_model = {}
            for parameter_name, parameter in torch.load(initialization).items():
                init_model[parameter_name] = parameter

        else:
            init_model = None

    final_model, training_log = cg_forward(
        average_weights,
        init_model,
        all_parameterNames,
        num_iterations,
    )

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

    merged_model, training_log = conjugateGradient_average(
        args.num_iterations,
        args.initialization,
        loaded_checkpoints,
        pretrained_checkpoint,
    )

    experiment_name = os.path.join("conjugate_gradients", "average")

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

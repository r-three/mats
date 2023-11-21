import argparse
import logging
import os
import torch
import copy
from tqdm import tqdm

from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.checkpoint_filepaths import *
from src.inference import *
from torch import from_numpy, tensor, zeros_like

from src.merging.save_metadata.save_gram_matrix import *

from src.model.utils import *

from src.merging.save_metadata.save_fisher import *

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg


# Matrix Multiplication Forward can be used for RegMean or Blockwise Fisher merging with PEFT methods
def cg_forward(
    sum_inputGramMatrices,
    sum_inputGramTimesWeight,
    init_model,
    num_iterations,
    all_moduleNames,
    is_linearLayer,
):
    final_model = {}
    training_log = {}

    for module_name in tqdm(all_moduleNames):
        if "ia3" in module_name:
            parameter_name = module_name
            current_gramMatrix = sum_inputGramMatrices[parameter_name]
        else:
            parameter_name = module_name + ".weight"
            current_gramMatrix = sum_inputGramMatrices[parameter_name].cuda()

        b = sum_inputGramTimesWeight[parameter_name].flatten().float()
        weight_shape = sum_inputGramTimesWeight[parameter_name].shape

        def matrixVector_product(vector):
            if is_linearLayer:
                # [input_dim, output_dim]
                reshaped_vector = from_numpy(vector).reshape(weight_shape).float()
            else:
                # [output_dim]
                reshaped_vector = from_numpy(vector).float()

            if "ia3" not in module_name:
                reshaped_vector = reshaped_vector.cuda()

            matrixVector = torch.matmul(current_gramMatrix, reshaped_vector)

            if "ia3" in module_name:
                return matrixVector.flatten().numpy()
            else:
                return matrixVector.flatten().cpu().numpy()

        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()),
            matvec=matrixVector_product,
        )

        if init_model is not None:
            if is_linearLayer:
                x0 = init_model[parameter_name].detach().flatten().cpu().numpy()
            else:
                x0 = init_model[parameter_name].detach().cpu().numpy()
            x_final, exit_code = cg(A, b, x0=x0, maxiter=num_iterations)
            initial_error = np.linalg.norm(matrixVector_product(x0) - b.numpy())
        else:
            x_final, exit_code = cg(A, b, maxiter=num_iterations)

        final_error = np.linalg.norm(matrixVector_product(x_final) - b.numpy())

        final_weight = torch.tensor(x_final).reshape(weight_shape)

        if is_linearLayer:
            # Transpose the weight back for linear layers
            final_model[parameter_name] = final_weight.T
        else:
            #
            final_model[parameter_name] = final_weight

        training_log[parameter_name] = {
            "exit_code": exit_code,
            "final_error": final_error.astype(float),
        }

        if init_model is not None:
            training_log[parameter_name].update(
                {
                    "initial_error": initial_error.astype(float),
                }
            )

    return final_model, training_log


def cg_forwardBackward(
    inputGramMatrices_acrossDatasets,
    outputGramMatrices_acrossDatasets,
    sum_inputGramTimesWeightTimesOutputGram,
    init_model,
    num_iterations,
    all_moduleNames,
):
    final_model = {}
    training_log = {}

    for module_name in tqdm(all_moduleNames):
        parameter_name = module_name + ".weight"
        weight_shape = sum_inputGramTimesWeightTimesOutputGram[parameter_name].shape

        training_log[parameter_name] = {}

        input_gramMatrices = []
        output_gramMatrices = []
        for input_gramMatrix, output_gramMatrix in zip(
            inputGramMatrices_acrossDatasets, outputGramMatrices_acrossDatasets
        ):
            input_gramMatrices.append(input_gramMatrix[parameter_name].cuda())
            output_gramMatrices.append(output_gramMatrix[parameter_name].cuda())

        def matrixVector_product(vector):
            reshaped_vector = from_numpy(vector).reshape(weight_shape).cuda().float()

            matrixVector = None
            for input_gramMatrix, output_gramMatrix in zip(
                input_gramMatrices, output_gramMatrices
            ):
                if matrixVector is None:
                    matrixVector = torch.matmul(
                        torch.matmul(input_gramMatrix, reshaped_vector),
                        output_gramMatrix,
                    )
                else:
                    matrixVector += torch.matmul(
                        torch.matmul(input_gramMatrix, reshaped_vector),
                        output_gramMatrix,
                    )

            return matrixVector.flatten().cpu().numpy()

        b = sum_inputGramTimesWeightTimesOutputGram[parameter_name].flatten()

        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()),
            matvec=matrixVector_product,
        )

        if init_model is not None:
            x0 = init_model[parameter_name].detach().flatten().cpu().numpy()
            final_x, exit_code = cg(A, b, x0=x0, maxiter=num_iterations)
            initial_error = np.linalg.norm(matrixVector_product(x0) - b.numpy())

        else:
            final_x, exit_code = cg(A, b, maxiter=num_iterations)

        final_error = np.linalg.norm(matrixVector_product(final_x) - b.numpy())
        final_weight = torch.tensor(final_x).reshape(weight_shape)

        # Transpose the weight back
        final_model[parameter_name] = final_weight.T
        training_log[parameter_name] = {
            "exit_code": exit_code,
            "final_error": final_error.astype(float),
        }

        if init_model is not None:
            training_log[parameter_name].update(
                {
                    "initial_error": initial_error.astype(float),
                }
            )

    return final_model, training_log


def conjugateGradient_fisherMerging(
    model_lambda,
    num_iterations,
    initialization,
    loaded_checkpoints,
    loaded_forwardGramMatrices,
    loaded_backwardGramMatrices,
    is_linearLayer,
    use_backward,
    pretrained_checkpoint,
):
    """
    linear_layer: whether linear layer or IA3
    use_backward: whether to use backward in addition to forward objective (only applicable for linear blockwise)
    """
    if use_backward:
        assert is_linearLayer
    start = time.time()

    checkpoints_andGramMatrices = {}
    for checkpoint_fp, parameter in loaded_checkpoints.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andGramMatrices[dataset] = {"checkpoint": parameter}

    all_moduleNames = None
    for (
        checkpoint_fp,
        inputActivations_gramMatrix,
    ) in loaded_forwardGramMatrices.items():
        dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
        checkpoints_andGramMatrices[dataset].update(
            {"forward": inputActivations_gramMatrix}
        )
        all_moduleNames = set(inputActivations_gramMatrix.keys())
    assert all_moduleNames is not None
    if "transformer.lm_head" in all_moduleNames:
        # Ignore lm_head since backward wont have it
        all_moduleNames.remove("transformer.lm_head")

    if use_backward:
        for (
            checkpoint_fp,
            outputActivationsGradients_gramMatrix,
        ) in loaded_backwardGramMatrices.items():
            dataset = getDataset_fromCheckpointFilepath(checkpoint_fp)
            checkpoints_andGramMatrices[dataset].update(
                {"backward": outputActivationsGradients_gramMatrix}
            )

    forwardGramMatrices_acrossDatasets = []
    weights_acrossDatasets = []
    gramTimesWeight_acrossDatasets = []
    nonMergedWeights_acrossDatasets = []

    if use_backward:
        backwardGramMatrices_acrossDatasets = []

    for (
        dataset,
        checkpoint_andGramMatrix,
    ) in checkpoints_andGramMatrices.items():
        checkpoint = checkpoint_andGramMatrix["checkpoint"]

        # Forward pass
        forward_gramMatrices = {}
        weights = {}
        for module_name, inputActivations_gramMatrix in checkpoint_andGramMatrix[
            "forward"
        ].items():
            # Skip this parameter since backward pass wont have it
            if use_backward and "lm_head" in module_name:
                continue
            if "ia3" in module_name:
                parameter_name = module_name
            else:
                parameter_name = module_name + ".weight"
            scaled_gramMatrix = scale_nonDiagonalElements(
                inputActivations_gramMatrix, model_lambda
            )
            # Transpose the weights for linear layers for efficient matrix multiplication
            if is_linearLayer:
                weights[parameter_name] = checkpoint[parameter_name].T
            # Else ignore transposing the weight
            else:
                weights[parameter_name] = checkpoint[parameter_name]

            forward_gramMatrices[parameter_name] = scaled_gramMatrix

        forwardGramMatrices_acrossDatasets.append(forward_gramMatrices)
        weights_acrossDatasets.append(weights)

        if use_backward:
            # Backward pass
            backward_gramMatrices = {}
            for (
                module_name,
                outputActivationGradient_gramMatrix,
            ) in checkpoint_andGramMatrix["backward"].items():
                scaled_gramMatrix = scale_nonDiagonalElements(
                    outputActivationGradient_gramMatrix, model_lambda
                )
                backward_gramMatrices[module_name + ".weight"] = scaled_gramMatrix

            checkEqual_dictKeys(
                backward_gramMatrices,
                forward_gramMatrices,
            )
            backwardGramMatrices_acrossDatasets.append(backward_gramMatrices)

        with torch.no_grad():
            forward_timesWeight = matmul(forward_gramMatrices, weights)
            if use_backward:
                gramTimesWeight_acrossDatasets.append(
                    matmul(
                        forward_timesWeight,
                        backward_gramMatrices,
                    )
                )

            else:
                gramTimesWeight_acrossDatasets.append(forward_timesWeight)

        # Get non merged weights
        nonMerged_weights = {}
        for parameter_name, parameter in checkpoint.items():
            if parameter_name not in forward_gramMatrices:
                nonMerged_weights[parameter_name] = parameter
        nonMergedWeights_acrossDatasets.append(nonMerged_weights)

    sum_forwardGramMatrices = scale_andSum(forwardGramMatrices_acrossDatasets, 1)
    average_weights = scale_andSum(
        weights_acrossDatasets, 1 / len(weights_acrossDatasets)
    )
    sum_gramTimesWeight = scale_andSum(gramTimesWeight_acrossDatasets, 1)

    if initialization == "average":
        init_model = average_weights
    elif initialization == "pretrained":
        init_model = pretrained_checkpoint
    else:
        if initialization is not None:
            init_model = {}
            # Transpose weights
            for parameter_name, parameter in torch.load(initialization).items():
                init_model[parameter_name] = parameter.T

        else:
            init_model = None

    if use_backward:
        final_model, training_log = cg_forwardBackward(
            forwardGramMatrices_acrossDatasets,
            backwardGramMatrices_acrossDatasets,
            sum_gramTimesWeight,
            init_model,
            num_iterations,
            all_moduleNames,
        )
    else:
        final_model, training_log = cg_forward(
            sum_forwardGramMatrices,
            sum_gramTimesWeight,
            init_model,
            num_iterations,
            all_moduleNames,
            is_linearLayer,
        )

    final_nonMergedModel = scale_andSum(
        nonMergedWeights_acrossDatasets, 1 / len(nonMergedWeights_acrossDatasets)
    )
    # Add the non-merged weights
    for parameter_name, parameter in final_nonMergedModel.items():
        final_model[parameter_name] = parameter
    end = time.time()
    diff_time = end - start

    return final_model, training_log, diff_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addInferenceConfigArguments_toParser(parser)
    parser = addFisherArguments_toParser(parser)
    parser = addMergingConfigArguments_toParser(parser)

    parser.add_argument("--initialization")
    parser.add_argument("--model_lambda", type=float)
    parser.add_argument("--num_iterations", type=int, required=True)
    parser.add_argument("--use_backward", action="store_true")
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

    loaded_forwardGramMatrices = {}
    loaded_backwardGramMatrices = {}

    for checkpoint_fp in checkpoints_fp:
        if "ia3" in args.checkpoint_descriptor:
            forward_gramMatrix = torch.load(
                get_fisherFP(
                    checkpoint_fp,
                    args.split,
                    args.use_true_fisher,
                    args.fisher_approximation,
                ),
                torch.device("cpu"),
            )
        else:
            forward_gramMatrix = torch.load(
                getInputActivationGramMatrix_fp(
                    checkpoint_fp, args.split, args.use_true_fisher
                ),
                torch.device("cpu"),
            )

        loaded_forwardGramMatrices[checkpoint_fp] = forward_gramMatrix
        if args.use_backward:
            backward_gramMatrix = torch.load(
                getOutputActivationGradientGramMatrix_fp(
                    checkpoint_fp, args.split, args.use_true_fisher
                ),
                torch.device("cpu"),
            )
            loaded_backwardGramMatrices[checkpoint_fp] = backward_gramMatrix

    if args.time:
        total_diffTime = []
        for i in range(10):
            merged_model, training_log, diff_time = conjugateGradient_fisherMerging(
                1.0,
                args.num_iterations,
                args.initialization,
                loaded_checkpoints,
                loaded_forwardGramMatrices,
                loaded_backwardGramMatrices,
                "full_model" in args.checkpoint_descriptor,
                args.use_backward,
                pretrained_checkpoint,
            )
            total_diffTime.append(diff_time)
        total_diffTime = np.asarray(total_diffTime)
        print("Mean: ", np.mean(total_diffTime))
        print("Std: ", np.std(total_diffTime))
    else:
        list_modelLambda = get_listModelLambda(args.model_lambda)
        for model_lambda in list_modelLambda:
            merged_model, training_log, _ = conjugateGradient_fisherMerging(
                model_lambda,
                args.num_iterations,
                args.initialization,
                loaded_checkpoints,
                loaded_forwardGramMatrices,
                loaded_backwardGramMatrices,
                "full_model" in args.checkpoint_descriptor,
                args.use_backward,
                pretrained_checkpoint,
            )

            if args.use_backward or "ia3" in args.checkpoint_descriptor:
                fisherArgument_string = get_fisherArgumentString(
                    "blockwise",
                    args.split,
                    args.use_true_fisher,
                )

                experiment_name = f"fisher_merging_{fisherArgument_string}"
            else:
                if args.use_true_fisher:
                    experiment_name = f"true_regmean_{args.split}"
                else:
                    experiment_name = f"regmean_{args.split}"


            if args.initialization is not None:
                # If we use path, then just use the name between the second to last and last /
                if "exp_out" in args.initialization:
                    if "conjugate_gradient" in args.initialization:
                        initialization_string = args.initialization.split("/")[-2]
                    else:
                        initialization_string = "_".join(
                            args.initialization.split("/")[6:-2]
                        )
                else:
                    initialization_string = args.initialization
                experiment_name += f"_initialize_{initialization_string}"

            experiment_name += (
                f"_model_lambda_{model_lambda}_iterations_{args.num_iterations}"
            )

            experiment_dir = getMerging_experimentDir(
                evaluationDataset_config.instruction_format,
                args.dataset_mixture_to_merge,
                model_config.pretrained_model,
                args.checkpoint_descriptor,
                os.path.join("conjugate_gradients", experiment_name),
            )
            safe_makedirs(experiment_dir)
            mergedCheckpoint_fp = os.path.join(experiment_dir, "merged_model.pt")
            torch.save(merged_model, mergedCheckpoint_fp)
            new_modelConfig = get_newModelConfig(
                model_config, {"filepath_to_load_model": mergedCheckpoint_fp}
            )
            with open(
                os.path.join(experiment_dir, "log.json"),
                "w+",
            ) as f:
                f.write(json.dumps(training_log, indent=4) + "\n")

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

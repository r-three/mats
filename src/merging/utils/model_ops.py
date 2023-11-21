import math
import torch
import copy
import time


def check_parameterNamesMatch(checkpoints):
    """
    Check that the parameter names are the same for all checkpoints

    Args:
        checkpoints:

    Returns:

    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) < 2:
        return True

    for checkpoint in checkpoints[1:]:
        current_parameterNames = set(checkpoint.keys())
        if current_parameterNames != parameter_names:
            raise ValueError(
                "Differing parameter names in models. "
                f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
            )


def reduce_modelParameters(model_parameters, reduceValue_fn, reduceCheckpoint_fn):
    """
    Reduce checkpoint into a single value

    Args:
        model_parameters:
        reduceValue_fn: Function to reduce parameter block into a single value.
        reduceCheckpoint_fn: Function to reduce values from each parameter block into a single value.

    Returns:

    """
    newModel_parameters = {}
    for parameter_name, parameter_values in model_parameters.items():
        newModel_parameters[parameter_name] = reduceValue_fn(parameter_values)

    return reduceCheckpoint_fn(list(newModel_parameters.values()))


def reduceSum_modelParameters(model_parameters):
    sum_modelParameters = reduce_modelParameters(
        model_parameters,
        lambda x: torch.sum(x),
        lambda x: torch.sum(torch.stack(x, dim=0)),
    )
    return sum_modelParameters


def reduceAll_modelParameters(allModels_parameters, reduce_fn):
    """
    Reduce a list of checkpoints into a single checkpoint

    Args:
        allModels_parameters: List of dictionaries
        reduce_fn: Takes a tensor where the first dimension iterates over checkpoints
    Returns:
        Model: dictionary
    """
    # Returns list of list of parameters where the outer list is the parameter names,
    # and inner list is the models.
    all_parameterValues = zip(*list(map(lambda x: x.values(), allModels_parameters)))

    # All models must have the same parameters
    all_parameterNames = allModels_parameters[0].keys()

    newModel_parameters = {}
    for parameter_name, parameter_values in zip(
        *[all_parameterNames, all_parameterValues]
    ):
        newModel_parameters[parameter_name] = reduce_fn(
            torch.stack(list(parameter_values), dim=0)
        )
    return newModel_parameters


def pairwiseMap_modelParameters(modelOne_parameters, modelTwo_parameters, map_fn):
    """

    Args:
        modelOne_parameters:
        modelTwo_parameters:
        map_fn:

    Returns:

    """
    # All models must have the same parameters
    all_parameterNames = modelOne_parameters.keys()

    newModel_parameters = {}
    for parameter_name in all_parameterNames:
        newModel_parameters[parameter_name] = map_fn(
            modelOne_parameters[parameter_name], modelTwo_parameters[parameter_name]
        )

    return newModel_parameters


def map_modelParameters(model_parameters, map_fn):
    """

    Args:
        model_parameters:
        map_fn:

    Returns:

    """
    newModel_parameters = {}
    for parameter_name, parameter_value in model_parameters.items():
        newModel_parameters[parameter_name] = map_fn(parameter_value)
    return newModel_parameters


def add(modelOne_parameters, modelTwo_parameters):
    """
    Add the parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    add_fn = lambda x, y: x + y
    added_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, add_fn
    )
    return added_model


def element_wise_multiply(modelOne_parameters, modelTwo_parameters):
    """
    Element wise multiply the parameters.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    elementWiseMul = lambda x, y: torch.mul(x, y)
    elementWiseMul_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, elementWiseMul
    )
    return elementWiseMul_model


def matmul(modelOne_parameters, modelTwo_parameters):
    """
    Matrix multiply the parameters.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    matmul_fn = lambda x, y: torch.matmul(x, y)
    matmul_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, matmul_fn
    )
    return matmul_model


def subtract(modelOne_parameters, modelTwo_parameters):
    """
    Subtract the parameters of modelTwo from the parameters of modelOne.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    subtract_fn = lambda x, y: x - y
    subtracted_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, subtract_fn
    )
    return subtracted_model


def divide(modelOne_parameters, modelTwo_parameters):
    """
    Subtract the parameters of modelTwo from the parameters of modelOne.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """

    divide_fn = lambda x, y: x / y
    divide_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, divide_fn
    )
    return divide_model


def scale(model_parameters, scaler):
    """
    Multiply model parameters by scaler.

    Args:
        model_parameters:
        scaler:

    Returns:

    """
    scale_fn = lambda x: x * scaler
    scaled_model = map_modelParameters(model_parameters, scale_fn)
    return scaled_model


def perturbModel_inRandomDirection(model_parameters, model_norm):
    """
    Create model parameters to have the same shape as model parameters
    but are randomly initialized with specified norm.

    Args:
        model_parameters:
        model_norm:

    Returns:

    """
    random_fn = lambda x: torch.randn(x.size()).to(
        list(model_parameters.values())[0].device
    )
    diff_parameters = map_modelParameters(model_parameters, random_fn)
    scaler = model_norm / l2_norm(diff_parameters)
    diff_parameters = scale(diff_parameters, scaler)
    return add(model_parameters, diff_parameters)


def scale_andSum(allModels_parameters, model_lambda):
    """
    Scale up a list of model parameters, and then sum them.

    Args:
        allModels_parameters: List of dictionaries that represent model parameters.
        model_lambda:

    Returns:

    """
    sum_fn = lambda parameters: torch.sum(parameters * model_lambda, dim=0)
    summed_model = reduceAll_modelParameters(allModels_parameters, sum_fn)
    return summed_model


def l2_norm(model_parameters):
    """
    Find the L2 norm of the model parameters.

    Args:
        model_parameters:

    Returns:

    """
    square_fn = lambda x: x * x

    squared_model = map_modelParameters(model_parameters, square_fn)
    sumOfSquared_model = reduce_modelParameters(
        squared_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    return math.sqrt(sumOfSquared_model)


def l1_norm(model_parameters):
    """
    Find the L1 norm of the model parameters.

    Args:
        model_parameters:

    Returns:

    """
    absoluteValue_fn = lambda x: torch.abs(x)

    absoluteValue_model = map_modelParameters(model_parameters, absoluteValue_fn)
    sumOfAbsoluteValue_model = reduce_modelParameters(
        absoluteValue_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    return sumOfAbsoluteValue_model.item()


def set_minimum(model_parameters, epsilon):
    """
    Set the minimum of the parameters to be epsilon. For any value less than epsilon,
    replace with epsilon

    Args:
        model_parameters:

    Returns:

    """
    new_modelParameters = {}
    for parameter_name, parameter in model_parameters.items():
        new_parameter = parameter.clone()
        new_parameter[new_parameter < epsilon] = epsilon
        new_modelParameters[parameter_name] = new_parameter
    return new_modelParameters


def square(model_parameters):
    square_fn = lambda x: x * x

    squared_model = map_modelParameters(model_parameters, square_fn)
    return squared_model


def inverse(model_parameters):
    inverse_model = map_modelParameters(model_parameters, lambda x: 1 / x)
    return inverse_model


def identity(model_parameters):
    identity_model = map_modelParameters(
        model_parameters, lambda x: torch.ones_like(x).to(x.device)
    )
    return identity_model


def log(model_parameters):
    log_model = map_modelParameters(model_parameters, lambda x: torch.log(x))
    return log_model


def matrix_inverse(model_parameters):
    """
    Find the inverse of the model parameters.

    Args:
        model_parameters:

    Returns:

    """
    matrixInverse_fn = lambda x: torch.linalg.inv(x)
    matrixInverse_model = {}
    for parameter_name, parameter in model_parameters.items():
        try:
            matrix_inverse = matrixInverse_fn(parameter)
            matrixInverse_model[parameter_name] = matrix_inverse
        except:
            print("except")
            # If matrix is not invertible because row/col is 0, then we remove the row/col with 0 and invert the submatrix
            # We then insert the row/col with 0 back afterwards
            nonZero_rowIdx = (torch.sum(parameter, dim=1) != 0).nonzero().squeeze()
            nonZero_colIdx = (torch.sum(parameter, dim=0) != 0).nonzero().squeeze()
            assert (nonZero_colIdx == nonZero_rowIdx).all()
            num_row = parameter.shape[0]
            nonZero_broadcastColIdx = nonZero_colIdx[None, :].repeat((num_row, 1))
            nonZero_broadcastRowIdx = nonZero_rowIdx[:, None].repeat(
                (1, nonZero_broadcastColIdx.shape[1])
            )

            # Get submatrix that is full rank
            fullRank_parameter = torch.gather(parameter, 1, nonZero_broadcastColIdx)
            fullRank_parameter = torch.gather(
                fullRank_parameter, 0, nonZero_broadcastRowIdx
            )

            # Invert submatrix that is full rank
            inverse_fullRankParameter = matrixInverse_fn(fullRank_parameter)
            inverse_parameter = copy.deepcopy(parameter)
            inverse_parameter[
                nonZero_rowIdx[:, None], nonZero_colIdx
            ] = inverse_fullRankParameter
            matrixInverse_model[parameter_name] = inverse_parameter
    # inverse_model = map_modelParameters(model_parameters, inverse_fn)
    return matrixInverse_model


def dot_product(modelOne_parameters, modelTwo_parameters):
    """
    Find the dot product between the parameters of two modelsd.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    multiply_fn = lambda x, y: x * y

    multiplied_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, multiply_fn
    )
    sumOfMultiplied_model = reduce_modelParameters(
        multiplied_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    return sumOfMultiplied_model


def cosine_sim(modelOne_parameters, modelTwo_parameters):
    """
    Find the cosine similarity between parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """
    multiply_fn = lambda x, y: x * y

    multiplied_model = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, multiply_fn
    )
    sumOfMultiplied_model = reduce_modelParameters(
        multiplied_model, lambda x: torch.sum(x), lambda x: sum(x)
    )
    cosine_sim = sumOfMultiplied_model.item() / (
        l2_norm(modelOne_parameters) * l2_norm(modelTwo_parameters)
    )

    return cosine_sim


def project(modelOne_parameters, modelTwo_parameters):
    """
    Project modelOne (a) onto modelTwo (b)

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:

    """

    b_dot_b = dot_product(modelTwo_parameters, modelTwo_parameters)
    a_dot_b = dot_product(modelOne_parameters, modelTwo_parameters)

    scaler = a_dot_b / b_dot_b
    multiply_fn = lambda x: x * scaler

    return map_modelParameters(modelTwo_parameters, multiply_fn)

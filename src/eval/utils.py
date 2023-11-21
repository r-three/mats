import copy
import torch
import os
import json

from src.eval.Evaluator import Evaluator
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.DatasetConfig import DatasetConfig


from src.utils.utils import get_average, get_median, get_interquartileRange
from src.utils.NoIndentEncoder import NoIndentEncoder, noIndent_dictOrList_onFirstLevel

from src.data.dataset_readers import getDatasets_inMixture


def prepare_batchOfEvalInfo(batch):
    batchOf_evalInfo = copy.deepcopy(batch)

    for key, value in batch.items():
        # Remove ids and mask since no longer needed
        if ("ids" in key) or ("mask" in key):
            del batchOf_evalInfo[key]
        else:
            # Convert tensors to list
            if torch.is_tensor(batchOf_evalInfo[key]):
                batchOf_evalInfo[key] = value.cpu().numpy().tolist()

    return batchOf_evalInfo


def updateDatasetArgs_inEvaluationConfig(evaluation_config, dataset_args):
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    newEvaluationDataset_keyValues = evaluationDataset_config.get_key_values()
    newEvaluationDataset_keyValues.update(dataset_args)
    newEvaluationDataset_config = DatasetConfig(
        key_prefix=None,
        update_dict=newEvaluationDataset_keyValues,
    )
    newEvaluation_config = EvaluationConfig(
        evaluationDataset_config=newEvaluationDataset_config,
        update_dict=evaluation_config.get_key_values(),
    )
    return newEvaluation_config


def concatenate_scores(list_results, getKey_fn):
    """Concatenate the score and use a key to specify the score

    Args:
        list_results (_type_): _description_
        key_toUse (_type_): _description_

    Returns:
        _type_: _description_
    """
    concatenated_scores = {}
    for result in list_results:
        concatenated_scores[getKey_fn(result)] = result["score"]
    return concatenated_scores


def get_runsDirs(list_ofResults):
    list_runDirs = []
    for result in list_ofResults:
        list_runDirs.append(result["run_dir"])
    return list_runDirs


def average_scores(list_results):
    """

    Args:
        multiple_configAndScores:

    Returns:

    """
    individual_averageScores = list(map(lambda x: x["score"]["average"], list_results))
    average_score = get_average(individual_averageScores)

    return {"average": average_score}


def group_by(list_ofItems, fn_toGetGroupByField):
    groups = {}

    for my_dict in list_ofItems:
        field = fn_toGetGroupByField(my_dict)

        if field in groups:
            groups[field].append(my_dict)
        else:
            groups[field] = [my_dict]

    return groups


def map_forDictionaries(my_dict, map_fn):
    mapped_dict = {}
    for k, v in my_dict.items():
        mapped_dict[k] = map_fn(v)
    return mapped_dict


def get_summaryOfScores_acrossTemplates(
    list_results,
):
    """

    Args:
        list_results:

    Returns:

    """
    individual_averageScores = list(map(lambda x: x["score"]["average"], list_results))

    summary_ofScores = {
        "median": get_median(individual_averageScores),
        "interquartile_range": get_interquartileRange(individual_averageScores),
        "average_scores_for_each_prompt": individual_averageScores,
    }

    return summary_ofScores


def saveResult_acrossDatasets(
    datasets, scores, split, getScore_fn, score_fp, saveAverage_acrossDatasets, title
):
    """
    Save the average of the average score for each dataset

    Args:
        datasets:
        scores:
        getScore_fn:
        score_fp:
        saveAverage_acrossDatasets:

    Returns:

    """
    labels_toDisplay = []
    scores_toDisplay = []

    if saveAverage_acrossDatasets:
        if split is not None:
            labels_toDisplay.append(f"{split} Avg.")
            scores_toDisplay.append(str(round(scores[split]["average"] * 100, 1)))
        else:
            labels_toDisplay.append("Avg.")
            scores_toDisplay.append(str(round(scores["average"] * 100, 1)))

    if len(datasets) == 1:
        scores_toDisplay.append(getScore_fn(scores))
    else:
        for dataset in datasets:
            scores_toDisplay.append(getScore_fn(scores[dataset]))

    label_str = ",".join(labels_toDisplay)
    scores_str = ",".join(scores_toDisplay)

    with open(score_fp, "a+") as f:
        if title is not None:
            f.write(title + "\n")
        f.write(label_str + "\n")
        f.write(scores_str + "\n")

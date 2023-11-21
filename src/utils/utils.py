import random
import numpy as np
import os
import argparse
import subprocess
import statistics
import torch
import copy
import json
import re
import sys

from statistics import mean
from scipy.stats import iqr

from src.utils.NoIndentEncoder import NoIndentEncoder, noIndent_dictOrList_onFirstLevel


def set_seeds(seed):
    """
    Set all random seeds to the fixed seed

    Args:
        seed:

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# From https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
def flatten_list(l):
    return [item for sublist in l for item in sublist]


def convert_listOfDict_toDictOfList(list_ofDict):
    """
    Args:
        list_ofDict:

    Returns:
        dict_ofList
    """
    dict_ofList = {}

    for single_dict in list_ofDict:
        for k, v in single_dict.items():
            if k in dict_ofList:
                dict_ofList[k].append(v)
            else:
                dict_ofList[k] = [v]

    return dict_ofList


# From https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def convert_dictOfLists_to_listOfDicts(dictOfLists):
    listOfDicts = []
    for datapoint_values in zip(*dictOfLists.values()):
        listOfDicts.append(dict(zip(dictOfLists, datapoint_values)))
    return listOfDicts


def safe_makedirs(dir_name):
    """
    Makes a directory if it doesn't exists yet

    Args:
        dir_name: directory name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def getValueOfKey_inDictionary(dictionary_toSearch, keys_toSearchFor):
    """
    Check if key or path of key exists in dictionary and return the value correspoding to the key

    Args:
        dictionary_toSearch:
        keys_toSearchFor: returns the value of the first key that is found in dictionary

    Returns:

    """

    for full_key in keys_toSearchFor:
        # Full key can be path in nested dictionary
        if isinstance(full_key, tuple):
            for key in full_key:
                # If key exists in dictionary, keep searching deeper
                if key in dictionary_toSearch:
                    dictionary_toSearch = dictionary_toSearch[key]

                    # If found value, return it
                    if not isinstance(dictionary_toSearch, dict):
                        return dictionary_toSearch
                    # Continue searching children dictionary_toSearch
                    else:
                        continue
                # Else skip to next key
                else:
                    continue

        else:
            # If key exists in dictionary, return it
            if full_key in dictionary_toSearch:
                dictionary_toSearch = dictionary_toSearch[full_key]

                # If found value, return it
                if not isinstance(dictionary_toSearch, dict):
                    return dictionary_toSearch
                else:
                    raise ValueError(
                        "Key specifies dictionary not value", dictionary_toSearch
                    )
            # Else skip to next key
            else:
                continue

    raise ValueError("None of the keys found", dictionary_toSearch)


class ParseKwargs(argparse.Action):
    """
    Parse Kwargs into dictionary
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def format_modelName(model_name):
    """
    Removes any directory prefix in model_name and replace / with -

    Args:
        model_name:

    Returns:

    """
    return model_name.replace("/fruitbasket/models/", "").replace("/", "-")


def saveTo_gcp(should_saveToGCP, filepath):
    """

    Args:
        should_saveToGCP:
        filepath:

    Returns:

    """
    if should_saveToGCP:
        subprocess.call(
            f"gsutil "
            f"-m "
            f"-o GSUtil:parallel_composite_upload_threshold=150M "
            f"cp -r {filepath} gs://model_merging/{filepath}",
            shell=True,
        )


def parse_modelName(model_name):
    """
    Removes any directory prefix in model_name and replace / with -

    Args:
        model_name:

    Returns:

    """
    return model_name.replace("/fruitbasket/models/", "").replace("/", "-")


def get_median(list_ofNumbers):
    """


    Args:
        all_scores: list of dictionaries, where one of the value is the score we are interested in

    Returns:

    """
    return round(statistics.median(list_ofNumbers), 3)


def get_interquartileRange(list_ofNumbers):
    """


    Args:
        list_ofNumbers:

    Returns:

    """
    return round(iqr(list_ofNumbers), 3)


def get_average(list_ofNumbers):
    """

    Args:
        list_ofNumbers:

    Returns:

    """
    return round(mean(list_ofNumbers), 3)


def round_list(my_list, significant_figures):
    """

    Args:
        list:
        significant_figures:

    Returns:

    """
    rounded_list = []

    for number in my_list:
        rounded_list.append(round(number, significant_figures))

    return rounded_list


def round_nestedList(nested_list, significant_figures):
    """
    Round nested list of numbers where list can be any depth

    Args:
        nested_list:
        significant_figures:

    Returns:
        round_nestedList
    """
    rounded_nestedList = []
    for sublist in nested_list:
        if isinstance(sublist[0], list):
            rounded_sublist = round_nestedList(sublist, significant_figures)
        else:
            rounded_sublist = round_list(sublist, significant_figures)

        rounded_nestedList.append(rounded_sublist)

    return rounded_nestedList


def read_jsonl(filepath):
    """
    Read JSONL filepath

    Args:
        filepath:

    Returns:
    """
    json_lines = []

    with open(filepath, "r") as f:
        for idx, line in enumerate(f.readlines()):
            json_lines.append(json.loads(line.strip("\n")))

    return json_lines


def write_jsonl(listOfJson_toWrite, filepath):
    with open(filepath, "w+") as f:
        for json_toWrite in listOfJson_toWrite:
            f.write(json.dumps(json_toWrite))
            f.write("\n")


def append_json(json_toWrite, filepath):
    with open(filepath, "a+") as f:
        f.write(
            json.dumps(
                noIndent_dictOrList_onFirstLevel(json_toWrite),
                cls=NoIndentEncoder,
                indent=2,
            )
            + "\n"
        )


def append_jsonl(json_toWrite, filepath):
    with open(filepath, "a+") as f:
        f.write(json.dumps(json_toWrite))
        f.write("\n")


def deleteFiles_inDirectory(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


def map_forDictionaries(my_dict, map_fn):
    mapped_dict = {}
    for k, v in my_dict.items():
        mapped_dict[k] = map_fn(v)
    return mapped_dict


# From https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200
def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def getValueFromKey_matchingRegex(dict_ofRegexKeyToValue, key_toMatch):
    """

    Args:
        dict_regex_keyToValue (_type_): _description_
        key_toMatch (_type_): _description_

    Returns:
        _type_: _description_
    """
    matching_value = None
    for regex_key, value in dict_ofRegexKeyToValue.items():
        if re.search(regex_key, key_toMatch) is not None:
            matching_value = value
    return matching_value


def parse_configFilepaths(listOf_filepaths, types):
    parsed_filepaths = {}
    for type in types:
        parsed_filepaths[type] = []

    for filepath in listOf_filepaths:
        found_filepathType = False
        for filepath_type in parsed_filepaths.keys():
            listOf_filepathTypeStr = [
                "/" + filepath_type + "/",
                filepath_type.strip("s") + "_config.json",
            ]
            for filepathType_str in listOf_filepathTypeStr:
                if filepathType_str in filepath:
                    parsed_filepaths[filepath_type].append(filepath)
                    found_filepathType = True
                    break
        if not found_filepathType:
            raise ValueError(f"Cannot find type of filepath {filepath}")

    return parsed_filepaths


def print_mem_usage(loc):
    """
    Print memory usage in GB
    :return:
    """
    print(
        "%s mem usage: %.3f GB, %.3f GB, %.3f GB"
        % (
            loc,
            float(torch.cuda.memory_allocated() / 1e9),
            float(torch.cuda.memory_reserved() / 1e9),
            float(torch.cuda.max_memory_allocated() / 1e9),
        )
    )
    sys.stdout.flush()


def get_bestCheckpointIdx(experiment_dir):
    performance_fp = os.path.join(experiment_dir, "performance.json")
    if not os.path.exists(performance_fp):
        performance_fp = os.path.join(experiment_dir, "log.json")

    all_scores = read_jsonl(performance_fp)

    best_score = 0
    best_checkpointIdx = 0
    for score in all_scores:
        if score["score_to_select_checkpoint"] > best_score:
            best_score = score["score_to_select_checkpoint"]
            best_checkpointIdx = score["batch_idx"]

    return best_checkpointIdx

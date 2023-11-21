from tqdm import tqdm
import logging
import torch.distributed as dist

from src.data.dataset_readers import get_datasetReader, getDatasets_inMixture
from src.data.batches import getSingleEpoch_OfBatches
from src.data.PytorchDataset import PytorchDataset
from src.data.DatasetConfig import DatasetConfig

from src.utils.utils import deep_update

from src.eval.utils import *
from src.eval.Evaluator import Evaluator
from src.eval.EvaluationConfig import EvaluationConfig

from src.utils.distributed import (
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)

import torch


def evaluation(
    model,
    evaluation_config,
    prediction_dir,
    cached_singleDatasetReaders,
    world_size,
    device,
):
    """

    Args:
        model:
        evaluation_config:
        num_samples:
        prediction_dir:
        cached_singleDatasetReaders:
        world_size
        device:

    Returns:

    """
    logging.info(f"Evaluating model")

    evaluationDataset_config = evaluation_config.get_datasetConfig()
    dataset_reader, cached_singleDatasetReaders = get_datasetReader(
        dataset_mixture=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders=cached_singleDatasetReaders,
    )

    model.eval()

    # When using DDP model, the model is wrapped with a DDP model and must be called with
    # model.module
    if is_distributedSetup(world_size):
        model = model.module

    pytorch_dataset = PytorchDataset(
        dataset_reader.get_dataset(
            evaluationDataset_config.split,
            evaluationDataset_config.template_idx,
            use_answerChoices=True,
            num_samples=evaluation_config.num_samples,
        ),
        evaluationDataset_config,
        model.tokenize_fn,
        device=device,
    )

    evalBatch_iterator = getSingleEpoch_OfBatches(
        pytorch_dataset, evaluation_config.eval_batch_size, world_size, device
    )
    metrics = dataset_reader.get_datasetMetrics()

    if is_nodeZero(device):
        evaluator = Evaluator(evaluation_config, metrics, prediction_dir)

    with torch.no_grad():
        for batch in tqdm(evalBatch_iterator):
            batchOf_evalInfo = prepare_batchOfEvalInfo(batch)

            if "Accuracy" in metrics:
                if evaluation_config.use_bfloat16_during_eval:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        (
                            predicted_choice,
                            score_ofChoices,
                            logProbs_ofAllChoicesIds,
                            len_allChoices,
                        ) = model.predict_mulChoice(
                            batch,
                            evaluation_config.length_normalization,
                            evaluation_config.sample_tokens,
                        )
                else:
                    (
                        predicted_choice,
                        score_ofChoices,
                        logProbs_ofAllChoicesIds,
                        len_allChoices,
                    ) = model.predict_mulChoice(
                        batch, evaluation_config.length_normalization
                    )

                batchOf_evalInfo.update(
                    {
                        "predicted_choice": predicted_choice,
                        "score_of_choices": score_ofChoices,
                        "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                        "len_all_choices": len_allChoices,
                    }
                )

            if "Squad" in metrics or "F1" in metrics:
                if evaluation_config.use_bfloat16_during_eval:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        generated_ids, generated_txt = model.generate(
                            batch,
                            evaluation_config.max_gen_len,
                            evaluation_config.sample_tokens,
                        )
                else:
                    generated_ids, generated_txt = model.generate(
                        batch,
                        evaluation_config.max_gen_len,
                        evaluation_config.sample_tokens,
                    )

                batchOf_evalInfo.update(
                    {"generated_ids": generated_ids, "prediction_text": generated_txt}
                )

            if is_distributedSetup(world_size):
                gathered_batchOfEvalInfo = [{}] * world_size

                dist.gather_object(
                    batchOf_evalInfo,
                    gathered_batchOfEvalInfo if is_nodeZero(device) else None,
                    dst=0,
                )

                if is_nodeZero(device):
                    batchOf_evalInfo = reduce_gatheredOutput(gathered_batchOfEvalInfo)

            if is_nodeZero(device):
                evaluator.add_batch(batchOf_evalInfo)

    if is_nodeZero(device):
        results = {
            "score": evaluator.get_result(),
            "run_dir": evaluator.get_evaluationRunDir(),
            "run_config": evaluation_config.get_key_values(),
            "dataset_config": evaluation_config.get_datasetConfig().get_key_values(),
        }
        return (
            results,
            cached_singleDatasetReaders,
        )
    else:
        return None, cached_singleDatasetReaders


def evaluate_split(
    model,
    evaluation_config,
    prediction_dir,
    cached_singleDatasetReaders,
    world_size,
    device,
):
    if evaluation_config.get_datasetConfig().split == "train_validation":
        train_evaluationConfig = updateDatasetArgs_inEvaluationConfig(
            evaluation_config, {"split": "train"}
        )
        train_results, _ = evaluation(
            model,
            train_evaluationConfig,
            prediction_dir,
            cached_singleDatasetReaders,
            world_size,
            device,
        )
        validation_evaluationConfig = updateDatasetArgs_inEvaluationConfig(
            evaluation_config, {"split": "validation"}
        )
        validation_results, cached_singleDatasetReaders = evaluation(
            model,
            validation_evaluationConfig,
            prediction_dir,
            cached_singleDatasetReaders,
            world_size,
            device,
        )
        # Combine the results from different splits
        results = {}
        for split, split_results in zip(
            ["train", "validation"], [train_results, validation_results]
        ):
            for key, value in split_results.items():
                if key in ["score", "run_dir"]:
                    if key in results:
                        results[key].update({split: value})
                    else:
                        results[key] = {split: value}
                else:
                    results[key] = value

        return results, cached_singleDatasetReaders

    else:
        return evaluation(
            model,
            evaluation_config,
            prediction_dir,
            cached_singleDatasetReaders,
            world_size,
            device,
        )


def evaluate_singleTemplate(
    model,
    evaluation_config,
    prediction_dir,
    cached_singleDatasetReaders,
    world_size,
    device,
):
    singleTemplate_scores = None
    runs_dir = None

    # Evaluate each dataset in the mixture separately
    if evaluation_config.dataset_mixture is not None:
        list_ofResults = []

        for dataset in getDatasets_inMixture(evaluation_config.dataset_mixture):
            dataset_evaluationConfig = updateDatasetArgs_inEvaluationConfig(
                evaluation_config, {"dataset": dataset}
            )
            results, cached_singleDatasetReaders = evaluate_split(
                model,
                dataset_evaluationConfig,
                prediction_dir,
                cached_singleDatasetReaders,
                world_size,
                device,
            )
            list_ofResults.append(results)

        # If results is None, then we are doing distributed data parallel and
        # only the node zero has to store the results
        if list_ofResults[0] is not None:
            average_score = average_scores(list_ofResults)
            singleTemplate_scores = concatenate_scores(
                list_ofResults, lambda x: x["dataset_config"]["dataset"]
            )
            singleTemplate_scores = deep_update(singleTemplate_scores, average_score)
            runs_dir = get_runsDirs(list_ofResults)

    # Evaluate single dataset
    else:
        assert evaluation_config.get_datasetConfig().dataset is not None

        results, cached_singleDatasetReaders = evaluate_split(
            model,
            evaluation_config,
            prediction_dir,
            cached_singleDatasetReaders,
            world_size,
            device,
        )
        if results is not None:
            singleTemplate_scores = results["score"]
            runs_dir = results["run_dir"]

    return singleTemplate_scores, runs_dir, cached_singleDatasetReaders


def evaluate_multipleTemplates(
    model,
    evaluation_config,
    prediction_dir,
    cached_singleDatasetReaders,
    world_size,
    device,
):
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    assert evaluationDataset_config.template_idx is None

    multipleTemplate_scores = None
    runs_dir = None

    # Evaluate each dataset and template pair in the mixture separately
    if evaluation_config.dataset_mixture is not None:
        list_ofResults = []

        for dataset in getDatasets_inMixture(evaluation_config.dataset_mixture):
            # dataset_mixture is None since we we want the dataset_reader for a single dataset
            dataset_reader, cached_singleDatasetReaders = get_datasetReader(
                dataset_mixture=None,
                dataset_config=evaluationDataset_config,
                cached_singleDatasetReaders=cached_singleDatasetReaders,
            )

            num_templates = dataset_reader.get_numTemplates()

            # Loop through all templates
            for template_idx in range(num_templates):
                datasetTemplate_evaluationConfig = updateDatasetArgs_inEvaluationConfig(
                    evaluation_config,
                    {
                        "dataset": dataset,
                        "template_idx": template_idx,
                    },
                )

                results, cached_singleDatasetReaders = evaluate_split(
                    model,
                    datasetTemplate_evaluationConfig,
                    prediction_dir,
                    cached_singleDatasetReaders,
                    world_size,
                    device,
                )
                list_ofResults.append(results)

        # Get the median score per dataset and the average of the median scores
        if list_ofResults[0] is not None:
            groupScores_byDataset = group_by(
                list_ofResults, lambda x: x["dataset_config"]["dataset"]
            )
            summaryOfScores_perDataset = map_forDictionaries(
                my_dict=groupScores_byDataset,
                map_fn=get_summaryOfScores_acrossTemplates,
            )
            concatentedScores_perDataset = map_forDictionaries(
                my_dict=groupScores_byDataset, map_fn=concatenate_scores
            )
            averageMedianScore_acrossDataset = get_average(
                list(
                    map(
                        lambda dataset: summaryOfScores_perDataset[dataset]["median"],
                        summaryOfScores_perDataset.keys(),
                    )
                )
            )
            multipleTemplate_scores = deep_update(
                deep_update(summaryOfScores_perDataset, concatentedScores_perDataset),
                averageMedianScore_acrossDataset,
            )
            runs_dir = get_runsDirs(list_ofResults)

    # Evaluate single dataset
    else:
        assert evaluation_config.get_datasetConfig().dataset is not None

        dataset_reader, cached_singleDatasetReaders = get_datasetReader(
            dataset_mixture=None,
            dataset_config=evaluation_config.get_datasetConfig(),
            cached_singleDatasetReaders=cached_singleDatasetReaders,
        )

        num_templates = dataset_reader.get_numTemplates()
        # Loop through all templates
        for template_idx in range(num_templates):
            template_evaluationConfig = updateDatasetArgs_inEvaluationConfig(
                evaluation_config,
                {
                    "template_idx": template_idx,
                },
            )
            results, cached_singleDatasetReaders = evaluate_split(
                model,
                template_evaluationConfig,
                prediction_dir,
                cached_singleDatasetReaders,
                world_size,
                device,
            )
            list_ofResults.append(results)

        # Get median score per dataset
        if list_ofResults[0] is not None:
            summary_ofScores = get_summaryOfScores_acrossTemplates(list_ofResults)
            concatenated_scores = concatenate_scores(list_ofResults)
            median_score = summary_ofScores["median"]
            multipleTemplate_scores = deep_update(
                median_score,
                deep_update(summary_ofScores, concatenated_scores),
            )
            runs_dir = get_runsDirs(list_ofResults)

    return multipleTemplate_scores, runs_dir, cached_singleDatasetReaders

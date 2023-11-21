import torch
import argparse
import logging
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel

from src.model.Checkpointer import Checkpointer
from src.model.load_model import (
    load_model,
)
from src.model.ModelConfig import ModelConfig
from src.model.utils import get_trainableParameters

from src.train.TrainingConfig import TrainingConfig
from src.train.utils import construct_optimizer, construct_scheduler

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.utils import average_scores, concatenate_scores
from src.eval.evaluate import evaluate_singleTemplate

from src.utils.utils import ParseKwargs, set_seeds, get_average, parse_configFilepaths
from utils.distributed import (
    reduce_gatheredOutput,
    is_nodeZero,
    is_distributedSetup,
)

from src.data.batches import getMultipleEpochs_ofBatches
from src.data.dataset_readers import get_datasetReader, getDatasets_inMixture
from src.data.PytorchDataset import PytorchDataset
from src.data.DatasetConfig import DatasetConfig


def evaluate_checkpoint(
    model,
    training_config,
    cached_singleDatasetReaders,
    batch_idx,
    world_size,
    device,
):
    """

    Args:
        model:
        evaluation_config:
        cached_singleDatasetReaders:
        batch_idx:
        world_size:
        device:

    Returns:

    """
    logging.info(f"Evaluating checkpoint")

    evaluation_config = training_config.get_evaluationConfig()
    # Assume during training, we only evaluate a single template
    assert evaluation_config.get_datasetConfig().template_idx != -2

    checkpoint_scores, _, cached_singleDatasetReaders = evaluate_singleTemplate(
        model,
        evaluation_config,
        os.path.join(
            training_config.experiment_dir, "predictions", f"batch_{batch_idx}"
        ),
        cached_singleDatasetReaders,
        world_size,
        device,
    )

    if checkpoint_scores is not None:
        if evaluation_config.get_datasetConfig().split == "train_validation":
            average_score = checkpoint_scores["validation"]["average"]
        else:
            average_score = checkpoint_scores["average"]
        checkpoint_scores["score_to_select_checkpoint"] = average_score

    return checkpoint_scores, cached_singleDatasetReaders


def train(device, world_size, port, training_config):
    if is_distributedSetup(world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=device, world_size=world_size)

    set_seeds(training_config.seed)

    model_config = training_config.get_modelConfig()

    model, trainableParameter_regex = load_model(model_config, device=device)

    trainable_parameters, num_parameters = get_trainableParameters(
        model, trainableParameter_regex
    )
    print(f"{num_parameters} parameters")

    optimizer = construct_optimizer(
        list(trainable_parameters.values()), training_config
    )

    scheduler = None
    if training_config.scheduler is not None:
        scheduler = construct_scheduler(optimizer, training_config)

    if is_distributedSetup(world_size):
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    trainingDataset_config = training_config.get_datasetConfig()

    dataset_reader, cached_singleDatasetReaders = get_datasetReader(
        training_config.train_dataset_mixture,
        trainingDataset_config,
        cached_singleDatasetReaders=None,
    )

    if is_distributedSetup(world_size):
        tokenize_fn = model.module.tokenize_fn
    else:
        tokenize_fn = model.tokenize_fn

    pytorch_dataset = PytorchDataset(
        dataset_reader.get_dataset(
            "train",
            trainingDataset_config.template_idx,
            use_answerChoices=False,
            num_samples=-1,
        ),
        trainingDataset_config,
        tokenize_fn,
        device,
    )

    if is_nodeZero(device):
        checkpointer = Checkpointer(training_config, world_size)

    if training_config.should_eval_before_training:
        logging.info(f"Evaluating before training")
        checkpoint_scores, cached_singleDatasetReaders = evaluate_checkpoint(
            model,
            training_config,
            cached_singleDatasetReaders,
            0,
            world_size,
            device,
        )

        if is_nodeZero(device):
            checkpointer.checkpoint(
                model, trainable_parameters, checkpoint_scores, batch_idx=0
            )

    if training_config.use_bfloat16_during_training:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    if training_config.train_batch_size % (
        training_config.micro_train_batch_size * world_size
    ) != 0 or training_config.train_batch_size < (
        training_config.micro_train_batch_size * world_size
    ):
        raise ValueError(
            f"train_batch_size {training_config.train_batch_size} with micro_train_batch_size {training_config.micro_train_batch_size} and world size {world_size}"
        )

    gradient_accumulation_factor = training_config.train_batch_size // (
        training_config.micro_train_batch_size * world_size
    )

    train_iterator = getMultipleEpochs_ofBatches(
        pytorch_dataset,
        training_config.micro_train_batch_size,
        should_shuffle=True,
        world_size=world_size,
        device=device,
    )

    for i in tqdm(range(training_config.num_batches * gradient_accumulation_factor)):
        batch_idx = i // (gradient_accumulation_factor)
        model.train()

        train_batch = next(train_iterator)

        if training_config.use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, current_metrics = model(train_batch)
                loss = loss / gradient_accumulation_factor
            scaler.scale(loss).backward()
        else:
            loss, current_metrics = model(train_batch)
            loss = loss / gradient_accumulation_factor
            loss.backward()

        if is_distributedSetup(world_size):
            gathered_currentMetrics = [{}] * world_size
            dist.gather_object(
                current_metrics,
                gathered_currentMetrics if is_nodeZero(device) else None,
                dst=0,
            )

            if is_nodeZero(device):
                current_metrics = reduce_gatheredOutput(
                    gathered_currentMetrics, get_average
                )

        if is_nodeZero(device):
            checkpointer.log_metric(current_metrics, batch_idx)

        if (i + 1) % gradient_accumulation_factor == 0:
            # Clip norm of gradient
            if training_config.norm_to_clip_gradient is not None:
                # Unscale gradient if using bfloat16 so clipping can be correct magnitude
                if training_config.use_bfloat16_during_training:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(trainable_parameters.values()),
                    training_config.norm_to_clip_gradient,
                )

            # Take a gradient step
            if training_config.use_bfloat16_during_training:
                scaler.step(optimizer)
                scaler.update()
                if training_config.scheduler is not None:
                    scheduler.step()
            else:
                optimizer.step()
                if training_config.scheduler is not None:
                    scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()

            if (batch_idx + 1) % training_config.checkpoint_frequency == 0:
                checkpoint_scores, cached_singleDatasetReaders = evaluate_checkpoint(
                    model,
                    training_config,
                    cached_singleDatasetReaders,
                    batch_idx,
                    world_size,
                    device,
                )

                if is_nodeZero(device):
                    (
                        current_log,
                        should_stopTraining,
                    ) = checkpointer.checkpoint(
                        model, trainable_parameters, checkpoint_scores, batch_idx
                    )

                    logging.info(f"Finished {batch_idx} batches with log {current_log}")
                    if should_stopTraining:
                        if is_distributedSetup(world_size):
                            dist.destroy_process_group()
                        return

    if is_distributedSetup(world_size):
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_filepaths", action="store", type=str, nargs="*", required=True
    )
    parser.add_argument(
        "-m", "--model_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-td", "--training_dataset_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-tr", "--training_run_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-ed", "--evaluation_dataset_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    parser.add_argument(
        "-er", "--evaluation_run_kwargs", nargs="*", action=ParseKwargs, default={}
    )

    parser.add_argument("-w", "--world_size", default=1, type=int)
    parser.add_argument("-p", "--port", default=12345, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training")

    parsed_filepaths = parse_configFilepaths(
        args.config_filepaths,
        types=[
            "models",
            "training_dataset",
            "training_run",
            "evaluation_dataset",
            "evaluation_run",
        ],
    )

    model_config = ModelConfig(
        config_filepaths=parsed_filepaths["models"], update_dict=args.model_kwargs
    )
    training_dataset_config = DatasetConfig(
        "train",
        config_filepaths=parsed_filepaths["training_dataset"],
        update_dict=args.training_dataset_kwargs,
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

    training_config = TrainingConfig(
        model_config,
        training_dataset_config,
        evaluation_config,
        config_filepaths=parsed_filepaths["training_run"],
        update_dict=args.training_run_kwargs,
    )

    if args.world_size != 1:
        mp.spawn(
            train,
            args=(args.world_size, args.port, training_config),
            nprocs=args.world_size,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(device, args.world_size, args.port, training_config)

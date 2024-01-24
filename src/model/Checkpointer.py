import os
import torch
import json
import re
import wandb

from collections import OrderedDict
from src.utils.utils import (
    getValueOfKey_inDictionary,
    saveTo_gcp,
    safe_makedirs,
    append_json,
    append_jsonl,
    deleteFiles_inDirectory,
)
from utils.distributed import is_distributedSetup

METRICS_PRIORITY = ["score_to_select_checkpoint", "loss", "batch_idx"]


class Checkpointer(object):
    def __init__(self, training_config, world_size):
        self.training_config = training_config
        self.world_size = world_size

        if training_config.log_wandb:
            exp_name = training_config.experiment_dir.replace("exp_out/", "").replace(
                "/", "_"
            )
            wandb.init(
                project="model_merging",
                name=exp_name,
                id=exp_name,
                dir=training_config.experiment_dir,
                resume=True,
                entity="raffel-reports",
            )

        self.runningSum_ofLogs = {}
        self.number_ofLogs = 0

        self.current_bestScore = 0

        if self.training_config.use_early_stopping:
            self.numCheckpoints_sinceBestCheckpoint = 0

    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    def _convertDistributedDict_toNonDistributedDict(self, state_dict):
        if is_distributedSetup(self.world_size):
            nonDistributed_stateDict = OrderedDict()
            pattern = re.compile("module.")
            for k, v in state_dict.items():
                if re.search("module", k):
                    nonDistributed_stateDict[re.sub(pattern, "", k)] = v
                else:
                    nonDistributed_stateDict = state_dict

            return nonDistributed_stateDict
        else:
            return state_dict

    def _is_bestCheckpoint(self, current_log):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        return current_score > self.current_bestScore

    def _update_bestCheckpoint(self, current_log):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        self.current_bestScore = current_score
        if self.training_config.use_early_stopping:
            self.numCheckpoints_sinceBestCheckpoint = 0

    def _save_checkpoint(self, trainable_parameters, save_fp):
        torch.save(
            self._convertDistributedDict_toNonDistributedDict(trainable_parameters),
            save_fp,
        )
        saveTo_gcp(self.training_config.should_save_training_run_to_gcp, save_fp)

    def log_metric(self, current_metrics, batch_idx):
        """
        Update running sum of metrics

        Args:
            current_metrics:

        """

        for k in current_metrics.keys():
            scaled_metric = current_metrics[k]

            if k not in self.runningSum_ofLogs:
                self.runningSum_ofLogs[k] = scaled_metric
            else:
                self.runningSum_ofLogs[k] += scaled_metric

        if self.training_config.log_wandb:
            wandb.log(current_metrics, step=batch_idx)

        self.number_ofLogs += 1

    def _get_averageMetrics(self):
        """
        Get average metric per batch since the last time we got the average.

        Note that average is per effective batch size, not batch size
        (i.e. every gradient update, not every forward pass).

        :return: average_metric
        """
        average_metric = {}
        for k in self.runningSum_ofLogs.keys():
            average_metric[k] = float(
                "%.3f" % (self.runningSum_ofLogs[k] / self.number_ofLogs)
            )

        # Reset running dict_metrics and counter when we take average
        self.runningSum_ofLogs = {}
        self.number_ofLogs = 0

        return average_metric

    def _write_performance(self, batch_idx, evaluation_scores):
        """
        Save scores and average metrics during training

        Args:
            batch_idx:
            scores:

        Returns:

        """
        performance = {}
        performance["batch_idx"] = batch_idx
        performance.update(evaluation_scores)
        performance.update(self._get_averageMetrics())

        checkpoint_fp = os.path.join(
            self.training_config.experiment_dir, "performance.json"
        )
        append_jsonl(performance, checkpoint_fp)
        saveTo_gcp(self.training_config.should_save_training_run_to_gcp, checkpoint_fp)

        return performance

    def checkpoint(self, model, trainable_parameters, evaluation_scores, batch_idx):
        """
        Handles checkpointing which means
        1) saving metrics and scores
        2) saving the model if needed

        Args:
            trainable_parameters:
            scores:
            batch_idx:


        Returns:
            current_log
        """

        if self.training_config.log_wandb:
            wandb.log(evaluation_scores, step=batch_idx)

        performance = self._write_performance(batch_idx, evaluation_scores)

        if self.training_config.use_early_stopping:
            self.numCheckpoints_sinceBestCheckpoint += 1

        # Don't need to save model at the beginning since it hasn't changed from the pretrained model
        if batch_idx > 0:
            checkpoint_dir = os.path.join(
                self.training_config.experiment_dir, "checkpoints"
            )
            safe_makedirs(checkpoint_dir)
            checkpoint_fp = os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pt")

            if self.training_config.should_save_every_checkpoint:
                self._save_checkpoint(
                    trainable_parameters,
                    checkpoint_fp,
                )
            # If we don't save every checkpoint, then we only save the best checkpoint
            else:
                if self._is_bestCheckpoint(performance):
                    deleteFiles_inDirectory(checkpoint_dir)
                    self._save_checkpoint(
                        trainable_parameters,
                        checkpoint_fp,
                    )

        # Update the best checkpoint outside
        if self._is_bestCheckpoint(performance):
            self._update_bestCheckpoint(performance)

        should_stopTraining = False
        if self.training_config.use_early_stopping and (
            self.numCheckpoints_sinceBestCheckpoint
            >= self.training_config.early_stopping_num_checkpoints_without_improvement
        ):
            should_stopTraining = True

        return performance, should_stopTraining
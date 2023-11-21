import os
import json

from src.eval.Scorer import Scorer
from src.eval.PredictionWriter import PredictionWriter


class Evaluator(object):
    def __init__(self, evaluation_config, metrics, prediction_dir):
        """
        Evaluates all metrics for a dataset

        Args:
            evaluation_config:
            metrics:
            prediction_fp:
        """
        self.evaluation_config = evaluation_config
        self.scorer = Scorer(evaluation_config, metrics)
        self.writer = PredictionWriter(evaluation_config, prediction_dir)

        self.seen_idxs = {}

    def add_batch(self, batchOf_evalInfo):
        batchOf_idxs = batchOf_evalInfo["idx"]

        # For distributed setup, the batch might have duplicate examples due to padding that we
        # have to remove.
        # 1) Compute the indices we have to remove
        idx_toRemove = []
        for batch_idx, idx in enumerate(batchOf_idxs):
            if idx in self.seen_idxs:
                idx_toRemove.append(batch_idx)
            self.seen_idxs[idx] = True

        # 2) Remove these indices
        filteredBatch_ofEvalInfo = {}
        for key, batchOf_values in batchOf_evalInfo.items():
            filtered_value = []
            for batch_idx, value in enumerate(batchOf_values):
                if batch_idx not in idx_toRemove:
                    filtered_value.append(value)

            filteredBatch_ofEvalInfo[key] = filtered_value

        self.scorer.add_batch(filteredBatch_ofEvalInfo)
        self.writer.log_batch(filteredBatch_ofEvalInfo)

    def get_result(self):
        """

        Returns:

        """
        return self.scorer.get_score()

    def get_evaluationRunDir(self):
        self.writer.close_logger()
        return self.writer.get_evaluationRunDir()

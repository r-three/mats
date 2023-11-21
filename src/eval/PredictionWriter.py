import json
import os

from src.utils.utils import convert_dictOfLists_to_listOfDicts, saveTo_gcp


class PredictionWriter(object):
    def __init__(self, evaluation_config, prediction_dir):
        """

        Args:
            evaluation_config:
            logger_fp:
        """
        self.evaluation_config = evaluation_config

        self.predictions_file = None
        if prediction_dir is not None:
            self.evaluationRun_dir = self._open_writer(prediction_dir)

    def _get_nextRunIdx(self, finalPrediction_dir):
        """Get the max run that exists

        Args:
            finalPrediction_dir (_type_): _description_

        Returns:
            _type_: _description_
        """
        run_idx = 0
        while True:
            run_dir = os.path.join(finalPrediction_dir, f"run_{run_idx}")
            if not os.path.exists(run_dir):
                return run_idx
            run_idx += 1

    def _open_writer(self, prediction_dir):
        """

        Args:
            evaluation_config:

        Returns:

        """

        evaluationDataset_config = self.evaluation_config.get_datasetConfig()

        finalPrediction_dir = os.path.join(
            prediction_dir,
            evaluationDataset_config.dataset
            + f"_template_{evaluationDataset_config.template_idx}",
            f"{evaluationDataset_config.split}",
        )

        next_runIdx = self._get_nextRunIdx(finalPrediction_dir)
        # Use the previous run if we want to overwrite it
        if self.evaluation_config.overwrite_previous_run and next_runIdx > 0:
            next_runIdx -= 1

        evaluationRun_dir = os.path.join(finalPrediction_dir, f"run_{next_runIdx}")
        if not os.path.exists(evaluationRun_dir):
            os.makedirs(evaluationRun_dir)

        evaluationRunConfig_fp = os.path.join(
            evaluationRun_dir, f"evaluation_run_config.json"
        )
        evaluationDatasetConfig_fp = os.path.join(
            evaluationRun_dir, f"evaluation_dataset_config.json"
        )
        self.predictions_fp = os.path.join(evaluationRun_dir, f"predictions.json")

        self.evaluation_config._save_config(
            evaluationRunConfig_fp,
            self.evaluation_config.should_save_evaluation_run_to_gcp,
        )
        self.evaluation_config.get_datasetConfig()._save_config(
            evaluationDatasetConfig_fp,
            self.evaluation_config.should_save_evaluation_run_to_gcp,
        )

        self.predictions_file = open(self.predictions_fp, "w+")
        return evaluationRun_dir

    def log_batch(self, batchOf_evalInfo):
        if self.predictions_file is not None:
            listOf_evalInfo = convert_dictOfLists_to_listOfDicts(batchOf_evalInfo)
            for eval_info in listOf_evalInfo:
                self.predictions_file.write(json.dumps(eval_info) + "\n")
            self.predictions_file.flush()

    def close_logger(self):
        if self.predictions_file is not None:
            saveTo_gcp(
                self.evaluation_config.should_save_evaluation_run_to_gcp,
                self.predictions_fp,
            )

    def get_evaluationRunDir(self):
        if self.predictions_file is not None:
            return self.evaluationRun_dir
        else:
            return None

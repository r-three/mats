import json
import os
import logging
import datetime

from shutil import copytree, ignore_patterns

from src.utils.utils import safe_makedirs, saveTo_gcp
from src.utils.Config import Config


class TrainingConfig(Config):
    def __init__(
        self,
        model_config,
        trainingDataset_config,
        evaluation_config,
        config_filepaths=None,
        update_dict=None,
    ):
        super().__init__()

        """
        other configs that are part of training run 
        """
        self.model_config = model_config
        self.train_dataset_config = trainingDataset_config
        self.evaluation_config = evaluation_config

        self.train_dataset_mixture = None

        """
        training run parameters 
        """
        self.micro_train_batch_size = None
        self.train_batch_size = None
        self.num_batches = None
        self.use_bfloat16_during_training = None

        """
        checkpoint parameters 
        """
        self.initial_checkpoint_filepath = None
        self.checkpoint_frequency = None
        self.use_early_stopping = None
        self.early_stopping_num_checkpoints_without_improvement = None
        self.should_save_every_checkpoint = None
        self.should_save_training_run_to_gcp = None
        self.should_eval_before_training = None
        self.log_wandb = None
        """
        optimization parameters 
        """
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_ratio = None
        self.weight_decay = None
        self.norm_to_clip_gradient = None

        """
        reproducabilty parameters 
        """
        self.seed = None
        self.experiment_name = None
        self.experiment_dir = None

        # Update config with values from list of files
        if config_filepaths:
            for filename in config_filepaths:
                super()._update_fromDict(
                    json.load(open(filename)),
                    prefix_ofUpdateKey=None,
                    assert_keyInUpdateDict_isValid=True,
                )

        # Update config with values from dict
        if update_dict:
            super()._update_fromDict(
                update_dict,
                prefix_ofUpdateKey=None,
                assert_keyInUpdateDict_isValid=True,
            )

        if self.experiment_dir is None:
            self.get_experimentDir()
            self._save_config(
                os.path.join(self.experiment_dir, "training_run_config.json"),
                self.should_save_training_run_to_gcp,
            )
            self.train_dataset_config._save_config(
                os.path.join(self.experiment_dir, "training_dataset_config.json"),
                self.should_save_training_run_to_gcp,
            )
            self.model_config._save_config(
                os.path.join(self.experiment_dir, "model_config.json"),
                self.should_save_training_run_to_gcp,
            )

    def get_experimentDir(self):
        now = datetime.datetime.now()
        timestamp = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        self.experiment_dir = "exp_out"

        if self.train_dataset_mixture is not None:
            self.experiment_dir = os.path.join(
                self.experiment_dir, self.train_dataset_mixture
            )

        self.experiment_dir = os.path.join(
            self.experiment_dir, self.train_dataset_config.get_experimentDir()
        )

        self.experiment_dir = os.path.join(
            self.experiment_dir, self.model_config.get_experimentDir()
        )

        if self.experiment_name is not None:
            self.experiment_dir = os.path.join(
                self.experiment_dir, self.experiment_name
            )

        self.experiment_dir = os.path.join(self.experiment_dir, timestamp)
        safe_makedirs(self.experiment_dir)

        copytree(
            os.path.join(os.environ["MMS_ROOT"], "src"),
            os.path.join(self.experiment_dir, "src"),
            ignore=ignore_patterns("*.pyc", "tmp*"),
        )

        saveTo_gcp(self.should_save_training_run_to_gcp, self.experiment_dir)

    def get_modelConfig(self):
        return self.model_config

    def get_evaluationConfig(self):
        return self.evaluation_config

    def get_datasetConfig(self):
        return self.train_dataset_config

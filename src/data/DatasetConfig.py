import json
import os

from src.utils.Config import Config


class DatasetConfig(Config):
    def __init__(
        self,
        key_prefix,
        config_filepaths=None,
        update_dict=None,
    ):
        """

        Args:
            config_filepaths
            kwargs:
        """
        super().__init__()
        assert key_prefix in ["train", "evaluation", None]
        self.instruction_format = None
        self.dataset = None
        self.split = None
        self.template_idx = None

        self.max_examples_per_dataset_for_mixture = None

        if key_prefix is not None:
            prefix_ofUpdateKey = key_prefix + "_"
        else:
            prefix_ofUpdateKey = ""

        # Update config with values from list of files
        if config_filepaths:
            for filename in config_filepaths:
                super()._update_fromDict(
                    json.load(open(filename)),
                    prefix_ofUpdateKey=prefix_ofUpdateKey,
                    assert_keyInUpdateDict_isValid=True,
                )

        # Update config with values from dict
        if update_dict:
            super()._update_fromDict(
                update_dict,
                prefix_ofUpdateKey=prefix_ofUpdateKey,
                assert_keyInUpdateDict_isValid=True,
            )

    def get_experimentDir(self):
        experiment_dir = ""

        if self.instruction_format is not None:
            experiment_dir = os.path.join(experiment_dir, self.instruction_format)

        if self.dataset is not None:
            experiment_dir = os.path.join(experiment_dir, self.dataset)

        return experiment_dir

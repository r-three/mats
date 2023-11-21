import os
import json

from src.utils.Config import Config
from src.utils.utils import format_modelName, parse_modelName


class ModelConfig(Config):
    def __init__(
        self,
        config_filepaths=None,
        update_dict=None,
    ):
        """

        Args:
            configDict_toInitializeFrom:
            fields_toUpdate:
            kwargs:
        """
        super().__init__()

        self.pretrained_model = None
        self.max_seq_len = None

        self.peft_method = None

        self.filepath_to_load_model = None

        # PEFT hyperparameters
        self.lora_r = None
        self.lora_modules = None
        self.prompt_tuning_num_prefix_emb = None

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

    def get_experimentDir(self):
        experiment_dir = format_modelName(self.pretrained_model)

        if self.peft_method is not None:
            experiment_dir = os.path.join(
                experiment_dir, parse_modelName(self.peft_method)
            )
        else:
            experiment_dir = os.path.join(experiment_dir, "full_model")

        if self.linearize:
            experiment_dir = os.path.join(experiment_dir, "linearize")

            if self.use_linearize:
                experiment_dir = os.path.join(experiment_dir, "use_linearize")
            else:
                assert self.use_linearize == False
                experiment_dir = os.path.join(experiment_dir, "use_function")

        return experiment_dir

import json
from src.utils.utils import (
    safe_makedirs,
    saveTo_gcp,
)

from src.utils.Config import Config


class EvaluationConfig(Config):
    def __init__(
        self,
        evaluationDataset_config,
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
        self.evaluationDataset_config = evaluationDataset_config

        self.dataset_mixture = None

        self.max_gen_len = None
        self.sample_tokens = None
        self.eval_batch_size = None

        self.length_normalization = None
        self.use_bfloat16_during_eval = None

        self.should_save_evaluation_run_to_gcp = None
        self.overwrite_previous_run = None

        self.num_samples = None

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

    def get_datasetConfig(self):
        return self.evaluationDataset_config

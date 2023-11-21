import copy
import ast
import json

from src.utils.utils import saveTo_gcp


class Config(object):
    def get_key_values(self):
        """
        Get the key/values in config which are indicated by the value not being an object

        Returns:

        """
        key_values = {}
        for key, value in self.__dict__.items():
            if not isinstance(value, Config):
                key_values[key] = value
        return key_values

    def _update_fromDict(
        self, dict_toUpdateFrom, prefix_ofUpdateKey, assert_keyInUpdateDict_isValid
    ):
        """

        Args:
            dict_toUpdateFrom:
            assert_keyInUpdateDict_isValid: If True, then error is thrown if key in the dict_toUpdateFrom does
                not exist in self.config

        Returns:

        """
        for k, v in dict_toUpdateFrom.items():
            # Check prefix is actually prefix and then remove prefix when updating config
            if prefix_ofUpdateKey is not None:
                assert k.startswith(prefix_ofUpdateKey)
                k = k[len(prefix_ofUpdateKey) :]

            try:
                # For strings that are actually filepaths, literal eval will fail so we have to ignore
                # strings which are filepaths. We check a string is a filepath if a "/" is in string.
                if not (isinstance(v, str) and "/" in v):
                    v = ast.literal_eval(v)
            except ValueError:
                v = v

            if hasattr(self, k):
                setattr(self, k, v)

            else:
                if assert_keyInUpdateDict_isValid:
                    raise ValueError(f"{k} is not in the config")

    def _save_config(self, config_fp, shouldSave_toGCP):
        """
        Save config at filename

        Args:
            filename:

        Returns:

        """
        with open(config_fp, "w+") as f:
            f.write(json.dumps(self.get_key_values(), indent=4, sort_keys=True))
            f.write("\n")

        saveTo_gcp(shouldSave_toGCP, config_fp)

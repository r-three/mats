import torch
import torch.nn as nn
import re


class IA3ElementwiseMultiply(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.ia3_vector = nn.Parameter(
            torch.ones(
                out_features,
            )
        )

    def forward(self, input):
        return input * self.ia3_vector


class IA3LinearLayer(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.linear_layer = linear_layer
        self.ia3_layer = IA3ElementwiseMultiply(self.linear_layer.out_features)

    def forward(self, input):
        hidden = self.linear_layer(input)
        output = self.ia3_layer(hidden)
        return output


def modify_withIA3(transformer, model_config):
    """

    Args:
        transformer:

    Returns:

    """
    module_toModify = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
    children_toModify = "k|v|wi_1.*"
    trainableParameter_regex = ".*ia3.*"

    for module_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(module_toModify, module_name):
            for children_name, children in dict(module.named_children()).items():
                if re.fullmatch(children_toModify, children_name):
                    assert isinstance(children, nn.Linear)
                    setattr(module, children_name, IA3LinearLayer(children))

    return transformer, trainableParameter_regex

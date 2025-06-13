from torch import dtype

from config import PretrainedConfig
from utils import *


class GPTPreTrainedModel(nn.Module):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = config.name_or_path

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> dtype:
        return get_parameter_dtype(self)

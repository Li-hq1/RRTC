"""Query/value low-rank adapters for the frozen ViT backbone."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MergedQKV(nn.Linear):
    """Q/V-only LoRA. Initialization and dense delta follow the research code.

    Adapted from Microsoft LoRA (MIT), Copyright Microsoft Corporation.
    No eval-time weight merging: adaptation runs with backbone dropout disabled.
    """

    def __init__(self, dim, rank=2, alpha=1.0):
        # nn.Linear calls reset_parameters once before the factors exist.
        super().__init__(dim, 3 * dim)
        self.r = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(self.weight.new_zeros((2 * rank, dim)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((2 * dim, rank)))
        # The historical constructor resets the base layer a second time.
        self.reset_parameters()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        index = torch.zeros(3, dim, dtype=torch.bool)
        index[[0, 2]] = True
        self.register_buffer("lora_ind", index.flatten(), persistent=False)

    def delta(self):
        compact = F.conv1d(self.lora_A.unsqueeze(0), self.lora_B.unsqueeze(-1), groups=2).squeeze(0)
        result = self.weight.new_zeros(self.weight.shape)
        result[self.lora_ind] = compact
        return result

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        result += x @ self.delta().T * self.scaling
        return result

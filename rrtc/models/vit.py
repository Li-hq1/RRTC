"""ViT feature statistics and masked prediction views."""

import torch
from torch import nn

from .lora import MergedQKV


class FeatureViews(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        mask1, mask2 = torch.ones_like(x), torch.ones_like(x)
        mask1[torch.rand_like(x) < self.ratio] = 0
        # Draw even in shared mode to preserve the historical RNG stream.
        mask2[torch.rand_like(x) < self.ratio] = 0
        return torch.cat((x, x * mask1, x * mask1), dim=0)


class FeatureModel(nn.Module):
    def __init__(self, backbone, config, adapters=True):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.records = []
        self.capture_input = False
        self.raw_callback = None
        if adapters:
            if not 1 <= config.blocks <= len(backbone.blocks):
                raise ValueError("blocks outside backbone depth")
            for block in backbone.blocks[: config.blocks]:
                old = block.attn.qkv
                layer = MergedQKV(old.in_features, config.rank, config.alpha).to(old.weight.device)
                layer.load_state_dict(old.state_dict(), strict=False)
                block.attn.qkv = layer
        self.views = FeatureViews(config.mask_ratio)
        backbone.head = nn.Sequential(self.views, backbone.head)
        # Monitor norm2 input and block output: 24 sites for ViT-Base.
        # The first norm1 input is omitted; the last block end feeds final norm.
        self.handles = [backbone.pos_drop.register_forward_hook(self._input_hook)]
        for block in backbone.blocks:
            self.handles.append(block.norm2.register_forward_pre_hook(self._mid_hook))
            self.handles.append(block.register_forward_hook(self._end_hook))
        backbone.eval()
        for name, param in backbone.named_parameters():
            param.requires_grad_("lora_" in name)

    def _record(self, x):
        if self.raw_callback is not None:
            self.raw_callback(x)
        mean = x.mean(dim=(0, 1), keepdim=True)
        moments = [mean, x.var(dim=(0, 1), keepdim=True, unbiased=False)]
        self.records.append(moments)

    def _input_hook(self, module, args, output):
        if self.capture_input:
            self._record(output)

    def _mid_hook(self, module, args):
        self._record(args[0])

    def _end_hook(self, module, args, output):
        self._record(output)

    def forward(self, x):
        self.records = []
        return self.backbone(x), self.records

    def adapters(self):
        return [b.attn.qkv for b in self.backbone.blocks if isinstance(b.attn.qkv, MergedQKV)]

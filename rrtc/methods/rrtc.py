import torch
from torch import nn

from .losses import alignment_loss, consistency_loss, correlation_loss
from .optim import make_optimizer


class RRTC(nn.Module):
    """One unlabeled update per batch; return predictions from before the update."""

    def __init__(self, model, statistics, config):
        super().__init__()
        self.model, self.config = model, config
        # Slot zero stores input features; alignment uses the two sites per block.
        if len(statistics) != 2 * len(model.backbone.blocks) + 1:
            raise ValueError("Expected 25 source sites for ViT-B (input + two per block)")
        self.statistics = []
        device = next(model.parameters()).device
        for site in statistics[1:]:
            if len(site) < 2:
                raise ValueError("Insufficient source moments")
            if (
                site[0].numel() != model.backbone.embed_dim
                or site[1].numel() != model.backbone.embed_dim
            ):
                raise ValueError("Expected channel-wise source statistics")
            values = [t.to(device) for t in site[:2]]
            if any(not torch.isfinite(t).all() for t in values) or (values[1] <= 0).any():
                raise ValueError("Non-finite moments or non-positive source variance")
            self.statistics.append(values)
        self.optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], config)
        self.last_losses = {}

    @torch.enable_grad()
    def forward(self, x):
        logits, records = self.model(x)
        align = alignment_loss(records, self.statistics)
        reg = correlation_loss(self.model.adapters(), self.config)
        mc = consistency_loss(logits, len(x))
        loss = align + self.config.reg_weight * reg + self.config.mc_weight * mc
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite adaptation loss")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.last_losses = {
            "alignment": float(align.detach()),
            "constraint": float(reg.detach()),
            "consistency": float(mc.detach()),
            "total": float(loss.detach()),
        }
        return logits[: len(x)].detach()

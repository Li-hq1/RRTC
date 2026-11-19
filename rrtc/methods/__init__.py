"""Build RRTC with a frozen source model and its paired feature statistics."""

import torch

from rrtc.config import asset_path
from rrtc.models.vit import FeatureModel

from .rrtc import RRTC


def build_method(backbone, experiment):
    statistics = torch.load(
        asset_path(experiment.model.statistics), map_location="cpu", weights_only=True
    )
    return RRTC(
        FeatureModel(backbone, experiment.method),
        statistics["cmd_base_mid"],
        experiment.method,
    )

from rrtc.config import DATASETS, asset_path

from .cifar import CIFARC
from .imagenet import ImageNetC


def dataset_root(config):
    root = asset_path(config.root)
    dirname = DATASETS[config.dataset][0]
    return root if root.name == dirname else root / dirname


def make_dataset(config, domain):
    if config.dataset == "imagenetc":
        return ImageNetC(
            dataset_root(config),
            domain,
            config.severity,
            config.num_examples,
            config.image_size,
            config.resize_size,
        )
    return CIFARC(config, domain)

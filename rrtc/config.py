"""Dataset presets and paths shared by evaluation and source collection."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs"
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
DATASETS = {
    "imagenetc": ("ImageNet-C", 1000),
    "cifar10c": ("CIFAR-10-C", 10),
    "cifar100c": ("CIFAR-100-C", 100),
}


@dataclass
class MethodConfig:
    rank: int = 2
    blocks: int = 6
    alpha: float = 1.0
    mask_ratio: float = 0.3
    gram_target: float = 0.5
    mc_weight: float = 1.0
    reg_weight: float = 10.0
    lr: float = 0.01
    seed: int = 1
    batch_size: int = 64
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0


@dataclass
class DataConfig:
    dataset: str = "imagenetc"
    root: str = "data"
    domains: list = field(default_factory=lambda: list(CORRUPTIONS))
    severity: int = 5
    num_examples: int = 5000
    image_size: int = 224
    resize_size: int = 256


@dataclass
class ModelConfig:
    name: str = "vit_base_patch16_224"
    num_classes: int = 1000
    checkpoint: str = "assets/imagenetc/source_checkpoint.npz"
    statistics: str = "assets/imagenetc/source_statistics.pt"


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    workers: int = 4
    threads: int = 4
    cudnn_tf32: bool = True


@dataclass
class Config:
    method: MethodConfig
    data: DataConfig
    model: ModelConfig
    runtime: RuntimeConfig

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = json.loads(Path(path).expanduser().read_text())
    config = Config(
        MethodConfig(**raw["method"]),
        DataConfig(**raw["data"]),
        ModelConfig(**raw["model"]),
        RuntimeConfig(**raw["runtime"]),
    )
    if config.data.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    if config.model.num_classes != DATASETS[config.data.dataset][1]:
        raise ValueError("The classifier size must match the dataset")
    if config.data.severity not in range(1, 6):
        raise ValueError("Corruption severity must be between 1 and 5")
    return config


def asset_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

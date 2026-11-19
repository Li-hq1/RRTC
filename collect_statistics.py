"""Collect source feature means and variances from clean training images."""

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from rrtc.config import asset_path, load_config
from rrtc.datasets.transforms import cifar_tensor, image_transform
from rrtc.models.factory import create_backbone
from rrtc.models.vit import FeatureModel
from rrtc.utils import seed_all


def merge_moments(previous, x):
    """Pool channel moments over images and tokens using a float64 accumulator."""
    x = x.detach().double().flatten(0, 1)
    count, mean = x.shape[0], x.mean(0)
    m2 = (x - mean).square().sum(0)
    if previous is None:
        return count, mean, m2
    n0, mean0, m20 = previous
    delta = mean - mean0
    n = n0 + count
    return n, mean0 + delta * (count / n), m20 + m2 + delta.square() * (n0 * count / n)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", required=True, help="Clean training data directory")
    parser.add_argument("--checkpoint", help="Overrides the source model path in the config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=12800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.num_images, args.batch_size) < 1 or args.workers < 0:
        parser.error("Image and batch counts must be positive; workers must be nonnegative")
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(args.config)
    checkpoint = asset_path(args.checkpoint or config.model.checkpoint)
    torch.set_num_threads(config.runtime.threads)
    seed_all(args.seed, cudnn_tf32=config.runtime.cudnn_tf32)
    data = config.data
    source_root = Path(args.source_root).expanduser()
    if data.dataset == "imagenetc":
        dataset = ImageFolder(
            source_root, transform=image_transform(data.image_size, data.resize_size)
        )
    else:
        cls = CIFAR10 if data.dataset == "cifar10c" else CIFAR100
        dataset = cls(
            source_root,
            train=True,
            download=False,
            transform=partial(cifar_tensor, image_size=data.image_size),
        )
    if args.num_images > len(dataset):
        raise ValueError(f"Requested {args.num_images} images from a dataset of {len(dataset)}")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
        : args.num_images
    ].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = FeatureModel(
        create_backbone(checkpoint, args.device, config.model.name, config.model.num_classes),
        config.method,
        adapters=False,
    )
    model.views.enabled = False
    model.capture_input = True
    records = [None] * 25
    site = 0

    def consume(x):
        nonlocal site
        records[site] = merge_moments(records[site], x)
        site += 1

    model.raw_callback = consume
    with torch.no_grad():
        for batch, (images, _) in enumerate(loader):
            site = 0
            model(images.to(args.device))
            if site != 25:
                raise RuntimeError(f"Expected 25 feature sites, got {site}")
            if batch % 20 == 0:
                print(
                    f"Collected {min((batch + 1) * args.batch_size, args.num_images)} images",
                    flush=True,
                )
    statistics = [
        [mean.float().view(1, 1, -1).cpu(), (m2 / n).float().view(1, 1, -1).cpu()]
        for n, mean, m2 in records
    ]
    if any(not torch.isfinite(t).all() for moments in statistics for t in moments) or any(
        (moments[1] <= 0).any() for moments in statistics
    ):
        raise ValueError("Non-finite moments or non-positive variance; use more source images")
    metadata = {
        "dataset": data.dataset,
        "num_images": args.num_images,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": data.image_size,
        "resize_size": data.resize_size,
        "variance": "population variance over all images and tokens",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cmd_base_mid": statistics, "metadata": metadata}, output)
    print(f"Saved source statistics to {output}")
    return metadata


if __name__ == "__main__":
    main()

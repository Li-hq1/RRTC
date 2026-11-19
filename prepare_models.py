"""Prepare downloaded source weights for the dataset presets."""

import argparse
import shutil
from pathlib import Path

import torch

from rrtc.config import DATASETS, PRESETS, asset_path, load_config
from rrtc.models.factory import checkpoint_state, create_backbone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Downloaded source weights")
    parser.add_argument(
        "--output", type=Path, help="Defaults to the path in configs/<dataset>.json"
    )
    args = parser.parse_args(argv)
    config = load_config(PRESETS / f"{args.dataset}.json")
    source = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output or asset_path(config.model.checkpoint)
    output = output.expanduser().resolve()
    suffix = ".npz" if args.dataset == "imagenetc" else ".pt"
    if output.suffix != suffix:
        parser.error(f"The prepared checkpoint must use {suffix}")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        raise FileExistsError(partial)
    torch.set_num_threads(4)
    try:
        if args.dataset == "imagenetc":
            if source.suffix != ".npz":
                raise ValueError("Expected the ImageNet AugReg ViT-B/16-224 NPZ weights")
            shutil.copyfile(source, partial)
            load_path = source
        else:
            torch.save(checkpoint_state(source), partial)
            load_path = partial
        # Strictly check the full model, including the dataset-specific classifier.
        create_backbone(load_path, "cpu", config.model.name, config.model.num_classes)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"Saved source checkpoint to {output}")
    return output


if __name__ == "__main__":
    main()

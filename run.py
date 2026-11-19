"""Evaluate RRTC on a continual stream of corruptions."""

import argparse
from datetime import datetime
from pathlib import Path

from rrtc.config import PRESETS, load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PRESETS / "imagenetc.json")
    parser.add_argument("--data-root", help="Benchmark directory or its parent")
    parser.add_argument("--checkpoint", help="Source model weights")
    parser.add_argument("--statistics", help="Source feature statistics")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path, help="Results directory")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for obj, name, value in (
        (config.data, "root", args.data_root),
        (config.model, "checkpoint", args.checkpoint),
        (config.model, "statistics", args.statistics),
        (config.runtime, "device", args.device),
        (config.runtime, "workers", args.workers),
        (config.method, "seed", args.seed),
    ):
        if value is not None:
            setattr(obj, name, value)
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    from rrtc.evaluate import evaluate

    output = args.output or Path("results") / (
        f"{config.data.dataset}_s{config.method.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    return evaluate(config, output)


if __name__ == "__main__":
    main()

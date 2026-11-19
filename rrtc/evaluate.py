"""Continual evaluation with predictions scored before each adaptation step."""

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rrtc.config import asset_path
from rrtc.datasets import make_dataset
from rrtc.methods import build_method
from rrtc.models.factory import create_backbone
from rrtc.utils import seed_all


def evaluate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    torch.set_num_threads(config.runtime.threads)
    seed_all(config.method.seed, cudnn_tf32=config.runtime.cudnn_tf32)
    device = torch.device(config.runtime.device)
    backbone = create_backbone(
        asset_path(config.model.checkpoint),
        device,
        config.model.name,
        config.model.num_classes,
    )
    model = build_method(backbone, config)
    errors = []
    # Build once: adapters and optimizer state carry over between corruptions.
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["corruption", "num_images", "error_percent"])
        for domain in config.data.domains:
            dataset = make_dataset(config.data, domain)
            loader = DataLoader(
                dataset,
                batch_size=config.method.batch_size,
                shuffle=False,
                num_workers=config.runtime.workers,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(config.method.seed),
            )
            correct = total = 0
            for step, (images, labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite predictions")
                correct += int((logits.argmax(1) == labels).sum())
                total += len(images)
                if step % 20 == 0:
                    print(
                        f"{domain}: {total}/{len(dataset)} images, "
                        f"error {100 * (1 - correct / total):.2f}%",
                        flush=True,
                    )
            error = 100 * (1 - correct / total)
            errors.append(error)
            writer.writerow([domain, total, error])
            stream.flush()
            print(f"{domain}: {error:.2f}%", flush=True)
        mean_error = sum(errors) / len(errors)
        writer.writerow(["mean", "", mean_error])
    print(f"Mean error: {mean_error:.2f}% | Results: {output}", flush=True)
    return {"mean_error": mean_error, "completed_domains": len(errors)}

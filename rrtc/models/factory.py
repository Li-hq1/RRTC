from pathlib import Path

import timm
import torch


def create_backbone(checkpoint, device, model_name="vit_base_patch16_224", num_classes=1000):
    checkpoint = Path(checkpoint).expanduser()
    if timm.__version__ != "0.4.12":
        raise RuntimeError(f"Reproduction requires timm 0.4.12, got {timm.__version__}")
    if num_classes != 1000 and checkpoint.suffix == ".npz":
        raise ValueError(
            "CIFAR requires a fine-tuned state_dict with a trained 10/100-class head, not an ImageNet NPZ"
        )
    checkpoint = Path(checkpoint).resolve(strict=True)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if checkpoint.suffix == ".npz":
        model.load_pretrained(str(checkpoint))
    else:
        state = checkpoint_state(checkpoint)
        # A missing or mismatched classifier must never be silently ignored.
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def checkpoint_state(checkpoint):
    """Load a tensor-only source checkpoint and remove a DataParallel prefix."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict or a model/state_dict container")
    state = state.get("state_dict", state.get("model", state))
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty source state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Source parameters must be finite tensors")
    return state

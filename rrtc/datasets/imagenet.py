import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import image_transform

METADATA = Path(__file__).resolve().parent / "metadata"


class ImageNetC(Dataset):
    def __init__(
        self,
        root,
        domain,
        severity=5,
        n_examples=5000,
        image_size=224,
        resize_size=256,
    ):
        root = Path(root)
        if root.name != "ImageNet-C":
            root = root / "ImageNet-C"
        self.transform = image_transform(image_size, resize_size)
        # RobustBench's fixed 5,000-image subset and ImageNet class indices.
        names = (METADATA / "imagenet_test_image_ids.txt").read_text().splitlines()
        mapping = json.loads((METADATA / "imagenet_class_to_id_map.json").read_text())
        if not 0 < n_examples <= len(names):
            raise ValueError(f"n_examples must be 1..{len(names)}")
        self.samples = [
            (root / domain / str(severity) / n, mapping[n.split("/")[0]])
            for n in names[:n_examples]
        ]
        missing = next((str(p) for p, _ in self.samples if not p.is_file()), None)
        if missing:
            raise FileNotFoundError(missing)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label

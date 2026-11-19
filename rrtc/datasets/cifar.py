import numpy as np
from torch.utils.data import Dataset

from rrtc.config import DATASETS

from .transforms import cifar_tensor


class CIFARC(Dataset):
    def __init__(self, config, domain):
        self.config = config
        from . import dataset_root

        root = dataset_root(config)
        self.images = np.load(root / (domain + ".npy"), mmap_mode="r", allow_pickle=False)
        self.labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
        if self.images.shape != (50000, 32, 32, 3) or self.images.dtype != np.uint8:
            raise ValueError("Expected CIFAR-C uint8 array [50000,32,32,3]")
        if self.labels.shape not in ((10000,), (50000,)):
            raise ValueError("Expected CIFAR-C labels [10000] or [50000]")
        self.start = (config.severity - 1) * 10000

    def __len__(self):
        return self.config.num_examples

    def __getitem__(self, index):
        # Copy read-only mmap views before conversion to torch tensors.
        arr = np.array(self.images[self.start + index], copy=True)
        c = self.config
        x = cifar_tensor(arr, c.image_size)
        label_index = index if len(self.labels) == 10000 else self.start + index
        y = int(self.labels[label_index])
        if not 0 <= y < DATASETS[c.dataset][1]:
            raise ValueError("CIFAR label outside configured class range")
        return x, y

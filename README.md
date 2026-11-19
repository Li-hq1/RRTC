# RRTC

Code for **Regress Rather Than Chase: Anchored Domain Regression for Efficient Continual Test-Time Adaptation**.

RRTC adapts a frozen ViT with source feature statistics and constrained low-rank adapters. This repository provides the continual adaptation experiments on ImageNet-C, CIFAR-10-C, and CIFAR-100-C.

## Setup

Install Python 3.10 or later and the dependencies below. We use PyTorch 2.7.1, torchvision 0.22.1, and timm 0.4.12. Choose a PyTorch CUDA build suitable for your GPU.

```bash
pip install -r requirements.txt
```

Run the commands below from the repository root.

## Data and pretrained models

Download [ImageNet-C and CIFAR-C](https://github.com/hendrycks/robustness) and place them under `data/`, or pass their location with `--data-root`:

```text
data/
  ImageNet-C/<corruption>/<severity>/<class>/<image>.JPEG
  CIFAR-10-C/{labels.npy,gaussian_noise.npy,...}
  CIFAR-100-C/{labels.npy,gaussian_noise.npy,...}
```

Use the following ViT-B/16 source checkpoints. CIFAR models are fine-tuned at resolution 384 and released by [ViDA](https://github.com/Yangsenqiao/vida).

| Dataset | Source weights |
|---|---|
| ImageNet-C | [AugReg ViT-B/16-224](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) |
| CIFAR-10-C | [ViT-B/16-384, CIFAR-10](https://drive.google.com/file/d/1pAoz4Wwos74DjWPQ5d-6ntyjQkmp9FPE/view) |
| CIFAR-100-C | [ViT-B/16-384, CIFAR-100](https://drive.google.com/file/d/1yRekkpkIdwX_LFsOh4Ba9ndaECnY-UC-/view) |

After downloading, prepare the weights with:

```bash
python prepare_models.py --dataset imagenetc --checkpoint /path/to/imagenet_weights.npz
python prepare_models.py --dataset cifar10c --checkpoint /path/to/vit_base_384_cifar10.t7
python prepare_models.py --dataset cifar100c --checkpoint /path/to/pretrain_cifar100.t7
```

The preparation script converts or copies the downloaded weights and checks the classifier; no training is needed. It creates `source_checkpoint.npz` for ImageNet and `source_checkpoint.pt` for CIFAR. The three small `source_statistics.pt` files are already included in this repository, so they are available after cloning:

```text
assets/
  imagenetc/{source_checkpoint.npz,source_statistics.pt}
  cifar10c/{source_checkpoint.pt,source_statistics.pt}
  cifar100c/{source_checkpoint.pt,source_statistics.pt}
```

With the data and checkpoints in place, proceed to evaluation. Recollecting source statistics is optional. ImageNet evaluation uses the RobustBench 5,000-image list and class mapping in `rrtc/datasets/metadata/` to fix the evaluation subset and label order.

## Evaluation

```bash
python run.py --config configs/imagenetc.json
python run.py --config configs/cifar10c.json
python run.py --config configs/cifar100c.json
```

To use another data location or GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/cifar100c.json \
  --data-root /path/to/CIFAR-100-C --output results/cifar100c_seed1
```

The configurations use severity 5, 15 corruptions in the standard order, batch size 64, and seed 1. The model is updated once per batch and carries its state across corruptions. Predictions are scored before each update. ImageNet-C uses 5,000 images per corruption; CIFAR-C uses 10,000. Images are scaled to [0, 1] without mean/std normalization; ImageNet uses resize to 256 and center crop to 224, and CIFAR uses tensor bilinear resize to 384.

Dataset-specific settings are in `configs/`. Common overrides are `--seed`, `--device`, `--workers`, `--checkpoint`, and `--statistics`. Each run saves its configuration and per-corruption error rates to a new directory under `results/`. The last CSV row is the mean corruption error. Relative data and model paths are resolved from the repository root.

## Source statistics

To collect a new source anchor from clean training images:

```bash
python collect_statistics.py --config configs/cifar10c.json \
  --source-root /path/to/cifar10-clean --num-images 12800 \
  --output assets/cifar10c/source_statistics_new.pt
```

For CIFAR, the source root should contain the extracted `cifar-10-batches-py/` or `cifar-100-python/` directory. For ImageNet, use `configs/imagenetc.json` and a training directory organized as `<class>/<image>`. Pass the new file to evaluation with `--statistics`. Use the same source checkpoint for collection and adaptation.

The included CIFAR statistics use 12,800 training images sampled with seed 1. The included ImageNet anchor is the one used in our experiments; its original collection subset was not fully recorded, so recollecting it may give different results.

## Acknowledgements

We build on [ViDA](https://github.com/Yangsenqiao/vida), [timm](https://github.com/huggingface/pytorch-image-models), [LoRA](https://github.com/microsoft/LoRA), and [RobustBench](https://github.com/RobustBench/robustbench). Code is released under the MIT license.

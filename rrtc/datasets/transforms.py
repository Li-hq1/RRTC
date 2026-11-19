from torch.nn import functional as F
from torchvision import transforms


def image_transform(image_size=224, resize_size=256):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def cifar_tensor(image, image_size=384):
    # Resize the float tensor; PIL resizing gives different interpolation results.
    x = transforms.functional.to_tensor(image)
    return F.interpolate(
        x.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )[0]

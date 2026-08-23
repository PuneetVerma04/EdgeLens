import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_casting_loaders(dataset_path: str, input_size: int, batch_size: int,
                         val_split: float, num_workers: int):
    """
    Loads the casting dataset from a flat ImageFolder directory and splits 80/20.
    Returns (train_loader, val_loader, class_names).
    """
    full = datasets.ImageFolder(dataset_path, transform=get_train_transforms(input_size))
    train_size = int((1 - val_split) * len(full))
    val_size = len(full) - train_size
    train_ds, val_ds = random_split(full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # Apply val transforms to val subset without mutating the base dataset
    val_ds.dataset = datasets.ImageFolder(dataset_path,
                                          transform=get_val_transforms(input_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, full.classes


def load_pcb_loaders(crops_path: str, input_size: int, batch_size: int,
                     num_workers: int):
    """
    Loads pre-extracted PCB crop dataset from crops_path/{train,val} directories.
    Run scripts/prepare_pcb_crops.py first to generate the crop directories.
    Returns (train_loader, val_loader, class_names).
    """
    train_path = os.path.join(crops_path, "train")
    val_path   = os.path.join(crops_path, "val")

    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise FileNotFoundError(
            f"Crop directories not found at {crops_path}. "
            "Run scripts/prepare_pcb_crops.py first."
        )

    train_ds = datasets.ImageFolder(train_path, transform=get_train_transforms(input_size))
    val_ds   = datasets.ImageFolder(val_path,   transform=get_val_transforms(input_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

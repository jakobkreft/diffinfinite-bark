"""Flat external train/test split dataloader for DiffInfinite-Bark.

The internal per-class split logic from upstream DiffInfinite has been removed
(it overlapped train and test for multi-label images, leaving almost no clean
held-out stems for FID). Train and test now come from two predetermined
folders, each containing `*.jpg` image + `*_mask.png` mask pairs.

Each `__getitem__` returns `(image, mask)` where `image` is a [0,1] float
tensor of shape (3, H, W) and `mask` is an int tensor of shape (1, H, W)
with pixel values in {0, 1, 2} (bark / knot / defect). The class label used
for CFG and embedding is derived downstream in `dm.py` via `mask.max()`.
"""
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


def set_global_seed(seed: int):
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


set_global_seed(10)


class ComposeState(T.Compose):
    """Compose that re-seeds RNG so the same random augmentation applies to
    every element in a (image, mask) tuple."""

    def __init__(self, transforms):
        self.transforms = list(transforms)
        self.seed = None
        self.retain_state = False

    def __call__(self, x):
        if self.seed is not None:
            set_global_seed(self.seed)
        if self.retain_state:
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            self.seed = None

        if isinstance(x, (list, tuple)):
            return self.apply_sequence(x)
        return self.apply_img(x)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        self.retain_state = True
        seq = list(map(self, seq))
        self.retain_state = False
        return seq


def identity(x):
    return x


def get_augmentation(name='identity'):
    if name == 'identity':
        return identity
    raise ValueError(f'unknown augmentation: {name}')


class RandomRotate90:
    """0/90/180/270-degree rotation (not the same as T.RandomRotation(90))."""

    def __call__(self, x):
        return x.rot90(random.randint(0, 3), dims=(-1, -2))

    def __repr__(self):
        return self.__class__.__name__


class BarkPairsDataset(Dataset):
    """Flat folder of (image, mask) pairs.

    Args:
        folder: directory containing `<stem>.jpg` and `<stem>_mask.png` pairs.
            Images can be any resolution — the spatial pipeline (e.g.
            T.RandomCrop(512)) handles the size.
        transform: a ComposeState applied to the (image, mask) tuple. The
            ComposeState mechanism re-seeds the RNG so the same random
            spatial draws apply to both image and mask.
        image_only_transform: applied to the image ONLY, after `transform`.
            Use for color jitter, brightness, etc. — anything that would
            destroy the integer-class semantics of the mask.
    """

    def __init__(self, folder: str, transform=None, image_only_transform=None):
        self.folder = Path(folder)
        assert self.folder.is_dir(), f'not a directory: {self.folder}'

        jpgs = sorted(self.folder.glob('*.jpg'))
        assert len(jpgs) > 0, f'no *.jpg files in {self.folder}'

        self.stems = []
        for jpg in jpgs:
            mask_path = self.folder / f'{jpg.stem}_mask.png'
            assert mask_path.exists(), f'missing mask for {jpg.name}'
            self.stems.append(jpg.stem)

        self.transform = transform
        self.image_only_transform = image_only_transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = Image.open(self.folder / f'{stem}.jpg').convert('RGB')
        mask = Image.open(self.folder / f'{stem}_mask.png')

        if self.transform is not None:
            img, mask = self.transform((img, mask))

        # Image-only transforms (e.g. color jitter) — applied after the
        # spatial pipeline so the mask is already finalized.
        if self.image_only_transform is not None:
            img = self.image_only_transform(img)

        # Mask: ensure (1, H, W) int tensor. ComposeState already calls
        # T.ToTensor() on it which produces a float tensor of shape (1, H, W)
        # with values in [0,1] — multiply back to integer class indices.
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dtype != torch.int32 and mask.dtype != torch.int64:
            # T.ToTensor() rescales by 1/255 for 8-bit input; un-rescale.
            mask = (mask * 255).round().int()

        return img, mask


def import_external_split(
    train_folder: str,
    test_folder: str = None,
    batch_size: int = 4,
    num_workers: int = 0,
    train_transform=None,
    eval_transform=None,
    image_only_transform=None,
):
    """Return (train_loader, test_loader) backed by on-disk folders.

    No splitting is performed. If `test_folder` is None, the test loader is
    constructed from `train_folder` (used only for periodic visualization in
    Trainer.eval_loop — the model never trains on it differently than the
    rest of the training data).

    `train_transform` runs through every sample served by the train loader
    (with `image_only_transform` chained after on the image only).
    `eval_transform` runs through samples served by the test loader, with
    NO color/image-only augmentations, so visualization snapshots are
    deterministic and clean.

    Falls back to `train_transform` for the test loader if `eval_transform`
    is not provided.
    """
    train_ds = BarkPairsDataset(
        folder=train_folder,
        transform=train_transform,
        image_only_transform=image_only_transform,
    )

    eval_folder = test_folder if test_folder is not None else train_folder
    test_ds = BarkPairsDataset(
        folder=eval_folder,
        transform=eval_transform if eval_transform is not None else train_transform,
        image_only_transform=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader

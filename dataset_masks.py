import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


def import_small_mask_dataset(
    data_path: str = 'D:/jk/diffinfinite_512',
    num_labels: int = 3,
    use_split: bool = True,
    size: int = 128,
    transform=T.PILToTensor(),
    batch_size: int = 32,
    num_workers: int = 0,
    small=True,
):
    trainset = SimpleMaskDataset(
        data_path=data_path,
        num_labels=num_labels,
        size=size,
        mode='train',
        train_split=0.9,
        transform=transform,
    )
    testset = SimpleMaskDataset(
        data_path=data_path,
        num_labels=num_labels,
        size=size,
        mode='test',
        train_split=0.9,
        transform=transform,
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, test_loader


class SimpleMaskDataset(Dataset):
    """Simple mask dataset that loads *_mask.png files directly.

    Each mask is a single-channel image with integer pixel values 0..num_labels-1.
    The label for each mask is the maximum pixel value present (the dominant non-background class).
    """

    def __init__(self,
                 data_path: str = 'D:/jk/diffinfinite_512',
                 num_labels: int = 3,
                 size: int = 128,
                 mode: str = 'train',
                 train_split: float = 0.9,
                 transform=T.PILToTensor()):

        self.data_path = data_path
        self.num_labels = num_labels
        self.size = size
        self.transform = transform

        # Find all mask files
        all_masks = sorted(glob.glob(os.path.join(data_path, '*_mask.png')))
        assert len(all_masks) > 0, f"No *_mask.png files found in {data_path}"

        # Determine label for each mask (max pixel value = dominant class)
        self.labels = []
        for m_path in all_masks:
            mask = np.array(Image.open(m_path))
            self.labels.append(int(mask.max()))

        # Train/test split
        if len(all_masks) > 1:
            train_masks, test_masks, train_labels, test_labels = train_test_split(
                all_masks, self.labels, train_size=train_split, random_state=42)
        else:
            train_masks, test_masks = all_masks, all_masks
            train_labels, test_labels = self.labels, self.labels

        if mode == 'train':
            self.masks = train_masks
            self.labels = train_labels
        else:
            self.masks = test_masks
            self.labels = test_labels

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = Image.open(self.masks[idx]).resize((self.size, self.size), Image.NEAREST)
        label = self.labels[idx]

        if self.transform is not None:
            mask = self.transform(mask)
        return mask, label

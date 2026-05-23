import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True):
        if train:
            images_path = data_dir / "train_images.npy"
            labels_path = data_dir / "train_labels.npy"
        else:
            images_path = data_dir / "test_images.npy"
            labels_path = data_dir / "test_labels.npy"

        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]

        if img.ndim == 1:
            img = img.reshape(28, 28)

        img = torch.tensor(img).unsqueeze(0).float() / 255.0
        label = int(self.labels[idx])

        return img, label

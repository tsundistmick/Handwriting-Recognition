import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from pathlib import Path

class MNISTDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True):
        data_dir = Path(data_dir)
        base = data_dir

        if train:
            images_path = base / "train_images.npy"
            labels_path = base / "train_labels.npy"
        else:
            images_path = base / "test_images.npy"
            labels_path = base / "test_labels.npy"

        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"MNIST .npy files not found. Expected:\n"
                f"- {images_path}\n"
                f"- {labels_path}\n"
                f"Pass data_dir as 'data/processed' (folder containing these files)."
            )

        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
    
    def __len__(self):
        return int(self.labels.shape[0])
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if isinstance(label, np.ndarray) and label.ndim > 0:
            label = int(np.argmax(label))
        else:
            label = int(label)

        if isinstance(img, np.ndarray):
            if img.ndim == 1 and img.size == 28 * 28:
                img = img.reshape(28, 28)
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img[0]

        img = torch.as_tensor(img, dtype=torch.float32)

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] != 1:
            if img.shape[-1] == 1:
                img = img.permute(2, 0, 1)

        if torch.is_floating_point(img) and img.numel() > 0 and float(img.max()) > 1.0:
            img = img / 255.0

        return img, label
class MINISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=3136, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10),
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.classifier(x)
            return x
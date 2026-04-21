import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path

class MNISTDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True):
        self.data = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=False
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = torch.tensor(list(img.getdata())).reshape(1, 28, 28).float() / 255.0
        return img, label
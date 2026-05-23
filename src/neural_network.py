import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import MNISTDataset


class MNISTModel(nn.Module):
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


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if (batch_idx + 1) % 100 == 0:
                current_acc = correct / total * 100
                print(
                    f"  Эпоха {epoch} | Батч {batch_idx + 1}/{len(self.train_loader)} "
                    f"| Loss: {total_loss / (batch_idx + 1):.4f} | Acc: {current_acc:.2f}%"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total * 100
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> tuple[float, float]:
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        return avg_loss, accuracy

    def fit(self, num_epochs: int, save_path: Path) -> None:
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"ЭПОХА {epoch}/{num_epochs}")
            print(f"{'=' * 60}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            print(f"\n  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            # validate() без аргумента использует val_loader
            val_loss, val_acc = self.validate()
            print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            self.scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Новый лучший результат! Модель сохранена → {save_path}")

        print(f"\nОбучение завершено. Лучшая val accuracy: {best_val_acc:.2f}%")


def main():
    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    DATA_DIR = Path("../data/processed")
    SAVE_PATH = Path("./best_model.pth")
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    full_train = MNISTDataset(DATA_DIR, train=True)
    test_dataset = MNISTDataset(DATA_DIR, train=False)

    train_dataset, val_dataset = random_split(
        full_train,
        lengths=[50_000, 10_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = MNISTModel()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров модели: {num_params:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    trainer.fit(num_epochs=NUM_EPOCHS, save_path=SAVE_PATH)
    print("\nЗагружаем лучшую модель для финального теста...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Финальная точность на тестовой выборке: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

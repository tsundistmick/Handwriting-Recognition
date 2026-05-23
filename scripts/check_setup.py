from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
IAM_ROOT = REPO_ROOT / "iam"
MNIST_DIR = REPO_ROOT / "data" / "processed"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _step(name: str, fn) -> bool:
    t0 = time.time()
    try:
        fn()
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        return False
    print(f"OK    {name} ({time.time() - t0:.2f}s)")
    return True


def check_imports() -> None:
    import torch
    import numpy
    import torchvision
    from PIL import Image

    _ = (torch, numpy, torchvision, Image)


def check_mnist() -> None:
    import torch
    from torch.utils.data import random_split

    from dataset import MNISTDataset
    from neural_network import MNISTModel

    required = [
        MNIST_DIR / "train_images.npy",
        MNIST_DIR / "train_labels.npy",
        MNIST_DIR / "test_images.npy",
        MNIST_DIR / "test_labels.npy",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "нет MNIST в data/processed: " + ", ".join(p.name for p in missing)
        )

    ds = MNISTDataset(MNIST_DIR, train=True)
    img, label = ds[0]
    train_part, val_part = random_split(
        ds,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(42),
    )
    out = MNISTModel()(torch.randn(2, 1, 28, 28))

    if len(ds) != 60_000:
        raise ValueError(f"ожидалось 60000 train, получено {len(ds)}")
    if tuple(img.shape) != (1, 28, 28) or not (0 <= label <= 9):
        raise ValueError(f"неверный сэмпл: shape={tuple(img.shape)}, label={label}")
    if len(train_part) != 50_000 or len(val_part) != 10_000:
        raise ValueError("random_split train/val")
    if tuple(out.shape) != (2, 10):
        raise ValueError(f"неверный выход модели: {tuple(out.shape)}")

    print(f"      MNIST: {len(ds)} train, split {len(train_part)}/{len(val_part)}")


def check_iam_files() -> None:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from prepare_iam_data import verify

    code = verify(IAM_ROOT)
    if code != 0:
        raise RuntimeError(
            "IAM на диске неполный — см. scripts/README.md "
            "(prepare_iam_data.py extract --lines ... --ascii ... --splits ...)"
        )


def check_iam_splits() -> None:
    from text import load_split_ids

    train_ids = load_split_ids(IAM_ROOT, "train")
    val_ids = load_split_ids(IAM_ROOT, "val")
    test_ids = load_split_ids(IAM_ROOT, "test")

    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            f"один из сплитов пуст: train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}"
        )

    overlap_tv = set(train_ids) & set(val_ids)
    overlap_tt = set(train_ids) & set(test_ids)
    if overlap_tv or overlap_tt:
        raise ValueError(
            f"сплиты пересекаются: train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}"
        )

    print(
        f"      IAM splits: train={len(train_ids)}, val={len(val_ids)}, "
        f"test={len(test_ids)}"
    )


def check_iam_loader() -> None:
    import torch
    from torch.utils.data import DataLoader

    from text import IAMDataset, collate_fn, load_split_ids, NUM_CLASSES

    val_ids = load_split_ids(IAM_ROOT, "val")
    ds = IAMDataset(IAM_ROOT, val_ids[:8], img_height=64, augment=False)
    if len(ds) == 0:
        raise ValueError("IAMDataset пуст на val[:8] — нет картинок строк")

    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    images, targets, target_lens, texts = next(iter(loader))

    if images.ndim != 4 or images.shape[1] != 1 or images.shape[2] != 64:
        raise ValueError(f"неверный батч картинок: {tuple(images.shape)}")
    if targets.dtype != torch.long or target_lens.dtype != torch.long:
        raise ValueError("targets/target_lens должны быть long")
    if int(target_lens.sum()) != int(targets.numel()):
        raise ValueError("сумма длин не совпадает с числом таргетов")
    if not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts должен быть списком строк")

    sample = texts[0][:60] + ("…" if len(texts[0]) > 60 else "")
    print(
        f"      IAM batch: images={tuple(images.shape)}, "
        f"targets={tuple(targets.shape)}, num_classes={NUM_CLASSES}, "
        f"sample={sample!r}"
    )


def main() -> int:
    print(f"Проверка репозитория: {REPO_ROOT}\n")

    steps = [
        ("зависимости (torch, numpy, torchvision, Pillow)", check_imports),
        ("MNIST + MNISTModel", check_mnist),
        ("IAM файлы на диске", check_iam_files),
        ("IAM splits (train/val/test из text.py)", check_iam_splits),
        ("IAMDataset + collate_fn (один батч)", check_iam_loader),
    ]

    failed = [name for name, fn in steps if not _step(name, fn)]

    print()
    if failed:
        print(f"Готово: {len(steps) - len(failed)}/{len(steps)} проверок прошло.")
        print("Не прошли:", ", ".join(failed))
        return 1

    print("Готово: все проверки прошли. Можно запускать обучение (python src/text.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

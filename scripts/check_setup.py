from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
IAM_ROOT = REPO_ROOT / "iam_data"
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
    from PIL import Image

    _ = (torch, numpy, Image)


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
            "IAM на диске: нет iam_data/iam_words/words.txt или words/*.png — "
            "см. scripts/README.md (prepare_iam_data.py extract)"
        )


def check_iam_split() -> None:
    from dataset import (
        IAM_SPLIT_SEED,
        IAM_VAL_RATIO,
        _filter_iam_entries_by_split,
        _iam_writer_id,
        _parse_iam_words_line,
    )

    words_txt = IAM_ROOT / "iam_words" / "words.txt"
    if not words_txt.is_file():
        raise FileNotFoundError(str(words_txt))

    entries: list[tuple[str, str]] = []
    with words_txt.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            parsed = _parse_iam_words_line(raw)
            if parsed is None:
                continue
            word_id, status, text = parsed
            if status != "ok":
                continue
            entries.append((word_id, text))

    if not entries:
        raise ValueError("в words.txt нет строк со статусом ok")

    train = _filter_iam_entries_by_split(
        entries, "train", IAM_VAL_RATIO, IAM_SPLIT_SEED
    )
    val = _filter_iam_entries_by_split(
        entries, "val", IAM_VAL_RATIO, IAM_SPLIT_SEED
    )
    train_writers = {_iam_writer_id(w) for w, _ in train}
    val_writers = {_iam_writer_id(w) for w, _ in val}

    if len(train) + len(val) != len(entries):
        raise ValueError("train + val != full")
    if train_writers & val_writers:
        raise ValueError("писатели пересекаются между train и val")

    ratio = len(val) / len(entries)
    print(
        f"      IAM split: train={len(train)}, val={len(val)}, "
        f"val_ratio={ratio:.1%}, writers={len(train_writers)}/{len(val_writers)}, "
        f"seed={IAM_SPLIT_SEED}"
    )


def check_iam_loader() -> None:
    from dataset import IAMDataset

    ds = IAMDataset(IAM_ROOT, split="train", skip_missing_files=False)
    if len(ds) == 0:
        raise ValueError("IAMDataset train пуст")

    img, text = ds[0]
    if img.ndim != 3 or img.shape[0] != 1:
        raise ValueError(f"неверная форма изображения: {tuple(img.shape)}")
    if not isinstance(text, str) or not text:
        raise ValueError("пустая текстовая метка")

    val_ds = IAMDataset(IAM_ROOT, split="val", skip_missing_files=False)
    if len(val_ds) == 0:
        raise ValueError("IAMDataset val пуст")

    print(f"      IAM loader: train={len(ds)}, val={len(val_ds)}, sample={text!r}")


def main() -> int:
    print(f"Проверка репозитория: {REPO_ROOT}\n")

    steps = [
        ("зависимости (torch, numpy, Pillow)", check_imports),
        ("MNIST + MNISTModel", check_mnist),
        ("IAM файлы на диске", check_iam_files),
        ("IAM split по писателям", check_iam_split),
        ("IAMDataset (чтение картинки)", check_iam_loader),
    ]

    failed = [name for name, fn in steps if not _step(name, fn)]

    print()
    if failed:
        print(f"Готово: {len(steps) - len(failed)}/{len(steps)} проверок прошло.")
        print("Не прошли:", ", ".join(failed))
        return 1

    print("Готово: все проверки прошли. Можно писать обучение.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

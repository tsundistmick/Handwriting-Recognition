import random
import re
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal
import numpy as np

IAM_VAL_RATIO = 0.2
IAM_SPLIT_SEED = 42


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


_WORD_ID_RE = re.compile(r"^[a-z]\d{2}-[\w]+-\d{2}-\d{2}$")


def _iam_word_image_path(words_root: Path, word_id: str) -> Path:
    parts = word_id.split("-")
    if len(parts) < 4:
        raise ValueError(f"unexpected word id: {word_id!r}")
    writer = parts[0]
    form_id = f"{parts[0]}-{parts[1]}"
    return words_root / writer / form_id / f"{word_id}.png"


def _parse_iam_words_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 9:
        return None
    word_id, status = parts[0], parts[1]
    if not _WORD_ID_RE.match(word_id):
        return None
    text = parts[-1]
    return word_id, status, text


def _iam_writer_id(word_id: str) -> str:
    return word_id.split("-", 1)[0]


def _iam_val_writers(
    writer_counts: dict[str, int],
    val_ratio: float,
    seed: int,
) -> set[str]:
    writers = list(writer_counts.items())
    rng = random.Random(seed)
    rng.shuffle(writers)
    writers.sort(key=lambda item: item[1], reverse=True)

    total = sum(count for _, count in writers)
    target_val = total * val_ratio

    val_writers: set[str] = set()
    val_count = 0
    for writer_id, count in writers:
        if val_count < target_val:
            val_writers.add(writer_id)
            val_count += count
    return val_writers


def _filter_iam_entries_by_split(
    entries: list[tuple[str, str]],
    split: Literal["train", "val"],
    val_ratio: float,
    seed: int,
) -> list[tuple[str, str]]:
    writer_counts: dict[str, int] = {}
    for word_id, _ in entries:
        writer_id = _iam_writer_id(word_id)
        writer_counts[writer_id] = writer_counts.get(writer_id, 0) + 1

    val_writers = _iam_val_writers(writer_counts, val_ratio, seed)
    if split == "val":
        allowed = val_writers
    else:
        allowed = set(writer_counts) - val_writers

    return [
        (word_id, text)
        for word_id, text in entries
        if _iam_writer_id(word_id) in allowed
    ]


class IAMDataset(Dataset):
    def __init__(
        self,
        iam_root: Path,
        words_list: Path | None = None,
        words_root: Path | None = None,
        only_ok: bool = True,
        skip_missing_files: bool = True,
        target_height: int | None = 32,
        split: Literal["train", "val"] | None = None,
        val_ratio: float = IAM_VAL_RATIO,
        split_seed: int = IAM_SPLIT_SEED,
    ):
        iam_root = Path(iam_root)
        if words_list is None:
            words_list = iam_root / "iam_words" / "words.txt"
        else:
            words_list = Path(words_list)
        if words_root is None:
            words_root = iam_root / "iam_words" / "words"
        else:
            words_root = Path(words_root)

        self.words_root = words_root
        self.target_height = target_height

        entries: list[tuple[str, str]] = []
        with words_list.open(encoding="utf-8", errors="replace") as f:
            for raw in f:
                parsed = _parse_iam_words_line(raw)
                if parsed is None:
                    continue
                word_id, status, text = parsed
                if only_ok and status != "ok":
                    continue
                img_path = _iam_word_image_path(words_root, word_id)
                if skip_missing_files and not img_path.is_file():
                    continue
                entries.append((word_id, text))

        if split is not None:
            entries = _filter_iam_entries_by_split(
                entries, split, val_ratio, split_seed
            )

        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx: int):
        from PIL import Image

        word_id, text = self._entries[idx]
        path = _iam_word_image_path(self.words_root, word_id)
        img = Image.open(path).convert("L")

        if self.target_height is not None:
            w, h = img.size
            if h <= 0:
                h = 1
            new_w = max(1, int(round(w * self.target_height / h)))
            img = img.resize((new_w, self.target_height), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, text

import re
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


class IAMDataset(Dataset):
    def __init__(
        self,
        iam_root: Path,
        words_list: Path | None = None,
        words_root: Path | None = None,
        only_ok: bool = True,
        skip_missing_files: bool = True,
        target_height: int | None = 32,
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

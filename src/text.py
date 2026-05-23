import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image
import numpy as np
import random
import math
from collections import defaultdict


ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

BLANK_IDX = 0
CHAR2IDX  = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
IDX2CHAR  = {i + 1: ch for i, ch in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1


class Codec:

    @staticmethod
    def encode(text: str) -> torch.Tensor:
        return torch.tensor(
            [CHAR2IDX[ch] for ch in text if ch in CHAR2IDX],
            dtype=torch.long,
        )

    @staticmethod
    def decode_greedy(log_probs: torch.Tensor) -> str:
        indices = log_probs.argmax(dim=-1).tolist()
        chars, prev = [], None
        for idx in indices:
            if idx != prev:
                if idx != BLANK_IDX:
                    chars.append(IDX2CHAR.get(idx, ""))
            prev = idx
        return "".join(chars)


class HandwritingAugment:

    def __init__(
        self,
        rotation_range: float = 5.0,
        scale_range: tuple  = (0.8, 1.2),
        brightness_range: tuple = (0.7, 1.3),
        contrast_range: tuple   = (0.7, 1.3),
        noise_std: float = 0.04,
        elastic_alpha: float = 15.0,
        elastic_sigma: float = 4.0,
        p_rotation: float  = 0.5,
        p_scale: float     = 0.4,
        p_brightness: float = 0.5,
        p_contrast: float  = 0.4,
        p_noise: float     = 0.3,
        p_elastic: float   = 0.3,
    ):
        self.rotation_range  = rotation_range
        self.scale_range     = scale_range
        self.brightness_range = brightness_range
        self.contrast_range   = contrast_range
        self.noise_std       = noise_std
        self.elastic_alpha   = elastic_alpha
        self.elastic_sigma   = elastic_sigma
        self.p_rotation  = p_rotation
        self.p_scale     = p_scale
        self.p_brightness = p_brightness
        self.p_contrast   = p_contrast
        self.p_noise     = p_noise
        self.p_elastic   = p_elastic

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        H, W = img.shape[1], img.shape[2]

        if random.random() < self.p_rotation:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img = TF.rotate(img, angle, fill=1.0)

        if random.random() < self.p_scale:
            scale = random.uniform(*self.scale_range)
            new_h = max(16, int(H * scale))
            img = TF.resize(img, [new_h, W], antialias=True)
            img = TF.resize(img, [H, W],     antialias=True)

        if random.random() < self.p_brightness:
            factor = random.uniform(*self.brightness_range)
            img = TF.adjust_brightness(img, factor)

        if random.random() < self.p_contrast:
            factor = random.uniform(*self.contrast_range)
            img = TF.adjust_contrast(img, factor)

        if random.random() < self.p_noise:
            noise = torch.randn_like(img) * self.noise_std
            img = (img + noise).clamp(0.0, 1.0)

        if random.random() < self.p_elastic:
            img = self._elastic_transform(img)

        return img

    def _elastic_transform(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape

        dx = torch.rand(1, H, W) * 2 - 1
        dy = torch.rand(1, H, W) * 2 - 1

        k_size = int(self.elastic_sigma * 6) | 1
        kernel = self._gaussian_kernel(k_size, self.elastic_sigma)
        kernel = kernel.view(1, 1, k_size, k_size)

        dx = F.conv2d(
            dx.unsqueeze(0), kernel,
            padding=k_size // 2
        ).squeeze(0) * self.elastic_alpha / W

        dy = F.conv2d(
            dy.unsqueeze(0), kernel,
            padding=k_size // 2
        ).squeeze(0) * self.elastic_alpha / H

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        grid = torch.stack([grid_x + dx.squeeze(0), grid_y + dy.squeeze(0)], dim=-1)
        grid = grid.unsqueeze(0)

        out = F.grid_sample(
            img.unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True
        )
        return out.squeeze(0)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.outer(g)
        return kernel / kernel.sum()


class IAMDataset(Dataset):

    def __init__(
        self,
        data_dir: Path,
        split_ids: list[str],
        img_height: int = 64,
        img_width: int  = 1024,
        augment: bool   = False,
    ):
        self.img_height = img_height
        self.img_width  = img_width
        self.augment = HandwritingAugment() if augment else None

        gt_path = data_dir / "gt" / "lines.txt"
        self.samples = []

        with open(gt_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(" ")
                if len(parts) < 9:
                    continue
                line_id = parts[0]
                status  = parts[1]
                text    = " ".join(parts[8:]).replace("|", " ")

                if status != "ok" or line_id not in split_ids:
                    continue

                p = line_id.split("-")
                img_path = (
                    data_dir / "lines"
                    / p[0]
                    / f"{p[0]}-{p[1]}"
                    / f"{line_id}.png"
                )
                if img_path.exists():
                    self.samples.append((img_path, text))

        print(f"  Загружено {len(self.samples)} строк "
              f"({'с аугментацией' if augment else 'без аугментации'})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]

        img = Image.open(img_path).convert("L")
        w, h = img.size
        new_w = min(int(w * self.img_height / h), self.img_width)
        img   = img.resize((new_w, self.img_height), Image.BILINEAR)

        img_tensor = torch.tensor(
            np.array(img), dtype=torch.float32
        ).unsqueeze(0) / 255.0

        if self.augment is not None:
            img_tensor = self.augment(img_tensor)

        label = Codec.encode(text)
        return img_tensor, label, text


def collate_fn(batch):
    images, labels, texts = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    H     = images[0].shape[1]

    padded = torch.ones(len(images), 1, H, max_w)
    for i, img in enumerate(images):
        padded[i, :, :, :img.shape[2]] = img

    label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets    = torch.cat(labels)

    return padded, targets, label_lens, list(texts)


class CRNNModel(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, (4, 1)), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.3,
        )

        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x.log_softmax(dim=2)


def cer(predicted: str, target: str) -> float:
    n = len(target)
    if n == 0:
        return 0.0
    dp = list(range(len(predicted) + 1))
    for tc in target:
        new_dp = [dp[0] + 1]
        for j, pc in enumerate(predicted):
            new_dp.append(min(
                new_dp[j] + 1,
                dp[j + 1] + 1,
                dp[j] + (0 if tc == pc else 1),
            ))
        dp = new_dp
    return dp[len(predicted)] / n


class BigramLM:

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.bigram_counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.unigram_counts: dict[str, float] = defaultdict(float)
        self.vocab = set(ALPHABET)

    def train(self, texts: list[str]) -> None:
        for text in texts:
            prev = "<s>"
            for ch in text:
                if ch in self.vocab:
                    self.bigram_counts[prev][ch] += 1
                    self.unigram_counts[ch] += 1
                    prev = ch

        vocab_size = len(self.vocab)
        self.log_probs: dict[str, dict[str, float]] = {}
        for a in list(self.bigram_counts.keys()) + ["<s>"]:
            counts = self.bigram_counts[a]
            total  = sum(counts.values()) + self.smoothing * vocab_size
            self.log_probs[a] = {
                b: math.log((counts.get(b, 0) + self.smoothing) / total)
                for b in self.vocab
            }

    def score(self, prev_char: str, next_char: str) -> float:
        if prev_char not in self.log_probs:
            prev_char = "<s>"
        lp = self.log_probs.get(prev_char, {})
        return lp.get(next_char, math.log(self.smoothing / (len(self.vocab) * self.smoothing)))


class KenLMWrapper:

    def __init__(self, arpa_path: Path, alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self.beta  = beta
        self.decoder = None
        self._load(arpa_path)

    def _load(self, arpa_path: Path) -> None:
        try:
            from pyctcdecode import build_ctcdecoder
            labels = [""] + list(ALPHABET)
            self.decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model=str(arpa_path),
                alpha=self.alpha,
                beta=self.beta,
            )
            print(f"  KenLM загружен из {arpa_path}")
        except ImportError:
            print("  pyctcdecode не установлен, используется BigramLM")
        except Exception as e:
            print(f"  Не удалось загрузить KenLM: {e}, используется BigramLM")

    def is_available(self) -> bool:
        return self.decoder is not None

    def decode(self, log_probs_np: np.ndarray) -> str:
        return self.decoder.decode(log_probs_np)


def beam_search_decode(
    log_probs: torch.Tensor,
    lm: BigramLM,
    beam_width: int  = 10,
    lm_weight: float = 0.3,
) -> str:
    T = log_probs.shape[0]

    beams = [("", "<s>", 0.0, 0.0)]

    full_beams = [("", None, BLANK_IDX, 0.0, 0.0)]

    for t in range(T):
        lp_t = log_probs[t]

        next_beams: dict[tuple, list] = defaultdict(list)

        for text, last_out, last_in, ctc_score, lm_score in full_beams:
            blank_score = ctc_score + lp_t[BLANK_IDX].item()
            key = (text, last_out, BLANK_IDX)
            next_beams[key].append((blank_score, lm_score))

            for idx in range(1, NUM_CLASSES):
                ch = IDX2CHAR.get(idx, "")
                char_ctc = lp_t[idx].item()

                if idx == last_in and last_in != BLANK_IDX:
                    key = (text, last_out, idx)
                    next_beams[key].append((ctc_score + char_ctc, lm_score))
                    continue

                new_text = text + ch
                prev_lm_char = last_out if last_out else "<s>"
                new_lm = lm_score + lm.score(prev_lm_char, ch)

                key = (new_text, ch, idx)
                next_beams[key].append((ctc_score + char_ctc, new_lm))

        collapsed = []
        for (text, last_out, last_in), scores in next_beams.items():
            ctc_scores = [s[0] for s in scores]
            lm_scores  = [s[1] for s in scores]
            best_ctc = max(ctc_scores)
            merged_ctc = best_ctc + math.log(
                sum(math.exp(s - best_ctc) for s in ctc_scores)
            )
            collapsed.append((text, last_out, last_in, merged_ctc, lm_scores[0]))

        collapsed.sort(
            key=lambda x: x[3] + lm_weight * x[4],
            reverse=True,
        )
        full_beams = collapsed[:beam_width]

    if not full_beams:
        return ""
    best = max(full_beams, key=lambda x: x[3] + lm_weight * x[4])
    return best[0]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lm: BigramLM,
        beam_width: int  = 10,
        lm_weight: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.lm           = lm
        self.beam_width   = beam_width
        self.lm_weight    = lm_weight

        self.criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None

    def _init_scheduler(self, num_epochs: int) -> None:
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            steps_per_epoch=len(self.train_loader),
            epochs=num_epochs,
        )

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, targets, target_lens, _) in enumerate(self.train_loader):
            images      = images.to(self.device)
            targets     = targets.to(self.device)
            target_lens = target_lens.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            log_probs = self.model(images)

            T = log_probs.shape[0]
            input_lens = torch.full((images.shape[0],), T, dtype=torch.long, device=self.device)

            loss = self.criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Эпоха {epoch} | Батч {batch_idx+1}/{len(self.train_loader)}"
                      f" | Loss: {total_loss/(batch_idx+1):.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader = None,
        use_beam: bool = True,
    ) -> tuple[float, float]:
        loader = loader or self.val_loader
        self.model.eval()

        total_loss = 0.0
        total_cer  = 0.0
        count = 0

        for images, targets, target_lens, texts in loader:
            images      = images.to(self.device)
            targets     = targets.to(self.device)
            target_lens = target_lens.to(self.device)

            log_probs = self.model(images)
            T = log_probs.shape[0]
            input_lens = torch.full((images.shape[0],), T, dtype=torch.long, device=self.device)

            loss = self.criterion(log_probs, targets, input_lens, target_lens)
            total_loss += loss.item()

            lp_cpu = log_probs.permute(1, 0, 2).cpu()
            for lp, true_text in zip(lp_cpu, texts):
                if use_beam:
                    pred = beam_search_decode(
                        lp, self.lm,
                        beam_width=self.beam_width,
                        lm_weight=self.lm_weight,
                    )
                else:
                    pred = Codec.decode_greedy(lp)

                total_cer += cer(pred, true_text)
                count += 1

        return total_loss / len(loader), total_cer / max(count, 1)

    def fit(self, num_epochs: int, save_path: Path) -> None:
        self._init_scheduler(num_epochs)
        best_cer = float("inf")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"ЭПОХА {epoch}/{num_epochs}")
            print(f"{'='*60}")

            train_loss = self.train_one_epoch(epoch)
            print(f"\n  Train Loss: {train_loss:.4f}")

            use_beam = (epoch >= num_epochs // 2)
            mode_str = "Beam+LM" if use_beam else "Greedy"

            val_loss, val_cer = self.validate(use_beam=use_beam)
            print(f"  Val [{mode_str}] Loss: {val_loss:.4f} | CER: {val_cer*100:.2f}%")

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Лучший CER {best_cer*100:.2f}%! Сохранено → {save_path}")

        print(f"\nОбучение завершено. Лучший CER: {best_cer*100:.2f}%")


def load_split_ids(data_dir: Path, split: str) -> list[str]:
    splits_dir = data_dir / "splits"
    split_map  = {"train": "trainset.txt", "val": "validationset1.txt", "test": "testset.txt"}
    split_file = splits_dir / split_map.get(split, "trainset.txt")
    gt_path    = data_dir / "gt" / "lines.txt"

    if split_file.exists():
        with open(split_file) as f:
            ids = set(l.strip() for l in f if l.strip())
        line_ids = []
        with open(gt_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                lid = parts[0]
                fid = "-".join(lid.split("-")[:2])
                if (lid in ids or fid in ids) and parts[1] == "ok":
                    line_ids.append(lid)
        return line_ids

    print(f"  Файлы сплитов не найдены в {splits_dir}, делаем 80/10/10.")
    all_ids = []
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "ok":
                all_ids.append(parts[0])
    random.seed(42)
    random.shuffle(all_ids)
    n = len(all_ids)
    return {"train": all_ids[:int(n*.8)],
            "val":   all_ids[int(n*.8):int(n*.9)],
            "test":  all_ids[int(n*.9):]}[split]


def recognize(
    model: nn.Module,
    img_path: Path,
    device: torch.device,
    lm: BigramLM = None,
    beam_width: int = 10,
    lm_weight: float = 0.3,
    img_height: int = 64,
) -> str:
    model.eval()
    img = Image.open(img_path).convert("L")
    w, h = img.size
    img  = img.resize((int(w * img_height / h), img_height), Image.BILINEAR)
    t    = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        lp = model(t.to(device)).squeeze(1).cpu()

    if lm is not None:
        return beam_search_decode(lp, lm, beam_width, lm_weight)
    return Codec.decode_greedy(lp)


def main():
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    DATA_DIR  = Path("./iam")
    SAVE_PATH = Path("./best_crnn.pth")
    ARPA_PATH = Path("./lm/english.arpa")

    BATCH_SIZE   = 16
    NUM_EPOCHS   = 30
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS  = 4
    IMG_HEIGHT   = 64
    BEAM_WIDTH   = 10
    LM_WEIGHT    = 0.3

    print("Загружаем сплиты...")
    train_ids = load_split_ids(DATA_DIR, "train")
    val_ids   = load_split_ids(DATA_DIR, "val")
    test_ids  = load_split_ids(DATA_DIR, "test")

    print("\nТренировочный датасет:")
    train_ds = IAMDataset(DATA_DIR, train_ids, IMG_HEIGHT, augment=True)
    print("Валидационный датасет:")
    val_ds   = IAMDataset(DATA_DIR, val_ids,   IMG_HEIGHT, augment=False)
    print("Тестовый датасет:")
    test_ds  = IAMDataset(DATA_DIR, test_ids,  IMG_HEIGHT, augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device.type=="cuda"))

    lm: BigramLM = None

    if ARPA_PATH.exists():
        kenlm = KenLMWrapper(ARPA_PATH, alpha=0.5, beta=1.0)
        if kenlm.is_available():
            print("  Используется KenLM.")

    print("\nОбучаем биграммную языковую модель на тренировочных текстах...")
    lm = BigramLM(smoothing=0.1)
    train_texts = [train_ds.samples[i][1] for i in range(len(train_ds.samples))]
    lm.train(train_texts)
    print(f"  Биграмм построено из {len(train_texts)} строк.")

    model = CRNNModel(NUM_CLASSES)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nПараметров модели: {params:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lm=lm,
        beam_width=BEAM_WIDTH,
        lm_weight=LM_WEIGHT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    trainer.fit(NUM_EPOCHS, SAVE_PATH)

    print("\nЗагружаем лучшую модель для финального теста...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    _, cer_greedy = trainer.validate(test_loader, use_beam=False)
    _, cer_beam   = trainer.validate(test_loader, use_beam=True)
    print(f"\nРезультаты на тестовой выборке:")
    print(f"  Greedy декодирование : CER = {cer_greedy*100:.2f}%")
    print(f"  Beam Search + LM     : CER = {cer_beam*100:.2f}%")
    print(f"  Улучшение от LM      : {(cer_greedy - cer_beam)*100:.2f}% абс.")


if __name__ == "__main__":
    main()

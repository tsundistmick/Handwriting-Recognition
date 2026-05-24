import csv
import pytesseract
from PIL import Image
from jiwer import cer, wer
from pathlib import Path

DATA_PATH   = Path("iam")
LINES_DIR  = DATA_PATH / "lines"
GT_FILE    = DATA_PATH / "gt" / "lines.txt"
SPLITS_DIR = DATA_PATH / "splits"
OUTPUT_CSV = Path("results") / "tesseract_baseline.csv"

TESS_CONFIG = "--psm 7 -l eng -c load_system_dawg=false -c load_freq_dawg=false"

def load_gt(gt_file: Path) -> dict[str, str]:
    gt = {}
    with open(gt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) < 9:
                continue
            line_id = parts[0]         
            status  = parts[1]          
            if status != "ok":
                continue
            transcription = " ".join(parts[8:]).replace("|", " ")
            gt[line_id] = transcription
    return gt

def load_split_ids(splits_dir: Path, split: str) -> list[str]:
    name_map = {
        "train": "trainset.txt",
        "val":   "validationset1.txt",   
        "test":  "testset.txt",
    }
    split_file = splits_dir / name_map[split]
    ids = []
    with open(split_file) as f:
        for line in f:
            line_id = line.strip()
            if line_id:
                ids.append(line_id)
    return ids


def line_id_to_image_path(line_id: str, lines_dir: Path) -> Path:
    parts   = line_id.split("-")        
    writer  = parts[0]                    
    form    = f"{parts[0]}-{parts[1]}"   
    fname   = f"{line_id}.png"
    return lines_dir / writer / form / fname


def run_tesseract(image_path: Path) -> str:
    img  = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img, config=TESS_CONFIG)
    return text.strip()


def main(split: str = "test"):
    gt = load_gt(GT_FILE)
    ids = load_split_ids(SPLITS_DIR, split)
    references   = []
    hypotheses   = []
    skipped      = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["line_id", "reference", "hypothesis"])

        for i, line_id in enumerate(ids):
            if line_id not in gt:
                skipped += 1
                continue

            img_path = line_id_to_image_path(line_id, LINES_DIR)
            if not img_path.exists():
                skipped += 1
                continue

            ref  = gt[line_id]
            hyp  = run_tesseract(img_path)

            references.append(ref)
            hypotheses.append(hyp)
            writer.writerow([line_id, ref, hyp])

    print(f"      Обработано:  {len(references)}")
    print(f"      Пропущено:   {skipped}")
    print(f"      CER: {cer(references, hypotheses):.4f}")
    print(f"      WER: {wer(references, hypotheses):.4f}")


if __name__ == "__main__":
    main(split="test")
import csv
import torch
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
from tesseract_baseline import load_gt, load_split_ids, line_id_to_image_path

DATA_PATH   = Path("iam")
LINES_DIR  = DATA_PATH / "lines"
GT_FILE    = DATA_PATH / "gt" / "lines.txt"
SPLITS_DIR = DATA_PATH / "splits"
OUTPUT_CSV = Path("results") / "trocr_baseline.csv"

MODEL_NAME  = "microsoft/trocr-base-handwritten"
BATCH_SIZE  = 8      
DEVICE      = "cpu"

def run_trocr_batch(images: list, processor, model) -> list:
    pixel_values = processor(
        images,
        return_tensors="pt",
        padding=True           
    ).pixel_values.to(DEVICE)

    with torch.no_grad():    
        generated_ids = model.generate(pixel_values)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts

def main(split: str = "test", max_samples: int = None):
    gt = load_gt(GT_FILE)
    ids = load_split_ids(SPLITS_DIR, split)
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()              

    references  = []
    hypotheses  = []
    skipped     = 0
    batch_images  = []
    batch_refs    = []
    batch_ids     = []

    def process_batch():
        texts = run_trocr_batch(batch_images, processor, model)
        for line_id, ref, hyp in zip(batch_ids, batch_refs, texts):
            references.append(ref)
            hypotheses.append(hyp)
            writer_csv.writerow([line_id, ref, hyp])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["line_id", "reference", "hypothesis"])

        for i, line_id in enumerate(ids):
            if max_samples and i >= max_samples:
                break

            if line_id not in gt:
                skipped += 1
                continue

            img_path = line_id_to_image_path(line_id, LINES_DIR)
            if not img_path.exists():
                skipped += 1
                continue

            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            batch_refs.append(gt[line_id])
            batch_ids.append(line_id)
            if len(batch_images) == BATCH_SIZE:
                process_batch()
                batch_images.clear()
                batch_refs.clear()
                batch_ids.clear()

        if batch_images:
            process_batch()

    print(f"      Обработано:  {len(references)}")
    print(f"      Пропущено:   {skipped}")
    print(f"      CER: {cer(references, hypotheses):.4f}")
    print(f"      WER: {wer(references, hypotheses):.4f}")


if __name__ == "__main__":
    main(split="test")
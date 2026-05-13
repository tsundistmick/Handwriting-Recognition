import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "iam_data"


def find_words_bundle(root: Path) -> tuple[Path, Path] | None:
    for words_txt in root.rglob("words.txt"):
        if "git" in words_txt.parts:
            continue
        parent = words_txt.parent
        words_dir = parent / "words"
        if words_dir.is_dir():
            return words_txt, words_dir
    return None


def verify(dest: Path) -> int:
    dest = Path(dest).resolve()
    words_txt = dest / "iam_words" / "words.txt"
    words_dir = dest / "iam_words" / "words"
    if not words_txt.is_file():
        print(f"нет файла: {words_txt}", file=sys.stderr)
        return 1
    if not words_dir.is_dir():
        print(f"нет папки: {words_dir}", file=sys.stderr)
        return 1
    pngs = list(words_dir.rglob("*.png"))
    if not pngs:
        print("в words/ не найдено ни одного .png", file=sys.stderr)
        return 1
    print(f"ok: {words_txt}")
    print(f"ok: {words_dir} ({len(pngs)} png)")
    return 0


def extract_zip(zip_path: Path, dest: Path) -> int:
    zip_path = Path(zip_path).resolve()
    dest = Path(dest).resolve()
    if not zip_path.is_file():
        print(f"нет архива: {zip_path}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    target_words = dest / "iam_words"
    target_words.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        bundle = find_words_bundle(tmp)
        if bundle is None:
            print(
                "в архиве не найдена пара words.txt + соседняя папка words/ "
                "(проверь, что это IAM word-level пакет).",
                file=sys.stderr,
            )
            return 1

        src_txt, src_words = bundle
        dst_txt = target_words / "words.txt"
        dst_words = target_words / "words"

        if dst_txt.exists():
            dst_txt.unlink()
        if dst_words.exists():
            shutil.rmtree(dst_words)

        shutil.copy2(src_txt, dst_txt)
        shutil.copytree(src_words, dst_words)

    print(f"распаковано в: {target_words}")
    return verify(dest)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Подготовка iam_data: распаковка локального zip с IAM words и проверка структуры.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"куда положить iam_words (по умолчанию {DEFAULT_DEST})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="проверить, что в --dest уже лежит words.txt + words/")
    pv.set_defaults(func=lambda a: verify(a.dest))

    pe = sub.add_parser("extract", help="распаковать zip, скачанный с IAM вручную")
    pe.add_argument("--archive", type=Path, required=True, help="путь к .zip с IAM words")
    pe.set_defaults(func=lambda a: extract_zip(a.archive, a.dest))

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

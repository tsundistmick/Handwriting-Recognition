import argparse
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "iam"

SPLIT_FILES = (
    "trainset.txt",
    "validationset1.txt",
    "validationset2.txt",
    "testset.txt",
)


def _extract_archive(archive: Path, dest: Path) -> None:
    archive = Path(archive)
    suffixes = "".join(archive.suffixes).lower()
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif suffixes.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"неизвестный формат архива: {archive.name}")


def _find_lines_root(extracted: Path) -> Path | None:
    for png in extracted.rglob("*.png"):
        candidate = png.parent.parent.parent
        try:
            candidate.relative_to(extracted)
        except ValueError:
            continue
        return candidate
    return None


def _find_lines_txt(extracted: Path) -> Path | None:
    for cand in extracted.rglob("lines.txt"):
        return cand
    return None


def _find_split_files(extracted: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in SPLIT_FILES:
        for cand in extracted.rglob(name):
            found[name] = cand
            break
    return found


def _move_into(src: Path, dst: Path, tmp_root: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == tmp_root.resolve():
        dst.mkdir(parents=True, exist_ok=False)
        for child in list(src.iterdir()):
            shutil.move(str(child), str(dst / child.name))
    else:
        shutil.move(str(src), str(dst))


def extract(
    lines_archive: Path,
    ascii_archive: Path,
    splits_archive: Path | None,
    dest: Path,
) -> int:
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    lines_archive = Path(lines_archive).resolve()
    ascii_archive = Path(ascii_archive).resolve()
    if not lines_archive.is_file():
        print(f"нет архива: {lines_archive}", file=sys.stderr)
        return 1
    if not ascii_archive.is_file():
        print(f"нет архива: {ascii_archive}", file=sys.stderr)
        return 1
    if splits_archive is not None:
        splits_archive = Path(splits_archive).resolve()
        if not splits_archive.is_file():
            print(f"нет архива: {splits_archive}", file=sys.stderr)
            return 1

    print(f"распаковка lines: {lines_archive.name} → {dest / 'lines'}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            _extract_archive(lines_archive, tmp)
        except Exception as exc:
            print(f"не удалось распаковать {lines_archive}: {exc}", file=sys.stderr)
            return 1
        src = _find_lines_root(tmp)
        if src is None:
            print(
                f"в {lines_archive.name} не найдено картинок строк "
                "(структура writer/form/line.png)",
                file=sys.stderr,
            )
            return 1
        _move_into(src, dest / "lines", tmp)

    print(f"распаковка ascii: {ascii_archive.name} → {dest / 'gt' / 'lines.txt'}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            _extract_archive(ascii_archive, tmp)
        except Exception as exc:
            print(f"не удалось распаковать {ascii_archive}: {exc}", file=sys.stderr)
            return 1
        src = _find_lines_txt(tmp)
        if src is None:
            print(f"в {ascii_archive.name} не найден lines.txt", file=sys.stderr)
            return 1
        gt_dst = dest / "gt" / "lines.txt"
        gt_dst.parent.mkdir(parents=True, exist_ok=True)
        if gt_dst.exists():
            gt_dst.unlink()
        shutil.copy2(src, gt_dst)

    if splits_archive is not None:
        print(f"распаковка splits: {splits_archive.name} → {dest / 'splits'}")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            try:
                _extract_archive(splits_archive, tmp)
            except Exception as exc:
                print(
                    f"не удалось распаковать {splits_archive}: {exc}",
                    file=sys.stderr,
                )
                return 1
            found = _find_split_files(tmp)
            if not found:
                print(
                    f"в {splits_archive.name} не найдено ни одного из "
                    + ", ".join(SPLIT_FILES),
                    file=sys.stderr,
                )
                return 1
            splits_dst = dest / "splits"
            if splits_dst.exists():
                shutil.rmtree(splits_dst)
            splits_dst.mkdir(parents=True)
            for name, p in found.items():
                shutil.copy2(p, splits_dst / name)
    else:
        print(
            "splits архив не указан — text.py перейдёт на fallback 80/10/10 "
            "по строкам (без writer-independent)."
        )

    print()
    return verify(dest)


def verify(dest: Path) -> int:
    dest = Path(dest).resolve()
    ok = True

    gt = dest / "gt" / "lines.txt"
    if gt.is_file():
        n = 0
        with gt.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(" ")
                if len(parts) >= 9 and parts[1] == "ok":
                    n += 1
        print(f"ok: {gt} (строк со статусом ok: {n})")
    else:
        print(f"нет файла: {gt}", file=sys.stderr)
        ok = False

    lines_dir = dest / "lines"
    if lines_dir.is_dir():
        n = sum(1 for _ in lines_dir.rglob("*.png"))
        if n == 0:
            print(f"в {lines_dir} нет .png", file=sys.stderr)
            ok = False
        else:
            print(f"ok: {lines_dir} ({n} png)")
    else:
        print(f"нет папки: {lines_dir}", file=sys.stderr)
        ok = False

    splits_dir = dest / "splits"
    if splits_dir.is_dir():
        present = [n for n in SPLIT_FILES if (splits_dir / n).is_file()]
        if present:
            print(f"ok: {splits_dir} ({', '.join(present)})")
        else:
            print(
                f"в {splits_dir} нет ни одного из " + ", ".join(SPLIT_FILES),
                file=sys.stderr,
            )
            ok = False
    else:
        print(f"нет папки {splits_dir} — будет fallback 80/10/10 в text.py")

    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Подготовка iam/: распаковка lines.tgz, ascii.tgz и "
            "(опц.) largeWriterIndependentTextLineRecognitionTask.zip "
            "с FKI IAM в корневую папку iam/."
        ),
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"куда положить iam (по умолчанию {DEFAULT_DEST})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "verify",
        help="проверить, что в --dest уже лежат lines/, gt/lines.txt, splits/",
    )
    pv.set_defaults(func=lambda a: verify(a.dest))

    pe = sub.add_parser("extract", help="распаковать архивы IAM lines в --dest")
    pe.add_argument("--lines", type=Path, required=True, help="путь к lines.tgz")
    pe.add_argument("--ascii", type=Path, required=True, help="путь к ascii.tgz")
    pe.add_argument(
        "--splits",
        type=Path,
        default=None,
        help=(
            "путь к largeWriterIndependentTextLineRecognitionTask.zip "
            "(опционально; без него — fallback 80/10/10)"
        ),
    )
    pe.set_defaults(func=lambda a: extract(a.lines, a.ascii, a.splits, a.dest))

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare existing BTL02 split conventions from the original FER2013 CSV/ZIP.

Training rows -> train.csv (training_loaders later reserves stratified 20% validation).
PublicTest followed by PrivateTest in original file order -> labeled test.csv.
"""
import argparse
import csv
import io
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


def prepare_data(source, output_dir):
    source, output_dir = Path(source), Path(output_dir)
    names = ("fer2013.csv", "train.csv", "test.csv")
    existing = [str(output_dir / name) for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("Không ghi đè dữ liệu đã có: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    # Stage and validate every row before publishing any CSV. No ZIP path extraction.
    with tempfile.TemporaryDirectory(prefix="fer2013_", dir=output_dir) as temporary:
        stage = Path(temporary)
        original = stage / "fer2013.csv"
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                members = [info for info in archive.infolist()
                           if not info.is_dir() and Path(info.filename).name.lower() == "fer2013.csv"]
                if len(members) != 1:
                    raise ValueError("ZIP cần đúng một file có tên fer2013.csv.")
                with archive.open(members[0]) as incoming, original.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        else:
            shutil.copyfile(source, original)
        with original.open(encoding="utf-8-sig", newline="") as file, \
                (stage / "train.csv").open("w", encoding="utf-8", newline="") as train, \
                (stage / "test.csv").open("w", encoding="utf-8", newline="") as test:
            reader = csv.DictReader(file)
            if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV cần emotion, pixels, Usage.")
            train_writer, test_writer = csv.writer(train), csv.writer(test)
            for writer in (train_writer, test_writer):
                writer.writerow(["emotion", "pixels"])
            for row_number, row in enumerate(reader, start=2):
                usage = row["Usage"]
                if usage not in {"Training", "PublicTest", "PrivateTest"}:
                    raise ValueError(f"Dòng {row_number}: Usage không hợp lệ: {usage!r}")
                try:
                    label = int(row["emotion"])
                    pixels = list(map(int, row["pixels"].split()))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(f"Dòng {row_number}: nhãn/pixel phải là số nguyên.") from exc
                if label not in range(7) or len(pixels) != 2304 or min(pixels) < 0 or max(pixels) > 255:
                    raise ValueError(f"Dòng {row_number}: cần nhãn 0..6 và 2304 pixel 0..255.")
                writer = train_writer if usage == "Training" else test_writer
                writer.writerow([label, row["pixels"]])
                counts[usage] += 1
            if any(counts[usage] == 0 for usage in ("Training", "PublicTest", "PrivateTest")):
                raise ValueError("Thiếu một trong ba tập FER2013.")
        for name in names:
            (stage / name).replace(output_dir / name)
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị FER2013 cho quy tắc split hiện tại của BTL02")
    parser.add_argument("source", type=Path, help="fer2013.csv hoặc ZIP chứa file này")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    print(prepare_data(args.source, args.output_dir))

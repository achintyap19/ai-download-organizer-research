# experiments/prepare_ground_truth.py
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.scanner import scan_directory

ALLOWED_CATEGORIES = [
    "Software_Installers",
    "Archives_Compressed",
    "Finance_Invoices",
    "Academic_Study",
    "Career_Identity",
    "Code_Development",
    "Media_Screenshots",
    "Media_Photos_Graphics",
    "Media_Audio_Video",
    "Spreadsheets_Data",
    "Uncategorized_Misc",
]


def main():
    raw_dir = ROOT / "datasets/sandbox_raw"
    out_csv = ROOT / "datasets/ground_truth.csv"

    records = scan_directory(raw_dir, extract_text=False)
    if not records:
        print(f"No files found in {raw_dir}. Copy sample files there first.")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "extension", "size_bytes", "is_exact_duplicate", "true_category"])
        for rec in records:
            writer.writerow([
                rec.name,
                rec.extension,
                rec.size_bytes,
                int(rec.is_duplicate),
                "",  # <-- Fill this column manually with one of ALLOWED_CATEGORIES
            ])

    print(f"Created {out_csv} with {len(records)} rows.")
    print("Allowed categories:", ", ".join(ALLOWED_CATEGORIES))


if __name__ == "__main__":
    main()
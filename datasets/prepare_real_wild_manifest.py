import csv
import hashlib
from pathlib import Path

RAW_DIR = Path("datasets/real_wild_downloads")
OUT_CSV = Path("datasets/real_wild_ground_truth.csv")

VALID_CATEGORIES = [
    "Academic_Study",
    "Finance_Invoices",
    "Career_Identity",
    "Spreadsheets_Data",
    "Software_Installers",
    "Archives_Compressed",
    "Media_Visuals",
    "Travel_Bookings",      # Held-out unseen domain for Stage 2 Support/Query test
    "Medical_Health",       # Held-out unseen domain for Stage 2 Support/Query test
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    if not RAW_DIR.exists():
        print(f"Directory not found: {RAW_DIR}")
        return

    files = sorted([p for p in RAW_DIR.rglob("*") if p.is_file()])
    if not files:
        print(f"No files found in {RAW_DIR}.")
        return

    seen_hashes = {}
    seen_prefixes = {}
    unique_rows = []
    skipped_dups = 0
    near_dup_warnings = 0

    for p in files:
        if p.stat().st_size == 0:
            continue
        digest = sha256_file(p)
        if digest in seen_hashes:
            skipped_dups += 1
            print(f"[SKIPPED EXACT DUPLICATE] {p.name} == {seen_hashes[digest]}")
            continue
        seen_hashes[digest] = p.name

        raw_head = p.read_bytes()[:512]
        head_sig = hashlib.md5(raw_head).hexdigest()
        if head_sig in seen_prefixes and p.suffix.lower() in {".txt", ".md", ".csv", ".html"}:
            near_dup_warnings += 1
            print(f"[WARNING NEAR-IDENTICAL HEADER] {p.name} ~ {seen_prefixes[head_sig]}")
        seen_prefixes[head_sig] = p.name

        unique_rows.append({
            "filename": p.name,
            "relative_path": str(p.relative_to(RAW_DIR)).replace("\\", "/"),
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "sha256": digest,
            "true_category": "",
            "source_domain": "downloads_untouched",
            "split": "blind_test", # blind_test | stage2_support | stage2_query
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(unique_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\nCreated {OUT_CSV} with {len(unique_rows)} unique files.")
    print(f"Skipped exact duplicates: {skipped_dups} | Near-identical header warnings: {near_dup_warnings}")
    print("\nHarvested files:")
    for i, r in enumerate(unique_rows, 1):
        print(f"  {i:02d}. {r['filename']} ({r['extension']}, {r['size_bytes']} B)")

if __name__ == "__main__":
    main()

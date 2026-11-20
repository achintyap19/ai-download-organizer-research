import csv
from pathlib import Path

RAW_DIR = Path("datasets/real_wild_downloads")
CSV_PATH = Path("datasets/real_wild_ground_truth.csv")

def read_raw_snippet(path: Path, max_chars: int = 180) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            import fitz
            with fitz.open(path) as doc:
                text = ""
                for page in doc[:2]:
                    text += page.get_text("text") + " "
                    if len(text.strip()) >= max_chars:
                        break
                clean = " ".join(text.split())
                return clean[:max_chars] if clean else "[SCANNED_OR_IMAGE_PDF_NO_TEXT_LAYER]"
        elif ext == ".docx":
            import docx
            d = docx.Document(path)
            text = " ".join(p.text for p in d.paragraphs if p.text.strip())
            return " ".join(text.split())[:max_chars]
        elif ext in {".txt", ".md", ".csv", ".html", ".json", ".xml", ".py", ".log"}:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return " ".join(raw.split())[:max_chars]
        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"}:
            return "[IMAGE_FILE]"
        elif ext in {".exe", ".msi", ".whl", ".deb", ".apk"}:
            return "[INSTALLER_OR_PACKAGE_BINARY]"
        elif ext in {".zip", ".7z", ".rar", ".tar", ".gz"}:
            return "[ARCHIVE_CONTAINER]"
        else:
            return f"[OTHER_BINARY:{ext}]"
    except Exception as e:
        return f"[READ_ERROR:{type(e).__name__}]"

def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total unique files in manifest: {len(rows)}\n")
    for idx, row in enumerate(rows, 1):
        p = RAW_DIR / row["relative_path"]
        snippet = read_raw_snippet(p)
        print(f"{idx:03d} | {row['filename']} | {row['extension']} | {snippet}")

if __name__ == "__main__":
    main()

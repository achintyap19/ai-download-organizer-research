# experiments/inspect_snippets.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.scanner import scan_directory

def main():
    records = scan_directory(ROOT / "datasets/sandbox_raw", extract_text=True)
    for rec in records:
        snippet = " ".join(rec.extracted_text.split())[:160]
        status = snippet if snippet else "[NO SELECTABLE TEXT - SCANNED PDF OR BINARY]"
        print(f"{rec.name:<30} | {status}")

if __name__ == "__main__":
    main()
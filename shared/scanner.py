from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
from .models import FileRecord

TEMP_EXTENSIONS: Set[str] = {".crdownload", ".part", ".tmp", ".download"}
INSTALLER_EXTENSIONS: Set[str] = {".exe", ".msi", ".msix", ".deb", ".rpm", ".appimage", ".dmg", ".pkg", ".apk", ".whl", ".jar", ".vsix"}
ARCHIVE_EXTENSIONS: Set[str] = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".xz", ".bz2"}
MEDIA_AV_EXTENSIONS: Set[str] = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
BINARY_SKIP_EXTENSIONS: Set[str] = INSTALLER_EXTENSIONS | ARCHIVE_EXTENSIONS | MEDIA_AV_EXTENSIONS

PLAIN_TEXT_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".sql",
    ".sh", ".bash", ".ps1", ".bat", ".cmd",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".xml", ".html", ".htm", ".css", ".svg",
}

COPY_SUFFIX_RE = re.compile(r"(\s*\(\d+\)|\s*-\s*copy)$", re.IGNORECASE)

_OCR_ENGINE = None


def _get_ocr_engine():
    """Lazy-loads RapidOCR once per process when a scanned PDF or raster image is encountered."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _run_ocr_on_image_bytes(img_bytes: bytes) -> str:
    try:
        engine = _get_ocr_engine()
        result, _ = engine(img_bytes)
        if not result:
            return ""
        return "\n".join(line[1] for line in result if len(line) >= 2)
    except Exception:
        return ""


def _hash_bytes(path: Path, full: bool = True, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        if full or file_size <= chunk_size * 2:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        else:
            h.update(f.read(chunk_size))
            f.seek(-chunk_size, 2)
            h.update(f.read(chunk_size))
    return h.hexdigest()


def _canonical_rank(record: FileRecord) -> Tuple[int, int, float]:
    has_copy_suffix = 1 if COPY_SUFFIX_RE.search(record.stem) else 0
    return (has_copy_suffix, len(record.name), record.modified_at.timestamp())


def detect_exact_duplicates(records: List[FileRecord]) -> Dict[str, List[FileRecord]]:
    by_size: Dict[int, List[FileRecord]] = defaultdict(list)
    for rec in records:
        if rec.size_bytes > 0:
            by_size[rec.size_bytes].append(rec)

    duplicate_groups: Dict[str, List[FileRecord]] = {}
    for size_group in by_size.values():
        if len(size_group) < 2:
            continue

        by_partial: Dict[str, List[FileRecord]] = defaultdict(list)
        for rec in size_group:
            by_partial[_hash_bytes(rec.path, full=False)].append(rec)

        for partial_group in by_partial.values():
            if len(partial_group) < 2:
                continue

            by_full: Dict[str, List[FileRecord]] = defaultdict(list)
            for rec in partial_group:
                full_digest = _hash_bytes(rec.path, full=True)
                rec.sha256 = full_digest
                by_full[full_digest].append(rec)

            for full_digest, exact_matches in by_full.items():
                if len(exact_matches) > 1:
                    exact_matches.sort(key=_canonical_rank)
                    canonical = exact_matches[0]
                    for dup in exact_matches[1:]:
                        dup.is_duplicate = True
                        dup.duplicate_of = canonical.path
                    duplicate_groups[full_digest] = exact_matches

    return duplicate_groups


def extract_document_content(path: Path, use_ocr: bool = True, max_chars: int = 2000) -> Tuple[str, str]:
    """
    Purpose-First 3-Tier Extraction Cascade:
    1. Tag temp/partial downloads and skip pure binaries/archives/audio-video (0 ms).
    2. Extract native digital text from PDF, DOCX, XLSX, PPTX, Jupyter Notebooks, Parquet, Markdown, Data, and Code.
    3. Trigger RapidOCR on scanned PDFs (<30 chars selectable text) AND all raster images so document photos/screenshots
       are classified by purpose rather than raw image extension.
    """
    ext = path.suffix.lower()
    if ext in TEMP_EXTENSIONS:
        return "", "TEMP_PARTIAL"
    if ext in BINARY_SKIP_EXTENSIONS:
        return "", "SKIPPED_BINARY"

    try:
        if path.stat().st_size == 0:
            return "", "EMPTY_FILE"

        if ext == ".ipynb":
            raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            parts = []
            for cell in raw.get("cells", [])[:25]:
                src = "".join(cell.get("source", []))
                if src.strip():
                    parts.append(src.strip())
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext in PLAIN_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return text[:max_chars], "NATIVE_TEXT" if text else "EMPTY_FILE"

        if ext == ".pdf":
            import pymupdf
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    return "", "PASSWORD_PROTECTED"
                text = ""
                for page in doc[:3]:
                    text += page.get_text("text") + "\n"
                    if len(text) >= max_chars:
                        break
                clean_text = text.strip()
                if len(clean_text) >= 30:
                    return clean_text[:max_chars], "NATIVE_TEXT"

                if use_ocr and len(doc) > 0:
                    ocr_parts = []
                    for page in doc[:2]:
                        pix = page.get_pixmap(dpi=150)
                        ocr_text = _run_ocr_on_image_bytes(pix.tobytes("png"))
                        if ocr_text.strip():
                            ocr_parts.append(ocr_text.strip())
                    combined_ocr = "\n".join(ocr_parts).strip()
                    if combined_ocr:
                        return combined_ocr[:max_chars], "OCR_USED"
                return clean_text[:max_chars], "EMPTY_OR_SCANNED"

        if ext == ".docx":
            import docx
            doc = docx.Document(path)
            parts = [p.text for p in doc.paragraphs[:40] if p.text.strip()]
            for table in doc.tables[:3]:
                for row in table.rows[:12]:
                    parts.append(" ".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            parts = [f"Sheets: {', '.join(wb.sheetnames)}"]
            for sheet in wb.sheetnames[:3]:
                ws = wb[sheet]
                for row in ws.iter_rows(max_row=25, values_only=True):
                    row_text = " ".join(str(v) for v in row if v is not None)
                    if row_text.strip():
                        parts.append(row_text)
            wb.close()
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(path)
            parts = []
            for slide in prs.slides[:6]:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text.strip())
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            header = "Columns: " + ", ".join(str(c) for c in df.columns)
            sample = df.head(10).to_string(index=False)
            return f"{header}\n{sample}"[:max_chars], "NATIVE_TEXT"

        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            if use_ocr:
                ocr_text = _run_ocr_on_image_bytes(path.read_bytes()).strip()
                if ocr_text:
                    return ocr_text[:max_chars], "OCR_USED"
                return "", "OCR_EMPTY_IMAGE"
            return "", "SKIPPED_IMAGE_NON_DOC"

    except Exception:
        return "", "CORRUPT_OR_UNREADABLE"

    return "", "UNSUPPORTED_EXT"


def scan_directory(target_dir: Path, extract_text: bool = True, use_ocr: bool = True) -> List[FileRecord]:
    records: List[FileRecord] = []
    for item in sorted(target_dir.iterdir()):
        if item.is_dir() or item.name.startswith("."):
            continue
        stat = item.stat()
        text, status = ("", "NOT_EXTRACTED")
        if extract_text:
            text, status = extract_document_content(item, use_ocr=use_ocr)

        records.append(
            FileRecord(
                path=item,
                name=item.name,
                stem=item.stem,
                extension=item.suffix.lower(),
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                extracted_text=text,
                extraction_status=status,
            )
        )

    detect_exact_duplicates(records)
    return records

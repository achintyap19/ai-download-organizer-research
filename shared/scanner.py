from collections import defaultdict
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
from .models import FileRecord

TEMP_EXTENSIONS: Set[str] = {".crdownload", ".part", ".tmp", ".download"}
BINARY_SKIP_EXTENSIONS: Set[str] = {
    ".exe", ".msi", ".msix", ".deb", ".dmg", ".iso", ".apk",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".xz",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".flac", ".m4a",
}
COPY_SUFFIX_RE = re.compile(r"(\s*\(\d+\)|\s*-\s*copy)$", re.IGNORECASE)
DOC_IMAGE_HINT_RE = re.compile(r"(?i)(screenshot|snip|whatsapp|scan|receipt|bill|challan|doc|page)")

_OCR_ENGINE = None


def _get_ocr_engine():
    """Lazy-loads RapidOCR once per process only if a scanned PDF or document image is encountered."""
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


def extract_document_content(path: Path, use_ocr: bool = True, max_chars: int = 1500) -> Tuple[str, str]:
    """
    3-Tier Extraction Cascade:
    1. Skip pure binaries/archives/media (0 ms).
    2. Extract native digital text from PDF, DOCX, XLSX, PPTX, Markdown, Code.
    3. Trigger RapidOCR ONLY when a PDF has <30 chars of selectable text or an image looks like a document/screenshot.
    Returns: (extracted_text, extraction_status)
    """
    ext = path.suffix.lower()
    if ext in BINARY_SKIP_EXTENSIONS:
        return "", "SKIPPED_BINARY"

    try:
        if ext in {".txt", ".md", ".csv", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".html", ".sql"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return text[:max_chars], "NATIVE_TEXT"

        if ext == ".pdf":
            import pymupdf
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    return "", "PASSWORD_PROTECTED"
                text = ""
                for page in doc[:2]:
                    text += page.get_text("text") + "\n"
                    if len(text) >= max_chars:
                        break
                clean_text = text.strip()
                if len(clean_text) >= 30:
                    return clean_text[:max_chars], "NATIVE_TEXT"

                # Conditional OCR fallback for scanned PDFs
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
            parts = [p.text for p in doc.paragraphs[:30] if p.text.strip()]
            for table in doc.tables[:2]:
                for row in table.rows[:10]:
                    parts.append(" ".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            parts = []
            for sheet in wb.sheetnames[:2]:
                ws = wb[sheet]
                for row in ws.iter_rows(max_row=20, values_only=True):
                    row_text = " ".join(str(v) for v in row if v is not None)
                    if row_text.strip():
                        parts.append(row_text)
            wb.close()
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(path)
            parts = []
            for slide in prs.slides[:5]:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text.strip())
            return "\n".join(parts)[:max_chars], "NATIVE_TEXT"

        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            if use_ocr and DOC_IMAGE_HINT_RE.search(path.stem):
                ocr_text = _run_ocr_on_image_bytes(path.read_bytes()).strip()
                if ocr_text:
                    return ocr_text[:max_chars], "OCR_USED"
            return "", "SKIPPED_IMAGE_NON_DOC"

    except Exception:
        return "", "CORRUPT_OR_UNREADABLE"

    return "", "UNSUPPORTED_EXT"


def scan_directory(target_dir: Path, extract_text: bool = True, use_ocr: bool = True) -> List[FileRecord]:
    records: List[FileRecord] = []
    for item in sorted(target_dir.iterdir()):
        if item.is_dir() or item.suffix.lower() in TEMP_EXTENSIONS or item.name.startswith("."):
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
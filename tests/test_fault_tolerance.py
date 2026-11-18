import importlib.util
from pathlib import Path
import sys
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.scanner import scan_directory


def _load_classifier():
    spec = importlib.util.spec_from_file_location("rule_cls", ROOT / "strategies/01_rule_based/classifier.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RuleBasedClassifier(content_aware=True)


def test_corrupt_and_password_protected_pdfs(tmp_path: Path):
    # 1. Corrupt 0-byte / garbage PDF
    corrupt_pdf = tmp_path / "broken_download.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4-CORRUPTED-TRUNCATED-BYTES")

    # 2. Password-protected PDF created via PyMuPDF
    encrypted_pdf = tmp_path / "bank_statement_locked.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential Statement")
    doc.save(
        str(encrypted_pdf),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner123",
        user_pw="user123",
    )
    doc.close()

    # 3. Unknown out-of-taxonomy PDF (e.g., appliance manual)
    unknown_pdf = tmp_path / "toaster_manual.pdf"
    doc2 = pymupdf.open()
    p2 = doc2.new_page()
    p2.insert_text((72, 72), "Toast settings dial 1 to 6. Clean crumb tray weekly.")
    doc2.save(str(unknown_pdf))
    doc2.close()

    records = scan_directory(tmp_path, extract_text=True, use_ocr=False)
    by_name = {r.name: r for r in records}

    assert by_name["broken_download.pdf"].extraction_status == "CORRUPT_OR_UNREADABLE"
    assert by_name["bank_statement_locked.pdf"].extraction_status == "PASSWORD_PROTECTED"
    assert by_name["toaster_manual.pdf"].extraction_status == "NATIVE_TEXT"

    classifier = _load_classifier()
    results = {r.file_path.name: r for r in classifier.classify_batch(records)}

    # Pipeline never crashes, flags unreadable files for review, and routes unknown PDFs safely
    assert results["broken_download.pdf"].needs_review is True
    assert results["bank_statement_locked.pdf"].category == "Finance_Invoices"
    assert results["bank_statement_locked.pdf"].needs_review is True
    assert results["toaster_manual.pdf"].category == "Uncategorized_Misc"
import csv
from pathlib import Path
import random
import zipfile
import docx
import openpyxl
from pptx import Presentation
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "datasets" / "unseen_base_200"
STREAM_DIR = ROOT / "datasets" / "unseen_stream_20"


def _make_native_pdf(path: Path, title: str, body: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), title, fontsize=14)
    y = 95
    for line in body.splitlines():
        page.insert_text((50, y), line, fontsize=10)
        y += 16
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: Path, title: str, body: str) -> None:
    """Creates a true image-only PDF (zero selectable text layer) to test Conditional RapidOCR."""
    tmp_doc = pymupdf.open()
    p = tmp_doc.new_page()
    p.insert_text((50, 60), title, fontsize=15)
    y = 95
    for line in body.splitlines():
        p.insert_text((50, y), line, fontsize=11)
        y += 18
    pix = p.get_pixmap(dpi=130)
    img_bytes = pix.tobytes("png")
    tmp_doc.close()

    scan_doc = pymupdf.open()
    sp = scan_doc.new_page(width=pix.width, height=pix.height)
    sp.insert_image(sp.rect, stream=img_bytes)
    scan_doc.save(str(path))
    scan_doc.close()


def _make_docx(path: Path, heading: str, paragraphs: list[str]) -> None:
    d = docx.Document()
    d.add_heading(heading, level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))


def _make_xlsx(path: Path, headers: list[str], rows: list[list]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)
    for r in rows:
        ws.append(r)
    wb.save(str(path))


def _make_pptx(path: Path, slide_title: str, bullets: list[str]) -> None:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = slide_title
    slide.placeholders[1].text = "\n".join(bullets)
    prs.save(str(path))


def _make_png(path: Path, label_text: str, width: int = 800, height: int = 600) -> None:
    doc = pymupdf.open()
    p = doc.new_page(width=width, height=height)
    if label_text:
        p.insert_text((40, 60), label_text, fontsize=16)
    pix = p.get_pixmap(dpi=100)
    pix.save(str(path))
    doc.close()


def _make_zip(path: Path, inner_name: str, inner_text: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, inner_text)


def build_all() -> None:
    rng = random.Random(42)
    for d in (BASE_DIR, STREAM_DIR):
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    base_rows = []

    def add_base(fname: str, cat: str, is_dup: int = 0, canon: str = ""):
        base_rows.append({
            "filename": fname,
            "true_category": cat,
            "is_exact_duplicate": is_dup,
            "canonical_filename": canon or fname,
        })

    # 1. Academic_Study (45 files: PDFs, DOCX, PPTX, Code, Diagrams — including opaque names & collisions)
    topics = [
        ("Operating Systems Deadlock & Bankers Algorithm", "Lecture Module 3: Mutual exclusion, hold and wait, circular wait, resource allocation graph."),
        ("Database Management Normalization BCNF", "Functional dependencies, 3NF decomposition, ACID transaction isolation levels, SQL indexing."),
        ("Computer Networks Subnetting & TCP Congestion", "CIDR block allocation, VLSM host bits, three-way handshake SYN ACK, sliding window flow control."),
        ("Compiler Design LR Parser & AST", "Lexical analysis tokens, context-free grammar, shift-reduce parsing table, semantic syntax tree."),
        ("Machine Learning Gradient Descent & Regularization", "Convex optimization, L1 Lasso vs L2 Ridge penalty, bias-variance tradeoff, cross-validation."),
    ]
    for i in range(1, 46):
        title, body = topics[i % len(topics)]
        # Include realistic collision words (like 'internship' or 'fee') inside academic text
        collision_note = "Note: Research internship credit rubric is evaluated in Semester VI." if i % 7 == 0 else ""
        if i <= 18:
            fname = f"doc_ref_{8000 + i}.pdf" if i % 3 == 0 else f"CS30{i % 9}_Unit{(i % 5)+1}_Notes.pdf"
            _make_native_pdf(BASE_DIR / fname, title, f"{body}\n{collision_note}\nProblem Set {i}")
        elif i <= 28:
            fname = f"Research_Draft_v{i}.docx"
            _make_docx(BASE_DIR / fname, title, [body, collision_note, "Abstract: Empirical evaluation of algorithms."])
        elif i <= 35:
            fname = f"Week_{i}_Slides.pptx"
            _make_pptx(BASE_DIR / fname, title, [body, "Key Exam Theorem & Proof"])
        elif i <= 41:
            ext = ".cpp" if i % 2 == 0 else ".py"
            fname = f"lab_solution_{i}{ext}"
            (BASE_DIR / fname).write_text(f"// {title}\n#include <iostream>\nint main() {{ return 0; }}\n", encoding="utf-8")
        else:
            fname = f"circuit_logic_fig_{i}.png"
            _make_png(BASE_DIR / fname, f"4-to-16 MULTIPLEXER LOGIC DIAGRAM\nBoolean Equation F = A'B + CD")
        add_base(fname, "Academic_Study")

    # 2. Finance_Invoices (35 files: native PDFs, scanned image-only PDFs, opaque numeric filenames)
    vendors = [
        ("TAX INVOICE - Cloud Hosting Services", "GSTIN: 09AAACR5055K1Z2 | Subtotal: INR 4,200.00 | CGST 9%: 378.00 | Total Amount Due: INR 4,956.00"),
        ("E-CHALLAN TRAFFIC VIOLATION RECEIPT", "Vehicle Reg: UP16AB1234 | Penalty Fine Amount: Rs. 1,000 | Payment Status: PAID | Transaction ID: TXN99281"),
        ("UNIVERSITY TUITION FEE RECEIPT", "Semester Fee Challan | Amount Paid: INR 85,000 | UTR Reference: HDFC0009182 | Billing Desk"),
        ("RETAIL ORDER INVOICE & BILLING STATEMENT", "Invoice No: INV-2026-441 | Net Payable: Rs. 2,349.00 | Paid via UPI | Thank you for your purchase"),
    ]
    for i in range(1, 36):
        title, body = vendors[i % len(vendors)]
        # Include realistic collision word ('unit price' or 'module') inside invoices
        body_with_trap = f"{body}\nItem Description: Enterprise Software License (1 Unit) - Billing Cycle Monthly"
        if i <= 20:
            fname = f"{710000 + i}.pdf" if i % 2 == 0 else f"INV_2026_{100 + i}.pdf"
            _make_native_pdf(BASE_DIR / fname, title, body_with_trap)
        else:
            fname = f"scan_doc_{i}.pdf"
            _make_scanned_pdf(BASE_DIR / fname, title, body_with_trap)
        add_base(fname, "Finance_Invoices")

    # 3. Career_Identity (20 files: resumes, offer letters, ID cards, scanned signatures)
    career_docs = [
        ("CURRICULUM VITAE - SOFTWARE ENGINEER", "Education: B.Tech Computer Science | Skills: Python, C++, React, SQL | Experience: Backend Intern"),
        ("INTERNSHIP OFFER LETTER", "Dear Candidate, We are pleased to offer you the position of Software Engineering Intern. Stipend: INR 25,000/month."),
        ("GOVERNMENT IDENTITY & PAN VERIFICATION", "Permanent Account Number | Date of Birth | Candidate Signature & Identity Proof"),
    ]
    for i in range(1, 21):
        title, body = career_docs[i % len(career_docs)]
        if i <= 12:
            fname = f"Candidate_Profile_{i}.pdf" if i % 2 == 0 else f"Resume_v{i}.docx"
            if fname.endswith(".pdf"):
                _make_native_pdf(BASE_DIR / fname, title, body)
            else:
                _make_docx(BASE_DIR / fname, title, [body])
        else:
            fname = f"WhatsApp_Scan_ID_{i}.png"
            _make_png(BASE_DIR / fname, f"CANDIDATE ID CARD & SIGNATURE\nRoll / Registration No: ID2026{i}")
        add_base(fname, "Career_Identity")

    # 4. Software_Installers (22 files)
    installers = ["DockerDesktopInstaller", "VSCodeUserSetup-x64", "Git-2.47.0-64-bit", "OllamaSetup", "node-v22.19.0-x64", "Postman-win64"]
    for i in range(1, 23):
        ext = ".msi" if i % 4 == 0 else ".exe"
        fname = f"{installers[i % len(installers)]}_{i}{ext}"
        (BASE_DIR / fname).write_bytes(b"MZ\x90\x00" + rng.randbytes(256))
        add_base(fname, "Software_Installers")

    # 5. Archives_Compressed (18 files)
    archives = ["dataset_backup", "project_release", "firmware_drivers", "fonts_bundle", "source_archive"]
    for i in range(1, 19):
        fname = f"{archives[i % len(archives)]}_v{i}.zip"
        _make_zip(BASE_DIR / fname, "README.txt", f"Archive package {i}")
        add_base(fname, "Archives_Compressed")

    # 6. Spreadsheets_Data (18 files)
    for i in range(1, 19):
        if i % 2 == 0:
            fname = f"experiment_metrics_{i}.xlsx"
            _make_xlsx(BASE_DIR / fname, ["id", "precision", "recall", "f1"], [[1, 0.88, 0.84, 0.86], [2, 0.91, 0.87, 0.89]])
        else:
            fname = f"sensor_telemetry_{i}.csv"
            (BASE_DIR / fname).write_text("timestamp,cpu_pct,ram_mb,latency_ms\n2026-11-01,24.5,142.1,12.4\n", encoding="utf-8")
        add_base(fname, "Spreadsheets_Data")

    # 7. Media_Photos_Graphics (14 files)
    for i in range(1, 15):
        fname = f"wallpaper_4k_render_{i}.jpg" if i % 2 == 0 else f"IMG_202611_{100+i}.png"
        _make_png(BASE_DIR / fname, "", width=1280, height=720)
        add_base(fname, "Media_Photos_Graphics")

    # 8. Media_Screenshots (8 files)
    for i in range(1, 9):
        fname = f"Screenshot_2026-11-18_14{i:02d}10.png"
        _make_png(BASE_DIR / fname, f"Terminal Output - Build Succeeded in 1.{i}s", width=1366, height=768)
        add_base(fname, "Media_Screenshots")

    # 9. Media_Audio_Video (6 files)
    for i in range(1, 7):
        ext = ".mkv" if i % 2 == 0 else ".mp4"
        fname = f"screen_recording_demo_{i}{ext}"
        (BASE_DIR / fname).write_bytes(b"\x00\x00\x00\x18ftypmp42" + rng.randbytes(256))
        add_base(fname, "Media_Audio_Video")

    # 10. Uncategorized_Misc (8 out-of-taxonomy files: appliance manuals, recipes, poetry)
    misc_docs = [
        ("MICROWAVE OVEN USER MANUAL", "Defrost timer knob, turntable glass plate cleaning instructions, wattage 900W."),
        ("SOURDOUGH BREAD BAKING RECIPE", "Ingredients: 500g bread flour, 350g water, 100g active starter, 10g sea salt. Bake at 230C."),
    ]
    for i in range(1, 9):
        title, body = misc_docs[i % len(misc_docs)]
        fname = f"misc_note_{i}.pdf"
        _make_native_pdf(BASE_DIR / fname, title, body)
        add_base(fname, "Uncategorized_Misc")

    # 11. Exact SHA-256 Duplicates (6 files -> total 200 base files!)
    dup_categories = [
        "Finance_Invoices",
        "Academic_Study",
        "Archives_Compressed",
        "Software_Installers",
        "Spreadsheets_Data",
        "Career_Identity",
    ]
    for cat in dup_categories:
        orig_name = next(r["filename"] for r in base_rows if r["true_category"] == cat)
        orig_path = BASE_DIR / orig_name
        dup_name = f"{orig_path.stem} (1){orig_path.suffix}"
        (BASE_DIR / dup_name).write_bytes(orig_path.read_bytes())
        add_base(dup_name, cat, is_dup=1, canon=orig_name)

    with open(ROOT / "datasets" / "unseen_base_ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_category", "is_exact_duplicate", "canonical_filename"])
        w.writeheader()
        w.writerows(base_rows)

    # =========================================================================
    # STREAM BATCH (+20 UNSEEN FILES):
    # - 12 files from EXISTING categories with unseen phrasings
    # - 8 files from 2 BRAND-NEW unseen categories (Medical_Health, Travel_Bookings)
    # =========================================================================
    stream_rows = []

    def add_stream(fname: str, cat: str, is_new_cat: int):
        stream_rows.append({"filename": fname, "true_category": cat, "is_new_category": is_new_cat})

    # 6 unseen-phrasing Academic files
    for i in range(1, 7):
        fname = f"viva_rubric_eval_{i}.pdf"
        _make_native_pdf(STREAM_DIR / fname, "LABORATORY VIVA-VOCE & PRACTICAL RUBRIC", "Experiment evaluation, viva questions, faculty assessment sheet.")
        add_stream(fname, "Academic_Study", 0)

    # 6 unseen-phrasing Finance files
    for i in range(1, 7):
        fname = f"credit_memo_txn_{i}.pdf"
        _make_native_pdf(STREAM_DIR / fname, "GST CREDIT NOTE & REFUND VOUCHER", "Taxable value adjustment, CGST/SGST reversal, credit note settlement.")
        add_stream(fname, "Finance_Invoices", 0)

    # 4 BRAND-NEW Category 1: Medical_Health
    for i in range(1, 5):
        fname = f"lab_diagnostic_cbc_{i}.pdf"
        _make_native_pdf(STREAM_DIR / fname, "PATHOLOGY DIAGNOSTIC REPORT - COMPLETE BLOOD COUNT", "Hemoglobin 14.2 g/dL, WBC Count, Platelet Count, Prescribed by Dr. Verma.")
        add_stream(fname, "Medical_Health", 1)

    # 4 BRAND-NEW Category 2: Travel_Bookings
    for i in range(1, 5):
        fname = f"eticket_pnr_booking_{i}.pdf"
        _make_native_pdf(STREAM_DIR / fname, "FLIGHT E-TICKET & BOARDING PASS ITINERARY", "PNR: X9K2LP | Departure: DEL Terminal 3 -> Arrival: BLR | Seat 14A.")
        add_stream(fname, "Travel_Bookings", 1)

    with open(ROOT / "datasets" / "unseen_stream_ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_category", "is_new_category"])
        w.writeheader()
        w.writerows(stream_rows)

    print(f"Generated {len(base_rows)} unseen base files and {len(stream_rows)} incremental stream files.")


if __name__ == "__main__":
    build_all()
import csv
import hashlib
from collections import Counter
from pathlib import Path

RAW_DIR = Path("datasets/real_wild_downloads")
OUT_CSV = Path("datasets/real_wild_ground_truth.csv")

LABELS = {
    # Academic_Study (21 unique files)
    "1409.3215.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "1512.03385.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "1706.03762.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "1810.04805.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "2005.14165.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "2103.00020.pdf": ("Academic_Study", "arxiv.org", "blind_test"),
    "ADA Major 2025.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "CourseCurriculum.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "DECO Major 2024.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "E-book of OOP using C++ (E Balagurusamy).pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "ETN Major 2025.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "Hari Arora AM3.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "hj.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "Linear and Binary Search Algo....pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "Module3+4+5_2026-27_CSE303_ADA assignment.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "OOPs C++ Major 2023.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "Problem-statements (1).pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "rfc7540.txt": ("Academic_Study", "rfc-editor.org", "blind_test"),
    "rfc9000.txt": ("Academic_Study", "rfc-editor.org", "blind_test"),
    "rfc9110.txt": ("Academic_Study", "rfc-editor.org", "blind_test"),
    "sample_term_paper[1].docx": ("Academic_Study", "downloads_untouched", "blind_test"),
    "wpr 2 latest.pdf": ("Academic_Study", "downloads_untouched", "blind_test"),
    "WPR Format Term Paper.docx": ("Academic_Study", "downloads_untouched", "blind_test"),

    # Finance_Invoices (15 unique files)
    "cash-payment-receipt.pdf": ("Finance_Invoices", "public_sample", "blind_test"),
    "cash-receipt-form.pdf": ("Finance_Invoices", "vertex42.com", "blind_test"),
    "Downloadable-PDF-Invoices-Add-On-Samples.pdf": ("Finance_Invoices", "public_sample", "blind_test"),
    "f1040.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "f1040sc.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "f1099nec.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "f941.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "fss4.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "fw2.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "fw9.pdf": ("Finance_Invoices", "irs.gov", "blind_test"),
    "invoice.html": ("Finance_Invoices", "github.com", "blind_test"),
    "pge sample utility bill.pdf": ("Finance_Invoices", "pge.com", "blind_test"),
    "sample-pdf-invoice.pdf": ("Finance_Invoices", "public_sample", "blind_test"),
    "Sample_Utility_Bill.pdf": ("Finance_Invoices", "crwwd.com", "blind_test"),
    "solar-choice-bill-sample.pdf": ("Finance_Invoices", "public_sample", "blind_test"),
    "wordpress-pdf-invoice-plugin-sample.pdf": ("Finance_Invoices", "slicedinvoices.com", "blind_test"),
    "Your Accounts_XXXXXX170_13112026.PDF": ("Finance_Invoices", "downloads_untouched", "blind_test"),

    # Career_Identity (10 unique files)
    "_Achintya_Pathak_Unpaid Internship_Offer_Letter_20261110.pdf": ("Career_Identity", "downloads_untouched", "blind_test"),
    "Achintya_Pathak_CV.pdf": ("Career_Identity", "downloads_untouched", "blind_test"),
    "AchintyaEIS (1).pdf": ("Career_Identity", "downloads_untouched", "blind_test"),
    "awesome_cv_coverletter.pdf": ("Career_Identity", "github.com", "blind_test"),
    "awesome_cv_full.pdf": ("Career_Identity", "github.com", "blind_test"),
    "awesome_cv_resume.pdf": ("Career_Identity", "github.com", "blind_test"),
    "Examination Admit Card.pdf": ("Career_Identity", "downloads_untouched", "blind_test"),
    "offer letter.pdf": ("Career_Identity", "downloads_untouched", "blind_test"),
    "sourabh_bajaj_resume.pdf": ("Career_Identity", "github.com", "blind_test"),
    "uttarpradesh.amity-DGMST-2025A2345924002IJANUARY.pdf": ("Career_Identity", "downloads_untouched", "blind_test"),

    # Spreadsheets_Data (12 unique files)
    "annual.csv": ("Spreadsheets_Data", "datasets_github", "blind_test"),
    "cats_uk.csv": ("Spreadsheets_Data", "tidytuesday", "blind_test"),
    "chocolate.csv": ("Spreadsheets_Data", "tidytuesday", "blind_test"),
    "coffee_ratings.csv": ("Spreadsheets_Data", "tidytuesday", "blind_test"),
    "diamonds.csv": ("Spreadsheets_Data", "seaborn_data", "blind_test"),
    "gdp.csv": ("Spreadsheets_Data", "datasets_github", "blind_test"),
    "iris.csv": ("Spreadsheets_Data", "seaborn_data", "blind_test"),
    "olympics.csv": ("Spreadsheets_Data", "tidytuesday", "blind_test"),
    "penguins.csv": ("Spreadsheets_Data", "seaborn_data", "blind_test"),
    "population.csv": ("Spreadsheets_Data", "datasets_github", "blind_test"),
    "titanic.csv": ("Spreadsheets_Data", "seaborn_data", "blind_test"),

    # Software_Installers (11 unique files)
    "click-8.1.7-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "fastapi-0.115.4-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "Git-2.50.1-64-bit.exe": ("Software_Installers", "downloads_untouched", "blind_test"),
    "httpx-0.27.2-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "Install VALORANT.exe": ("Software_Installers", "downloads_untouched", "blind_test"),
    "msys2-x86_64-20250622.exe": ("Software_Installers", "downloads_untouched", "blind_test"),
    "pytest-8.3.3-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "requests-2.32.3-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "rich-13.9.4-py3-none-any.whl": ("Software_Installers", "pypi.org", "blind_test"),
    "VSCodeUserSetup-x64-1.95.3.exe": ("Software_Installers", "downloads_untouched", "blind_test"),
    "WirelessLan_ROG_Realtek_B_V6001.15.155.1Sub2_41146.exe": ("Software_Installers", "downloads_untouched", "blind_test"),

    # Archives_Compressed (8 unique files)
    "click-8.1.7.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "flask-3.0.3.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "httpx-0.27.2.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "jq-1.7.1.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "pytest-8.3.3.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "requests-2.32.3.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "rich-13.9.4.zip": ("Archives_Compressed", "github.com", "blind_test"),
    "ripgrep-14.1.1.zip": ("Archives_Compressed", "github.com", "blind_test"),

    # Media_Visuals (5 unique files)
    "500px-Camponotus_flavomarginatus_ant.jpg": ("Media_Visuals", "wikimedia.org", "blind_test"),
    "500px-Cat03.jpg": ("Media_Visuals", "wikimedia.org", "blind_test"),
    "500px-Gull_portrait_ca_usa.jpg": ("Media_Visuals", "wikimedia.org", "blind_test"),
    "500px-Image_created_with_a_mobile_phone.png": ("Media_Visuals", "wikimedia.org", "blind_test"),
    "500px-PNG_transparency_demonstration_1.png": ("Media_Visuals", "wikimedia.org", "blind_test"),

    # Travel_Bookings — Held-out unseen category (2 support + 3 disjoint query = 5 unique files)
    "000262548.pdf": ("Travel_Bookings", "public_sample", "stage2_support"),
    "Flight_ETicket_1111200071731.pdf": ("Travel_Bookings", "public_sample", "stage2_support"),
    "2742721020156116.pdf": ("Travel_Bookings", "irctc_sample", "stage2_query"),
    "Sample-Travel-Itinerary-2.pdf": ("Travel_Bookings", "public_sample", "stage2_query"),
    "sample.pdf": ("Travel_Bookings", "public_sample", "stage2_query"),

    # Medical_Health — Held-out unseen category (2 support + 4 disjoint query = 6 unique files)
    "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf": ("Medical_Health", "drlogy.com", "stage2_support"),
    "complete-blood-count-(cbc).pdf": ("Medical_Health", "public_sample", "stage2_support"),
    "PK0016.pdf": ("Medical_Health", "public_sample", "stage2_query"),
    "sterling-accuris-pathology-sample-report-unlocked.pdf": ("Medical_Health", "sterlingaccuris.com", "stage2_query"),
    "Wellwise Platinum Profile - 5048.pdf": ("Medical_Health", "public_sample", "stage2_query"),
    "Z021.pdf": ("Medical_Health", "public_sample", "stage2_query"),
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    files = sorted([p for p in RAW_DIR.iterdir() if p.is_file()])
    rows = []
    unmapped = []
    for p in files:
        if p.name not in LABELS:
            unmapped.append(p.name)
            continue
        cat, domain, split = LABELS[p.name]
        rows.append({
            "filename": p.name,
            "relative_path": p.name,
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "true_category": cat,
            "source_domain": domain,
            "split": split,
        })

    if unmapped:
        print("UNMAPPED FILES:", unmapped)
        return

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Locked {OUT_CSV} with {len(rows)} unique, verified ground-truth files.")
    print("\nBreakdown by (split, true_category):")
    for (sp, cat), cnt in sorted(Counter((r["split"], r["true_category"]) for r in rows).items()):
        print(f"  {sp:15s} | {cat:22s} | {cnt}")

if __name__ == "__main__":
    main()

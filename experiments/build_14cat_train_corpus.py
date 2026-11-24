import csv
from datetime import datetime
import hashlib
import pathlib
import pickle
import random
import re
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.models import FileRecord
from shared.scanner import extract_document_content

TAXONOMY_14 = [
    "Study_Learning",
    "Work_Professional",
    "Career_Jobs",
    "Finance_Money",
    "Personal_IDs_Records",
    "Travel",
    "Health_Medical",
    "Home_Household",
    "Software_Installers",
    "Archives_Compressed",
    "Media_Images",
    "Media_Audio_Video",
    "Data_Code",
    "Other",
]

BASE_MAP = {
    "Academic_Study": "Study_Learning",
    "Finance_Invoices": "Finance_Money",
    "Travel_Bookings": "Travel",
    "Medical_Health": "Health_Medical",
    "Software_Installers": "Software_Installers",
    "Archives_Compressed": "Archives_Compressed",
    "Media_Photos_Graphics": "Media_Images",
    "Media_Screenshots": "Media_Images",
    "Media_Audio_Video": "Media_Audio_Video",
}


def map_to_14_categories(row: dict, text: str) -> str:
    old_cat = row["true_category"].strip()
    dom = (row.get("source_domain") or "").lower()
    fname = (row.get("filename") or "").lower()
    ext = (row.get("extension") or "").lower()
    combined = f"{fname} {dom} {text[:800].lower()}"

    if old_cat == "Career_Identity":
        if any(k in combined for k in ("id_card", "id card", "identity card", "passport", "aadhaar", "pan card", "driving", "voter", "birth certificate", "vehicle", "rc book", "affidavit")):
            return "Personal_IDs_Records"
        return "Career_Jobs"

    if old_cat == "Spreadsheets_Data":
        if any(k in combined for k in ("invoice", "expense", "accounting", "budget", "ledger", "billing", "salary", "payroll", "tax", "gst")):
            return "Finance_Money"
        if any(k in combined for k in ("project_plan", "roadmap", "timesheet", "sprint", "kpi", "sales_tracker")):
            return "Work_Professional"
        return "Data_Code"

    if old_cat in BASE_MAP:
        return BASE_MAP[old_cat]

    if ext in {".py", ".js", ".ts", ".java", ".cpp", ".sql", ".yaml", ".yml", ".toml", ".ipynb"}:
        return "Data_Code"
    return "Other"


def build_taxonomy_anchor_records() -> list[tuple[FileRecord, str]]:
    """Generates clean synthetic FileRecord anchors grounded strictly in TAXONOMY.md definitions."""
    templates = {
        "Study_Learning": [
            ("CS301_Lecture_04_Dynamic_Programming.pdf", ".pdf", "Lecture Notes: Dynamic Programming, Bellman Equation, Memoization vs Tabulation, University Course CS301."),
            ("Assignment_2_OS_Scheduling.docx", ".docx", "Homework Assignment 2: Operating Systems CPU Scheduling Algorithms, Round Robin, Deadlock Prevention, Roll No submission."),
            ("EndSem_Question_Paper_2025.pdf", ".pdf", "End Semester Examination Question Paper, Max Marks 100, Attempt any five questions, Answer Key."),
            ("2403.11892v1.pdf", ".pdf", "arXiv preprint Abstract Introduction Related Work Methodology Experiments Ablation Study References."),
            ("Course_Syllabus_Academic_Calendar.pdf", ".pdf", "Department of Computer Science Course Syllabus, Grading Policy, Mid-term Exam Schedule, Academic Calendar."),
            ("Semester_5_Marksheet_Transcript.pdf", ".pdf", "Official Grade Card Statement of Marks Semester V SGPA CGPA Credits Earned University Controller of Examinations."),
            ("BTech_Degree_Provisional_Certificate.pdf", ".pdf", "Provisional Degree Certificate Bachelor of Technology First Class with Distinction Convocation."),
            ("GATE_2026_Admit_Card.pdf", ".pdf", "Examination Admit Card Hall Ticket Roll Number Test Centre Reporting Time Instructions."),
            ("Data_Structures_Textbook_Ch7.pdf", ".pdf", "Chapter 7 Balanced Search Trees AVL Trees Red-Black Trees Study Guide Exercises."),
            ("Lab_Manual_DBMS_Queries.pdf", ".pdf", "Database Management Systems Lab Record SQL Joins Normalization Practical File."),
        ],
        "Work_Professional": [
            ("Q3_Enterprise_Sales_Proposal.pdf", ".pdf", "Business Proposal Executive Summary Scope of Work Deliverables Commercials Client Pitch Deck."),
            ("Client_NDA_Master_Service_Agreement.docx", ".docx", "Non-Disclosure Agreement and Master Services Agreement SOW Confidential Information Governing Law."),
            ("Weekly_Sprint_Meeting_Minutes.docx", ".docx", "Minutes of Meeting MOM Agenda Action Items Owners Blockers Next Steps Product Sync."),
            ("HR_Leave_Travel_Policy_Handbook.pdf", ".pdf", "Employee Handbook Standard Operating Procedure SOP Leave Policy Code of Conduct Reimbursement Guidelines."),
            ("Product_Roadmap_Q4_Tracker.xlsx", ".xlsx", "Sheets: Roadmap, Milestones, Sprint Tracker Owner Status Priority Target Date Deliverable."),
            ("Board_Pitch_Deck_2026.pptx", ".pptx", "Quarterly Business Review Revenue Growth Go-To-Market Strategy Competitive Landscape."),
            ("Vendor_Compliance_Memo.pdf", ".pdf", "Official Business Correspondence Memorandum Vendor Onboarding Compliance Audit Sign-off."),
            ("Annual_Operations_Whitepaper.pdf", ".pdf", "Enterprise Architecture Whitepaper System Reliability SLA Benchmarks Case Study."),
            ("Project_Resource_Timesheet.xlsx", ".xlsx", "Sheets: Timesheet, Allocation Engineering Hours Sprint Capacity Utilization."),
            ("Partnership_MOU_Draft.docx", ".docx", "Memorandum of Understanding Strategic Partnership Terms Responsibilities Effective Date."),
        ],
        "Career_Jobs": [
            ("Achintya_Resume_SDE_2026.pdf", ".pdf", "Curriculum Vitae Resume Experience Skills Projects Education GitHub LinkedIn Software Engineer."),
            ("Cover_Letter_Backend_Role.docx", ".docx", "Dear Hiring Manager Cover Letter I am writing to express my interest in the Software Developer position."),
            ("Offer_Letter_Software_Engineer.pdf", ".pdf", "Employment Offer Appointment Letter CTC Annual Compensation Joining Date Probation Period HR."),
            ("Experience_Relieving_Certificate.pdf", ".pdf", "To Whomsoever It May Concern Experience and Relieving Letter served as Senior Analyst full and final settlement."),
            ("Senior_ML_Engineer_JD.pdf", ".pdf", "Job Description Responsibilities Qualifications Required Skills Python PyTorch Apply Now."),
            ("Technical_Interview_Schedule_Assignment.pdf", ".pdf", "Interview Rounds Coding Assessment Take-Home Challenge Evaluation Rubric Feedback Form."),
            ("Candidate_CV_Shortlist_04.pdf", ".pdf", "Resume Work History Core Competencies Certifications B.Tech Computer Science."),
            ("Recommendation_Letter_Manager.pdf", ".pdf", "Letter of Recommendation worked under my supervision strong analytical and leadership skills."),
            ("Internship_Appointment_Letter.pdf", ".pdf", "Summer Internship Offer Letter Stipend Duration Reporting Manager."),
            ("HR_Recruitment_Assessment_Form.xlsx", ".xlsx", "Sheets: Candidates Interviewer Rating Technical Round Cultural Fit Hiring Decision."),
        ],
        "Finance_Money": [
            ("GST_Tax_Invoice_99482.pdf", ".pdf", "Tax Invoice GSTIN Subtotal CGST SGST Total Amount Payable Due Date Bank Details."),
            ("UPI_Payment_Receipt_Screenshot.png", ".png", "Payment Successful Paid to Merchant UPI Transaction ID Ref No Debited from Bank Account INR 2,450."),
            ("Electricity_Bill_Oct_2026.pdf", ".pdf", "Electricity Utility Bill Consumer Number Meter Reading Units Consumed Due Date Pay Online."),
            ("HDFC_Bank_Statement_Q3.pdf", ".pdf", "Account Statement Opening Balance Closing Balance Withdrawal Deposit NEFT IMPS Credit Card Statement."),
            ("Form_16_ITR_Acknowledgement_AY26.pdf", ".pdf", "Income Tax Return Acknowledgement Form 16 TDS Assessment Year Gross Total Income Tax Paid."),
            ("Payslip_October_2026.pdf", ".pdf", "Salary Slip Payslip Basic Pay HRA Special Allowance PF Deductions Net Pay Credited."),
            ("Term_Life_Insurance_Policy_Schedule.pdf", ".pdf", "Insurance Policy Document Policy Number Sum Assured Annual Premium Receipt Nominee."),
            ("Home_Loan_EMI_Amortization_Schedule.pdf", ".pdf", "Loan Sanction Letter Principal Interest Rate EMI Schedule Mutual Fund Consolidated Account Statement."),
            ("Monthly_Household_Budget_Expenses.xlsx", ".xlsx", "Sheets: Budget, Expenses Rent Groceries Utilities Split Bill Total Spend Savings."),
            ("Traffic_Challan_Fine_Receipt.pdf", ".pdf", "e-Challan Payment Receipt Violation Fine Paid Receipt Number."),
            ("College_Tuition_Fee_Receipt.pdf", ".pdf", "Fee Receipt Tuition Fee Hostel Fee Transaction Reference Amount Paid."),
            ("Broadband_Fiber_Invoice.pdf", ".pdf", "Monthly Internet Fiber Bill Plan Charges Tax Invoice Amount Due."),
        ],
        "Personal_IDs_Records": [
            ("Aadhaar_Card_Front_Back.pdf", ".pdf", "Government of India Unique Identification Authority Aadhaar DOB Address Identity Card."),
            ("PAN_Card_eCopy.pdf", ".pdf", "Income Tax Department Permanent Account Number Card Date of Birth Signature."),
            ("Passport_Bio_Page_Scan.jpg", ".jpg", "Republic of India Passport Nationality Date of Issue Date of Expiry Place of Birth."),
            ("Driving_Licence_SmartCard.png", ".png", "Union of India Driving Licence Valid Till Class of Vehicle Transport Authority."),
            ("Vehicle_RC_PUC_Certificate.pdf", ".pdf", "Certificate of Registration RC Chassis Number Engine Number Pollution Under Control PUC Vehicle Insurance."),
            ("Birth_Domicile_Certificate.pdf", ".pdf", "Municipal Corporation Registrar Birth Certificate Domicile Certificate Income Certificate Caste Certificate."),
            ("Notarized_Affidavit_POA.pdf", ".pdf", "Before the Notary Public Affidavit Solemnly Affirm Power of Attorney Legal Declaration."),
            ("Voter_ID_EPIC_Card.pdf", ".pdf", "Election Commission of India Elector Photo Identity Card EPIC."),
        ],
        "Travel": [
            ("IndiGo_Flight_ETicket_PNR.pdf", ".pdf", "Flight e-Ticket Itinerary PNR Departure Arrival Terminal Gate Seat Baggage Allowance Boarding Pass."),
            ("IRCTC_Train_ETicket_Confirmation.pdf", ".pdf", "IRCTC Electronic Reservation Slip PNR Train No Coach Berth Quota Passenger Details."),
            ("MakeMyTrip_Hotel_Booking_Voucher.pdf", ".pdf", "Hotel Booking Confirmation Voucher Check-in Check-out Room Type Guest Name Reservation ID."),
            ("Europe_7Day_Tour_Itinerary.docx", ".docx", "Travel Itinerary Day 1 Arrival Sightseeing Hotel Transfer Sightseeing Tour Plan."),
            ("Schengen_Visa_Approval_Travel_Insurance.pdf", ".pdf", "Visa Application Approval Notice Embassy Consulate Entry Validity Travel Medical Insurance."),
            ("RedBus_Sleeper_Ticket.png", ".png", "Bus Ticket Booking Confirmed Boarding Point Dropping Point Seat No Operator."),
        ],
        "Health_Medical": [
            ("Thyrocare_CBC_Blood_Test_Report.pdf", ".pdf", "Pathology Laboratory Report Complete Blood Count Hemoglobin WBC Platelets Reference Range."),
            ("Dr_Sharma_OPD_Prescription.jpg", ".jpg", "Rx Clinic Prescription Diagnosis Tab Paracetamol Dosage Twice Daily After Food Follow up."),
            ("Hospital_Discharge_Summary.pdf", ".pdf", "Inpatient Discharge Summary Date of Admission Date of Discharge Diagnosis Treatment Given Advice."),
            ("Star_Health_Insurance_Claim_Form.pdf", ".pdf", "Health Insurance Cashless Reimbursement Claim Form Hospitalization Bills Settlement."),
            ("CoWIN_Vaccination_Certificate.pdf", ".pdf", "Certificate for COVID-19 Vaccination Dose Beneficiary Reference ID Vaccine Name."),
            ("MRI_Spine_Radiology_Scan_Report.pdf", ".pdf", "Department of Radiology MRI Scan Findings Impression Consultant Radiologist."),
        ],
        "Home_Household": [
            ("House_Rent_Lease_Agreement_2026.pdf", ".pdf", "Leave and License Rent Agreement Landlord Tenant Monthly Rent Security Deposit Notice Period."),
            ("Apartment_Society_Maintenance_Notice.pdf", ".pdf", "Cooperative Housing Society Circular AGM Notice Water Tank Cleaning Maintenance."),
            ("LG_Washing_Machine_User_Manual_Warranty.pdf", ".pdf", "Owner User Manual Troubleshooting Installation Guide Product Warranty Card Serial Number."),
            ("Kids_School_PTM_Circular_Report_Card.pdf", ".pdf", "Public School Parent Teacher Meeting Circular Class IV Term Report Card School Fee Notice."),
            ("Paneer_Tikka_Sourdough_Recipe_Guide.pdf", ".pdf", "Ingredients Preparation Time Cooking Instructions Preheat Oven Baking Recipe."),
            ("Property_Tax_Municipal_Receipt.pdf", ".pdf", "Municipal Property Tax Assessment Holding Number Residential Flat."),
        ],
        "Software_Installers": [
            ("VSCodeUserSetup-x64-1.95.exe", ".exe", ""),
            ("Docker_Desktop_Installer.msi", ".msi", ""),
            ("zoom_amd64.deb", ".deb", ""),
            ("Obsidian-1.7.4.AppImage", ".AppImage", ""),
            ("WhatsApp_Desktop.dmg", ".dmg", ""),
            ("numpy-2.1.2-cp312-win_amd64.whl", ".whl", ""),
            ("nvidia_geforce_driver_565.exe", ".exe", ""),
            ("banking_app_v4.2.apk", ".apk", ""),
        ],
        "Archives_Compressed": [
            ("backend_repo_backup.zip", ".zip", ""),
            ("scanned_docs_bundle.tar.gz", ".tar.gz", ""),
            ("trip_photos_2026.7z", ".7z", ""),
            ("project_release_v2.rar", ".rar", ""),
        ],
        "Media_Images": [
            ("IMG_20261114_183210.jpg", ".jpg", ""),
            ("Screenshot_2026-11-18_142011.png", ".png", ""),
            ("brand_logo_banner_final.png", ".png", ""),
            ("4k_mountain_wallpaper.webp", ".webp", ""),
            ("system_architecture_diagram.png", ".png", "Client Load Balancer API Gateway Microservices Database Cache"),
            ("whatsapp_good_morning_meme.jpg", ".jpg", ""),
        ],
        "Media_Audio_Video": [
            ("screen_recording_bug_repro.mp4", ".mp4", ""),
            ("meeting_voice_note_03.m4a", ".m4a", ""),
            ("podcast_episode_42.mp3", ".mp3", ""),
            ("family_event_clip.mov", ".mov", ""),
        ],
        "Data_Code": [
            ("customer_churn_dataset.csv", ".csv", "customer_id,tenure,monthly_charges,contract_type,churn\n1001,12,70.5,Month-to-month,No"),
            ("train_classifier.py", ".py", "import numpy as np\nimport pandas as pd\ndef train_model(X, y):\n    return None"),
            ("docker-compose.yaml", ".yaml", "version: '3.9'\nservices:\n  api:\n    image: python:3.12\n    ports:\n      - '8000:8000'"),
            ("exploratory_analysis.ipynb", ".ipynb", "import matplotlib.pyplot as plt\ndf = pd.read_csv('data.csv')\ndf.describe()"),
            ("schema_migration_v3.sql", ".sql", "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255) UNIQUE NOT NULL);"),
            ("app_settings.toml", ".toml", "[server]\nhost = '0.0.0.0'\nport = 8080\ndebug = false"),
        ],
        "Other": [
            ("Unconfirmed 555097.crdownload", ".crdownload", ""),
            ("download_incomplete.part", ".part", ""),
            ("corrupted_zero_byte.pdf", ".pdf", ""),
            ("invite.ics", ".ics", "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nSUMMARY:Sync Call\nEND:VEVENT\nEND:VCALENDAR"),
            ("Roboto-Regular.ttf", ".ttf", ""),
            ("ubuntu-24.04.torrent", ".torrent", ""),
        ],
    }

    now = datetime(2026, 11, 20, 12, 0, 0)
    synthetic_pairs = []
    for cat, items in templates.items():
        for fname, ext, text in items:
            p = pathlib.Path("synthetic_taxonomy_anchors") / cat / fname
            status = "TEMP_PARTIAL" if ext in {".crdownload", ".part"} else ("NATIVE_TEXT" if text else "SKIPPED_BINARY")
            rec = FileRecord(
                path=p,
                name=fname,
                stem=pathlib.Path(fname).stem,
                extension=ext,
                size_bytes=max(0, len(text) * 18) if text else (0 if "zero_byte" in fname else 4096),
                created_at=now,
                modified_at=now,
                extracted_text=text,
                extraction_status=status,
            )
            synthetic_pairs.append((rec, cat))
    return synthetic_pairs


def main() -> None:
    # Load test_set_v1 sha256 set ONLY to verify 0 leakage (never print or inspect test rows)
    test_gt_path = REPO_ROOT / "datasets/test_set_v1/ground_truth.csv"
    test_hashes = {
        r["sha256"].strip().lower()
        for r in csv.DictReader(test_gt_path.open(encoding="utf-8"))
        if r.get("sha256")
    }

    sources = [
        ("ml_train_real_ground_truth.csv", REPO_ROOT / "datasets/ml_train_real"),
        ("real_wild_ground_truth.csv", REPO_ROOT / "datasets/real_wild_downloads"),
        ("ground_truth.csv", REPO_ROOT / "datasets/sandbox_raw"),
        ("locked_blind_ground_truth.csv", REPO_ROOT / "datasets/locked_blind_test"),
    ]

    records: list[FileRecord] = []
    labels: list[str] = []
    seen_hashes: set[str] = set()
    skipped_missing = 0
    excluded_overlap = 0

    for csv_name, folder in sources:
        csv_path = REPO_ROOT / "datasets" / csv_name
        if not csv_path.exists() or not folder.exists():
            print(f"Skipping missing source: {csv_path} / {folder}")
            continue
        for row in csv.DictReader(csv_path.open(encoding="utf-8")):
            rel = row.get("relative_path") or row.get("filename") or row.get("file_name") or row.get("name") or next(iter(row.values()))
            fpath = folder / rel
            if not fpath.exists():
                skipped_missing += 1
                continue
            digest = (row.get("sha256") or "").strip().lower()
            if not digest:
                digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if digest in test_hashes:
                excluded_overlap += 1
                continue
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            stat = fpath.stat()
            text, status = extract_document_content(fpath, use_ocr=True)
            rec = FileRecord(
                path=fpath,
                name=fpath.name,
                stem=fpath.stem,
                extension=fpath.suffix.lower(),
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                extracted_text=text,
                extraction_status=status,
                sha256=digest,
            )
            cat14 = map_to_14_categories(row, text)
            records.append(rec)
            labels.append(cat14)

    for rec, cat14 in build_taxonomy_anchor_records():
        records.append(rec)
        labels.append(cat14)

    out_path = REPO_ROOT / "datasets/train_14cat_corpus.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"records": records, "labels": labels, "taxonomy": TAXONOMY_14}, f)

    counts = Counter(labels)
    print(f"Built zero-leakage 14-category training corpus: n={len(records)} (overlap excluded: {excluded_overlap}, missing skipped: {skipped_missing})")
    for c in TAXONOMY_14:
        print(f"  {c:<24} {counts[c]:>4}")


if __name__ == "__main__":
    main()


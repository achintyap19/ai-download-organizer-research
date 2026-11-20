# Strategy 01: Dynamic Two-Tier Rule-Based Architecture & Rigorous Benchmark

## 1. Architectural Positioning & Responses to Design Review Questions

### 1.1 DropIt Content-Aware Capabilities vs. Structured Document Parsing
- **Correction & Nuance:** DropIt (v8.0+) includes an *Additional Filters → File Content* regex option that scans raw plain-text streams and supports backreferencing matched strings into destination paths.
- **Why DropIt Remains Insufficient for Unstructured Downloads Folders:**
  1. **No Office Container Decompression:** Modern `.docx`, `.xlsx`, and `.pptx` files are ZIP-compressed XML archives; raw stream regex cannot read their body text or spreadsheet cells without structured parsing.
  2. **Zero OCR Support:** Scanned PDFs (such as traffic challans `1.pdf`–`7.pdf` with zero selectable text layer) and text-heavy screenshots are completely opaque to regex engines.
  3. **Static Rules vs. Dynamic Workspace Adaptation:** Traditional rule tools require manual rule authoring whenever a user encounters new file domains (`Travel_Bookings`, `Medical_Health`). Strategy 01 pairs universal seed rules with autonomous runtime rule induction.

### 1.2 Evaluation of Existing Rule-Based Tools
| Tool | Platform / License | Content Rules | Dry-Run Preview | Atomic Rollback | Why Not Extended |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hazel 6** | macOS (Proprietary) | Spotlight + OCR | Per-file rule check | Single-file revert | Closed-source; macOS-only (incompatible with Windows 11). |
| **DropIt** | Windows (GPL-3.0) | Raw stream regex | Optional confirmation | None (batch journal) | Written in legacy **AutoIt** (unmaintained since Oct 2018); lacks Python/ML/ONNX bindings. |
| **Directory Opus / XYplorer** | Windows (Commercial) | Scripted metadata | Live rename table | GUI undo stack | Closed commercial plugin ecosystems unsuitable for ML/LLM benchmarking. |
| **organize-rt (`tfeldmann/organize`)** | Cross-platform (MIT, Python) | YAML + `pypdf` | `--sim` dry-run | None | Lacks conditional OCR fallback, runtime rule induction, SQLite SHA-256 rollback journal, and shared classifier harness. |

### 1.3 Two-Tier Zero-Overfit Rule Engine (`seed_rules.yaml` + `runtime_rules.yaml`)
To eliminate dataset memorization while supporting new domains at runtime, `DynamicRuleClassifier` (`strategies/01_rule_based/dynamic_classifier.py`) separates rules into two tiers:
1. **Tier 1 — Universal Seed Rules (`seed_rules.yaml`):** Frozen, domain-general regex patterns (e.g., `invoice`, `syllabus`, `resume`) containing **zero** dataset-specific identifiers, roll numbers, or personal filenames.
2. **Tier 2 — Dynamic Runtime Rules (`runtime_rules.yaml`):** Starts empty on a new machine/dataset. When low-confidence or unseen-domain files accumulate in `needs_review = True`, `induce_runtime_regex()` clusters flagged files and synthesizes discriminative filename + body regex rules (`rt_<category>_<id>`) persisted to workspace memory.
- **Preprocessing vs. Inference:** `PyMuPDF`, `python-docx`, `openpyxl`, `python-pptx`, and `RapidOCR` act strictly as deterministic **input extractors** mapping binary bytes into plain text (`FileRecord.extracted_text`). Strategy 01 uses zero learned weights or vector embeddings.

### 1.4 Lazy Conditional OCR Cascade & Fault Tolerance
Running OCR blindly across hundreds of files would require `~6+ minutes` on CPU. Instead, `shared/scanner.py` enforces a 3-tier cascade:
1. **Binary/Media Skip (`0 ms`):** `.exe`, `.msi`, `.zip`, `.mp4`, `.mkv` skip text extraction immediately.
2. **Native Document Parsing:** Extracts text directly from PDFs, Office XML containers, and source files.
3. **Conditional `RapidOCR` Gate:** Invoked *only* when a `.pdf` contains `< 30` non-whitespace characters across its first 2 pages or an image filename hints at a document/screenshot.
4. **Fault Tolerance:** Every parser runs inside granular exception handlers recording explicit `extraction_status` states (`NATIVE_TEXT`, `OCR_USED`, `PASSWORD_PROTECTED`, `CORRUPT_OR_UNREADABLE`, `SKIPPED_BINARY`), falling back to metadata + `needs_review = True` without aborting batch execution.

### 1.5 Scaling Dry-Run Previews & SHA-256 Rollback Integrity
- **Confidence-Tiered Dry-Run Triage (`shared/planner.py`):** Aggregates moves by destination folder, collapses high-confidence auto-moves (`>= 0.55`), and expands low-confidence, tie-conflict, or unreadable files in a dedicated **Needs-Review Queue**.
- **Post-Move Edit & Rename Safety (`shared/executor.py`):** Stores `(source_path, destination_path, sha256_at_move)` in SQLite (`operation_journal`). Before executing `--rollback <batch_id>`, it verifies destination SHA-256 integrity, flagging modified files (`ROLLED_BACK_MODIFIED_CONTENT`) and cleanly skipping missing/renamed destination files (`SKIPPED_MISSING_AT_DEST`).

### 1.6 Multi-Signal Scoring & Anti-False-Positive Guards
1. **Weighted Multi-Modal Scoring:** Combines `extensions`, `filename_regex`, and `content_regex` with diminishing-returns hit scaling, a **Header Boost (`+0.12`)** for matches in the first 250 characters, and a **Cross-Modal Corroboration Bonus (`+0.10`)** when filename and body agree.
2. **Ambiguous Generic Document Guards:**
   - Standalone generic documents (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`) with zero filename or content matches are capped at `0.25` (forcing `needs_review = True`).
   - Filename-only matches on generic documents with `body_hits == 0` (when readable body text exists) are capped at **`0.48`** (`unconfirmed_fn_only`), preventing false confident routing (e.g., `lab_diagnostic_cbc_1.pdf` falsely auto-filing into `Academic_Study` via `"lab"`).
3. **Ambiguity Margin (`<= 0.10`):** Top-two rules within `0.10` points flag `needs_review = True` and record `conflict_with`.

---

## 2. Audit-Proof Real-World Blind Benchmark (`experiments/evaluate_real_wild_blind.py`)

To eliminate synthetic template repetition, duplicate-inflated accuracy, and post-hoc rule tuning, Strategy 01 was evaluated on a brand-new corpus of **$N = 96$ unique, non-templated real-world files** (`datasets/real_wild_ground_truth.csv`). Every file was deduplicated via SHA-256 and manually annotated from raw document snippets prior to inference (`git commit fe97d3b`), and the frozen rule engine (`seed_rules.yaml` and `dynamic_classifier.py`) was executed **once** (`git commit a28ede8`) without any rule modifications.

### 2.1 Stage 1: Real-World Blind Ablation (`n = 85` Unique Files, 7 Core Categories)
Evaluated across untouched user downloads and diverse public sources (`arXiv`, `IRS.gov`, `IETF RFCs`, `PyPI`, `GitHub Releases`, `TidyTuesday`, `Seaborn`, `Wikimedia Commons`, and public utility/invoice samples) using **only** frozen `seed_rules.yaml`:

| Configuration | Accuracy | Macro F1 | Review Queue | Scan Latency | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A:** Metadata & Filename Regex Only | `75.29%` (`64/85`) | `80.13%` | `58.82%` | `11.4 ms` | Nearly `59%` of real downloads lack descriptive filenames and require manual review. |
| **Config B:** + Native Document Parsing | **`76.47%` (`65/85`)** | **`80.79%`** | `35.29%` | `1,004.9 ms` | Native text extraction (`PyMuPDF`, `python-docx`) cuts the human review queue by **`23.53` percentage points** in `1.00 s`. |
| **Config C:** + Conditional `RapidOCR` | `75.29%` (`64/85`) | `80.13%` | **`32.94%`** | `38,734.6 ms` | Further lowers review rate to `32.94%`, but adds `37.7 s` CPU OCR overhead and 1 noisy-token false positive (`hj.pdf`). |

#### Per-Category Performance on the Real-World Blind Set (`n = 85`, Config C)
- **Perfect Precision & Recall (`100.00%` F1):** `Archives_Compressed` ($n=8$), `Media_Photos_Graphics` ($n=5$), and `Spreadsheets_Data` ($n=11$).
- **High Precision, Lower Recall on Out-of-Distribution Files:**
  - `Academic_Study` ($n=23$): Precision `61.11%`, Recall `95.65%`, F1 **`74.58%`** (acts as the `.pdf` fallback attractor when domain keywords miss).
  - `Finance_Invoices` ($n=17$): Precision `90.00%`, Recall `52.94%`, F1 **`66.67%`**.
  - `Software_Installers` ($n=11$): Precision `100.00%`, Recall `45.45%`, F1 **`62.50%`**.
  - `Career_Identity` ($n=10$): Precision `100.00%`, Recall `40.00%`, F1 **`57.14%`**.

### 2.2 Stage 2: Disjoint Few-Shot Unseen-Domain Adaptation (`Travel_Bookings` & `Medical_Health`)
To evaluate runtime adaptation without support/query leakage, 11 real-world files from two completely held-out domains absent from `seed_rules.yaml` were partitioned into a **labeled support set ($k = 2$ files per domain, 4 total)** and a **strictly disjoint unseen query set ($m = 7$ files: 3 Travel, 4 Medical)**:

| Evaluation Phase | Accuracy ($m = 7$) | Macro F1 | Labeling Cost | Per-Domain Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| **Before Adaptation (Frozen Seed Rules)** | `0.00%` (`0/7`) | `0.00%` | `0` labels | Zero false routing into unseen categories prior to rule induction. |
| **After $k = 2$ Support Rule Induction** | **`57.14%` (`4/7`)** | **`67.86%`** | $k = 2$ files/domain | `Medical_Health`: **`75.00%` (`3/4`)** · `Travel_Bookings`: **`33.33%` (`1/3`)** |

### 2.3 Independent Non-Circular Duplicate Detection Audit (`10` Seeded Copy Pairs)
To avoid circular evaluation against the detector's own labels, 10 distinct source files were paired with 10 externally generated, renamed copies (`Copy_01_final_<name>`) with pre-recorded ground-truth pairs before running `detect_exact_duplicates()` (`Size` $\rightarrow$ `Partial SHA-256` $\rightarrow$ `Full SHA-256`):
- **Precision:** **`100.00%`** | **Recall:** **`100.00%`** | **F1-Score:** **`100.00%`** (`TP = 10, FP = 0, FN = 0`).

---

## 3. Empirical Error Analysis of the 21 Blind Misclassifications (Motivating Strategies 02–05)

Inspecting all **21 errors (`24.71%` error rate)** on the frozen real-world blind set reveals five concrete structural limitations of deterministic rule engines:

1. **Extension & Packaging Blindness (`6` errors):** All six PyPI wheel installers (`*.whl`) fell into `Uncategorized_Misc` (`conf = 0.0`) because `seed_rules.yaml` only enumerated `.exe`/`.msi` under `Software_Installers`.
2. **Regional & Sub-Domain Vocabulary Shift (`8` errors):** While `100%` of commercial invoices and utility bills were classified accurately, 7 US IRS tax forms (`f1040.pdf`, `fw2.pdf`, `f1099nec.pdf`, etc.) and 1 Indian bank balance snapshot (`Your Accounts_...PDF`) lacked Indian GST/invoice keywords and defaulted to `Academic_Study` (`conf = 0.20`).
3. **Rule Priority & Multi-Domain Overlap (`6` errors):** Resumes, CVs, cover letters, admit cards, and university marksheets (`Career_Identity`) contain strong academic terms (`B.Tech`, `University`, `Computer Science`, `Examination`), causing `academic_study` rules to tie or override `career_identity`.
4. **OCR Latency & Noisy-Token Sensitivity (`1` error + `37×` slowdown):** Invoking `RapidOCR` on scanned PDFs increased scan latency from `1.00 s` to `38.73 s` and caused a noisy OCR token in `hj.pdf` to trigger a false `Finance_Invoices` match (dropping Config C accuracy by `1.18%` below Config B).
5. **Lexical Sparsity in $k = 2$ Few-Shot Induction (`3/7` unseen queries missed):** Exact token intersection (`min_doc_freq = 2`) across $k = 2$ support files generalized well on standardized lab reports (`3/4` Medical queries solved, missing only `Z021.pdf` due to custom PUA font encoding), but struggled across heterogeneous travel formats (visa itinerary vs. domestic e-ticket), motivating **TF-IDF statistical weighting (Strategy 02)**, **dense semantic embeddings (Strategy 03)**, and **multimodal LLM escalation (Strategies 04–05)**.

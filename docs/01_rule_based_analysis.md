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

### 2.1 Stage 1: Real-World Blind Ablation & Operational Safety (`n = 85` Unique Files, 7 Core Categories)
Evaluated across untouched user downloads and diverse public sources (`arXiv`, `IRS.gov`, `IETF RFCs`, `PyPI`, `GitHub Releases`, `TidyTuesday`, `Seaborn`, `Wikimedia Commons`, and public utility/invoice samples) using **only** frozen `seed_rules.yaml`:

| Configuration | Overall Accuracy | Auto-Move Accuracy | Manual Review Queue | Unflagged Wrong Moves | Macro F1 | Scan Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A:** Metadata & Filename Regex Only | `75.29%` (`64/85`) | **`100.00%`** (`35/35`) | `58.82%` (`50/85`) | **`0` (`0.00%`)** | `80.13%` | `11.4 ms` |
| **Config B:** + Native Document Parsing | **`76.47%` (`65/85`)** | `92.73%` (`51/55`) | `35.29%` (`30/85`) | `4` (`4.71%`) | **`80.79%`** | `1,004.9 ms` |
| **Config C:** + Conditional `RapidOCR` | `75.29%` (`64/85`) | `91.23%` (`52/57`) | **`32.94%`** (`28/85`) | `5` (`5.88%`) | `80.13%` | `38,734.6 ms` |

#### Key Operational Safety Takeaways
- **Metadata-Only Zero-Error Guarantee (`Config A`):** Achieves **`100.00%` Auto-Move Accuracy** (`35/35`) and **`0` Unflagged Wrong Moves**, at the cost of holding `58.82%` of files in the human review queue.
- **Native Document Parsing Trade-Off (`Config B`):** Cuts the human review burden by **`23.53` percentage points** (`58.82%` $\rightarrow$ `35.29%`) while maintaining **`92.73%` Auto-Move Accuracy** (`51/55`) and only `4` unflagged errors (`4.71%`).
- **Conditional OCR Trade-Off (`Config C`):** Further reduces review burden to **`32.94%`** (`52/57` auto-moved at `91.23%` Auto-Move Accuracy), adding 1 noisy-token unflagged error (`hj.pdf`, `5` total / `5.88%`).

#### Per-Category Performance on the Real-World Blind Set (`n = 85`, Config C)
- **Perfect Precision & Recall (`100.00%` F1):** `Archives_Compressed` ($n=8$), `Media_Photos_Graphics` ($n=5$), and `Spreadsheets_Data` ($n=11$).
- **High Precision, Lower Recall on Out-of-Distribution Files:**
  - `Academic_Study` ($n=23$): Precision `61.11%`, Recall `95.65%`, F1 **`74.58%`** (acts as the `.pdf` fallback attractor when domain keywords miss).
  - `Finance_Invoices` ($n=17$): Precision `90.00%`, Recall `52.94%`, F1 **`66.67%`**.
  - `Software_Installers` ($n=11$): Precision `100.00%`, Recall `45.45%`, F1 **`62.50%`**.
  - `Career_Identity` ($n=10$): Precision `100.00%`, Recall `40.00%`, F1 **`57.14%`**.

### 2.2 Stage 2: Disjoint Few-Shot Unseen-Domain Adaptation (`Travel_Bookings` & `Medical_Health`)
To evaluate runtime adaptation without support/query leakage, 11 real-world files from two completely held-out domains absent from `seed_rules.yaml` were partitioned into a **labeled support set ($k = 2$ files per domain, 4 total)** and a **strictly disjoint unseen query set ($m = 7$ files: 3 Travel, 4 Medical)**:

| Evaluation Phase | Overall Acc ($m = 7$) | Auto-Move Acc | Review Queue | Unflagged Wrong | Macro F1 | Labeling Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Before Adaptation (Frozen Seed Rules)** | `0.00%` (`0/7`) | `0.00%` (`0/1`) | `85.71%` (`6/7`) | `1` (`14.29%`) | `0.00%` | `0` labels |
| **After $k = 2$ Support Rule Induction** | **`42.86%` (`3/7`)** | **`75.00%` (`3/4`)** | **`42.86%` (`3/7`)** | `1` (`14.29%`) | **`53.57%`** | $k = 2$ files/domain |

### 2.3 Stage 2 Non-Regression Check (`n = 85` Blind Set With Learned Medical/Travel Rules Active)
To verify whether dynamically induced runtime rules degrade existing categories, the frozen `n = 85` blind test (`Config C`) was re-evaluated with the learned `Medical_Health` and `Travel_Bookings` rules persisted in `runtime_rules.yaml`:

| Metric (`n = 85`, Config C) | Pre-Adaptation (Seed Only) | Post-Adaptation (+ Learned Rules) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | `75.29%` (`64/85`) | `74.12%` (`63/85`) | **`-1.17%`** (`-1` net) |
| **Auto-Move Accuracy** | `91.23%` (`52/57`) | `83.87%` (`52/62`) | **`-7.36%`** |
| **Manual Review Queue** | `32.94%` (`28/85`) | `27.06%` (`23/85`) | `-5.88%` |
| **Unflagged Wrong Moves** | `5` (`5.88%`) | `10` (`11.76%`) | **`+5` regressions** |

- **Root Cause of the 5 Post-Adaptation Regressions:** `rt_medical_health_001` (`priority=86`) extracted generic tokens shared between the $k=2$ medical support PDFs that also matched 4 US IRS tax forms (`f1040.pdf`, `f941.pdf`, `fss4.pdf`, `fw9.pdf`; `Finance_Invoices` $\rightarrow$ `Medical_Health`) and `E-book of OOP using C++ (E Balagurusamy).pdf` (`Academic_Study` $\rightarrow$ `Medical_Health`) with `needs_review = False`.
- **Architectural Implication:** Unregularized regex induction from tiny ($k=2$) support sets lacks inverse document frequency (IDF) weighting and semantic regularization, causing high-priority lexical false positives on out-of-distribution documents.

### 2.4 Independent Non-Circular Duplicate Detection Audit (`10` Seeded Copy Pairs)
To avoid circular evaluation against the detector's own labels, 10 distinct source files were paired with 10 externally generated, renamed copies (`Copy_01_final_<name>`) with pre-recorded ground-truth pairs before running `detect_exact_duplicates()` (`Size` $\rightarrow$ `Partial SHA-256` $\rightarrow$ `Full SHA-256`):
- **Precision:** **`100.00%`** | **Recall:** **`100.00%`** | **F1-Score:** **`100.00%`** (`TP = 10, FP = 0, FN = 0`).

---

## 3. Empirical Error Analysis of Blind Misclassifications (Motivating Strategies 02–05)

Inspecting the blind misclassifications and non-regression audit reveals six concrete structural limitations of deterministic rule engines:

1. **Extension & Packaging Blindness (`6` errors):** All six PyPI wheel installers (`*.whl`) fell into `Uncategorized_Misc` (`conf = 0.0`) because `seed_rules.yaml` only enumerated `.exe`/`.msi` under `Software_Installers`.
2. **Regional & Sub-Domain Vocabulary Shift (`8` errors):** While `100%` of commercial invoices and utility bills were classified accurately, 7 US IRS tax forms (`f1040.pdf`, `fw2.pdf`, `f1099nec.pdf`, etc.) and 1 Indian bank balance snapshot (`Your Accounts_...PDF`) lacked Indian GST/invoice keywords and defaulted to `Academic_Study` (`conf = 0.20`).
3. **Rule Priority & Multi-Domain Overlap (`6` errors):** Resumes, CVs, cover letters, admit cards, and university marksheets (`Career_Identity`) contain strong academic terms (`B.Tech`, `University`, `Computer Science`, `Examination`), causing `academic_study` rules to tie or override `career_identity`.
4. **OCR Latency & Noisy-Token Sensitivity (`1` error + `37×` slowdown):** Invoking `RapidOCR` on scanned PDFs increased scan latency from `1.00 s` to `38.73 s` and caused a noisy OCR token in `hj.pdf` to trigger a false `Finance_Invoices` match (dropping Config C accuracy by `1.18%` below Config B).
5. **Post-Adaptation Lexical False-Positive Drift (`5` regressions):** High-priority runtime regex rules induced from $k=2$ files lack global IDF calibration, causing generic tokens in `rt_medical_health_001` to hijack previously flagged IRS tax forms and textbooks into unflagged wrong moves.
6. **Lexical Sparsity in $k = 2$ Few-Shot Induction:** Exact token intersection (`min_doc_freq = 2`) across $k = 2$ support files struggles across heterogeneous formats (visa itinerary vs. domestic e-ticket), motivating **TF-IDF statistical weighting (Strategy 02)**, **dense semantic embeddings (Strategy 03)**, and **multimodal LLM escalation (Strategies 04–05)**.


---

## 12-Persona External Blind Benchmark (`datasets/test_set_v1`, `n = 590` Unique Files, Commit `96366b3`)

Evaluating `DynamicRuleClassifier` (`seed_rules.yaml` + `runtime_rules.yaml`) on `datasets/test_set_v1` (`score.py --legacy`):

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Overall Accuracy (Strict / Lenient)** | **`60.0%`** *(95% CI: `56.0?63.9%`)* | Documents: `66.5%` (`n=394`), Images: `44.9%` (`n=158`), Format-only: `55.3%` (`n=38`) |
| **Auto-Move Precision** | **`50.8%`** (`321` auto-moved) | `158` unflagged wrong auto-moves (`26.8%`), `45.6%` routed to review |
| **Macro-F1 (`--legacy`, 8 buckets)** | **`50.1%`** | `NO_LEGACY_EQUIVALENT` (`n=219`, `37.1%`) has `0.0%` recall under closed 7-cat rules |
| **Real Downloads (`n = 76`)** | **`65.8%`** | `73.7%` auto-move precision (`10` unflagged wrong) |
| **Real Photos (`n = 41`)** | **`75.6%`** | `100.0%` auto-move precision (`0` unflagged wrong) |

### Key Bottlenecks Observed
1. **Closed 7-Category Ceiling (`219` `NO_LEGACY_EQUIVALENT` files):** `37.1%` of `test_set_v1` falls outside the legacy 7-category taxonomy (`Work_Professional`, `Home_Household`, `Personal_IDs_Records`, `Travel`, `Health_Medical`, `Other`), causing closed-world misclassifications into `Finance_Invoices` (`16`) and `Academic_Study` (`9`).
2. **Legacy Category Naming Mismatch:** Rules predicting `Media_Screenshots` (`25` on photos, `18` on finance receipts) and `Media_Audio_Video` (`10`) were scored as misses under `--legacy`.
3. **Uninformative & Mixed-Language Filenames:** Accuracy dropped to `50.4%` on uninformative names (`n=228`) and `34.5%` on mixed-language files (`n=29`), confirming why **Strategy 02C** (`81.2%` lenient) and **Strategy 03 (Multilingual Dense Embeddings)** are necessary.

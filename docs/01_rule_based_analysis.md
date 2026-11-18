# Strategy 01: Deterministic Rule-Based Architecture & Empirical Benchmark

## 1. Architectural Positioning & Responses to Design Review Questions

### 1.1 DropIt Content-Aware Capabilities vs. Structured Document Parsing
* **Correction & Nuance:** DropIt (v8.0+) includes an *Additional Filters → File Content* regex option that scans raw plain-text streams and supports backreferencing matched strings into destination paths.
* **Why DropIt Remains Insufficient for Unstructured Downloads Folders:**
  1. **No Office Container Decompression:** Modern `.docx`, `.xlsx`, and `.pptx` files are ZIP-compressed XML archives; raw stream regex cannot read their body text or spreadsheet cells without structured parsing.
  2. **Zero OCR Support:** Scanned PDFs (such as traffic challans `1.pdf`–`7.pdf` with zero selectable text layer) and text-heavy screenshots are completely opaque to regex engines.
  3. **Syntactic vs. Semantic Brittleness:** Hardcoded regex patterns fail whenever filenames are opaque (`59021271.pdf`) or phrasing varies (`"C++ Programming Course"` vs. `"syllabus"`).

### 1.2 Evaluation of Existing Rule-Based Tools
| Tool | Platform / License | Content Rules | Dry-Run Preview | Atomic Rollback | Why Not Extended |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hazel 6** | macOS (Proprietary, ) | Spotlight + OCR | Per-file rule check | Single-file revert | Closed-source; macOS-only (incompatible with Windows 11). |
| **DropIt** | Windows (GPL-3.0) | Raw stream regex | Optional confirmation | None (batch journal) | Written in legacy **AutoIt** (unmaintained since Oct 2018); lacks Python/ML/ONNX bindings. |
| **Directory Opus / XYplorer** | Windows (Commercial) | Scripted metadata | Live rename table | GUI undo stack | Closed commercial plugin ecosystems unsuitable for ML/LLM benchmarking. |
| **organize-rt (`tfeldmann/organize`)** | Cross-platform (MIT, Python) | YAML + `pypdf` | `--sim` dry-run | None | Lacks conditional OCR fallback, SQLite SHA-256 rollback journal, and shared classifier harness. |

### 1.3 Why Strategy 01 Remains Strictly Rule-Based (vs. Strategy 05 Hybrid)
* **Preprocessing vs. Inference:** `PyMuPDF`, `python-docx`, `openpyxl`, `python-pptx`, and `RapidOCR` act strictly as deterministic **input extractors** mapping binary bytes into plain text (`FileRecord.extracted_text`).
* **Decision Engine:** Strategy 01 uses zero learned weights, zero vector embeddings, and zero probabilistic inference. Identical inputs always yield identical rule scores. In contrast, **Strategy 05 (Hybrid)** introduces learned semantic embeddings and local LLM reasoning into the classification decision loop.

### 1.4 Lazy Conditional OCR Cascade & Fault Tolerance
Running OCR blindly across all 64 files would require `~6+ minutes` on CPU. Instead, `shared/scanner.py` enforces a 3-tier cascade:
1. **Binary/Media Skip (`0 ms`):** `.exe`, `.msi`, `.zip`, `.mp4`, `.mkv` skip text extraction immediately.
2. **Native Document Parsing (`1,703.2 ms` total across 64 files):** Extracts text directly from PDFs, Office XML containers, and source files.
3. **Conditional `RapidOCR` Gate (`13 / 64` files triggered):** Invoked *only* when a `.pdf` contains `< 30` non-whitespace characters across its first 2 pages or an image filename hints at a document/screenshot.
4. **Fault Tolerance:** Every parser runs inside granular exception handlers recording explicit `extraction_status` states (`NATIVE_TEXT`, `OCR_USED`, `PASSWORD_PROTECTED`, `CORRUPT_OR_UNREADABLE`, `SKIPPED_BINARY`), falling back to metadata + `needs_review = True` without aborting batch execution.

### 1.5 Scaling Dry-Run Previews for 100+ Files
Rather than dumping a flat list of 100+ moves (causing alert fatigue), `shared/planner.py` groups previews into:
1. **Category Roll-Up Summary:** Aggregated counts per destination folder and duplicate savings.
2. **Confidence-Tiered Triage:** High-confidence moves (`>= 0.70`, `73.4%` of files in Config C) are collapsed for one-click approval, while the `26.6%` (`17/64`) low-confidence, tie-conflict, or unreadable files are surfaced in an expanded **Needs-Review Queue**.

### 1.6 Post-Move Edit & Rename Safety During Rollback
`shared/executor.py` stores `(source_path, destination_path, sha256_at_move)` in SQLite (`operation_journal`). Before executing `--rollback <batch_id>`, it verifies destination SHA-256 integrity:
* **Unmodified (`sha256` matches):** Atomically moves back to `source_path`.
* **Modified content (`sha256` changed):** Flags `ROLLED_BACK_MODIFIED_CONTENT` / requires `--force` to prevent accidental loss of edits.
* **Renamed or missing at destination:** Logs `SKIPPED_MISSING_AT_DEST` cleanly without crashing the remaining batch rollback.

### 1.7 Deterministic Tie-Breaking & Conflict Flagging
1. **Signal Specificity Weighting:** Extension (`+0.45`) + Filename Regex (`+0.35`) + Content Keywords (`+0.25` single hit, `+0.35` multi-hit), with a **Generic Document Guard** capping standalone `.pdf`/`.docx`/`.xlsx` scores at `0.30` unless reinforced by a filename or content signal.
2. **Rule Priority (`priority: 1..100`):** Deterministic tie-breaker when raw scores match.
3. **Ambiguity Margin (`<= 0.10`):** When two rules score within `0.10` points, the engine sets `needs_review = True` and populates `conflict_with` for human review or Strategy 05 LLM escalation.

---

## 2. Empirical Benchmark Results (`n = 64` Ground-Truth Files)

### 2.1 Ablation Comparison Across Rule Configurations
| Metric | Config A: DropIt-Style (Metadata + Regex) | Config B: Strategy 01 (Native Parsing) | Config C: Strategy 01 (+ Conditional OCR) | Delta (A → C) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | `67.19%` (`43/64`) | `73.44%` (`47/64`) | **`84.38%`** (`54/64`) | **`+17.19%`** |
| **Macro Precision** | `80.48%` | `74.37%` | **`74.37%`** | `-6.11%` |
| **Macro Recall** | `68.38%` | `70.29%` | **`77.29%`** | **`+8.91%`** |
| **Macro F1-Score** | `66.78%` | `66.72%` | **`73.79%`** | **`+7.01%`** |
| **Weighted F1-Score** | `73.00%` | `76.00%` | **`87.00%`** | **`+14.00%`** |
| **Exact Duplicate P / R** | `100.00% / 100.00%` | `100.00% / 100.00%` | **`100.00% / 100.00%`** | `0.00%` |
| **Scan + Extraction Time** | `~0 ms` | `1,703.2 ms` (`26.6 ms/file`) | `80,325.9 ms` (`13` OCR files) | `+78.6 s` (OCR) |
| **Rule Inference Latency** | `2.76 ms` (`0.043 ms/file`) | `3.53 ms` (`0.055 ms/file`) | **`4.39 ms`** (`0.069 ms/file`) | `+1.63 ms` |
| **Manual Review Queue** | `46.9%` (`30/64`) | `37.5%` (`24/64`) | **`26.6%`** (`17/64`) | **`-20.3%`** |

### 2.2 Per-Class Performance (Config C: Full Rule-Based Engine)
| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `Academic_Study` | 21 | `0.89` | `0.76` | **`0.82`** |
| `Archives_Compressed` | 5 | `1.00` | `1.00` | **`1.00`** |
| `Career_Identity` | 3 | `0.50` | `0.33` | **`0.40`** |
| `Finance_Invoices` | 10 | `1.00` | `0.80` | **`0.89`** |
| `Media_Audio_Video` | 2 | `1.00` | `1.00` | **`1.00`** |
| `Media_Photos_Graphics` | 6 | `0.71` | `0.83` | **`0.77`** |
| `Media_Screenshots` | 1 | `0.33` | `1.00` | **`0.50`** |
| `Software_Installers` | 13 | `1.00` | `1.00` | **`1.00`** |
| `Spreadsheets_Data` | 3 | `1.00` | `1.00` | **`1.00`** |

---

## 3. Error Taxonomy of Remaining 10 Misclassifications (Motivating Strategies 02–05)

1. **Semantic Vocabulary Gaps (`3 files` → `Uncategorized_Misc`):**
   * `Achintya Pathak_C++ Programming Course For Beginners.pdf`, `AI_Downloads_Folder_Organization_White_Paper_Achintya_Pathak.md`, `cpp.txt`
   * *Root Cause:* Opaque or domain-specific phrasing (`"C++"`, `"white paper"`) absent from hardcoded `content_keywords`. Addressed in **Strategy 02 (TF-IDF ML)** and **Strategy 03 (Dense Embeddings)**.
2. **Single-Keyword Collisions (`3 files`):**
   * `AI_Downloads_Folder_Organization_White_Paper.docx` (`Academic_Study` → `Career_Identity` due to a single mention of `"internship"`), plus `31.pdf` and `59021271.pdf` (`Finance_Invoices` → `Academic_Study`).
   * *Root Cause:* Substring matching lacks inverse document frequency (IDF) weighting across the full document.
3. **Visual / Unlabeled Image Ambiguities (`4 files`):**
   * `decoder_diagram.png` (`Academic_Study`), `solo-leveling-game-3840x2160-17908.jpeg` (`Media_Photos_Graphics`), and `WhatsApp Image 2025-06-08...jpeg` (`Career_Identity` signature/ID scans).
   * *Root Cause:* Non-screenshot images bypass OCR in Strategy 01 to save latency, requiring multimodal/vision or selective OCR escalation in **Strategies 04 & 05**.

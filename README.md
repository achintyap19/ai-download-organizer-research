# AI-Driven Downloads Folder Organizer — Comparative Strategy Benchmark

An empirical systems and machine learning research project evaluating **five architectural paradigms** for automated, safe, and content-aware organization of unstructured local `Downloads` directories.

## Research Objective
Compare deterministic, statistical, semantic, and generative classification pipelines on a **locked, non-templated real-world ground-truth corpus** (`N = 96` unique files deduplicated by SHA-256: `n = 85` blind test files across 7 core domains + `n = 11` disjoint files for $k=2$ few-shot adaptation) across **Overall Accuracy**, **Auto-Move Accuracy**, **Manual Review Queue %**, **Unflagged Wrong Moves**, **Macro F1**, and **Latency**.

| Strategy | Paradigm | Status | Overall Acc (`n=85`) | Auto-Move Acc | Review Queue | Unflagged Wrong | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01. Rule-Based Engine** | Two-Tier YAML Rules + Native Parsers + Conditional `RapidOCR` | **Completed** | **`76.47%`** *(Cfg B)* / **`75.29%`** *(Cfg C)* | **`92.73%`** *(Cfg B)* / **`91.23%`** *(Cfg C)* | **`35.29%`** *(Cfg B)* / **`32.94%`** *(Cfg C)* | **`4` (`4.71%`)** *(Cfg B)* / **`5` (`5.88%`)** *(Cfg C)* | **`80.79%`** *(Cfg B)* / **`79.64%`** *(Cfg C)* |
| **02A. Traditional ML (Baseline)** | Multi-Modal TF-IDF + LSA + Structural Stats *(Synthetic Train)* | **Completed** | `60.00%` *(Cfg C)* | `78.95%` | `32.94%` | `12` (`14.12%`) | `60.60%` |
| **02B. Traditional ML (Real-World)** | Multi-Modal TF-IDF + LSA + Structural Stats *(SHA-256-Disjoint Real Train, Rigor-Audited)* | **Completed** | **`90.59%`** *(Cfg C, CV-Locked)* / **`87.06%`** *(Raw Stems)* | **`93.75%`** *(CV-Locked)* / **`95.31%`** *(0.56/0.14 Gate)* | **`5.88%`** *(CV-Locked)* / **`24.71%`** *(0.56/0.14 Gate)* | **`5` (`5.88%`)** / **`3` (`3.53%`)** | **`89.35%`** *(CV-Locked)* / **`86.90%`** *(Raw)* |
| **02C. Purpose-First ML (14-Cat)** | 7-Branch Multi-Modal TF-IDF + Purpose-Gated Ext + Soft-Voting (`test_set_v1`, `n=590`) | **Completed** (`586dcc7`) | **`66.3%`** *(Strict)* / **`81.2%`** *(Lenient)* | **`86.7%`** (`496` moved) | **`15.9%`** | **`66` (`11.2%`)** | **`67.5%`** *(14-cat)* | **Completed** (586dcc7) | **66.3%** *(Strict)* / **81.2%** *(Lenient)* | **86.7%** (496 moved) | **15.9%** | **66 (11.2%)** | **67.5%** *(14-cat)* |
| **03. Dense Embeddings** | Local Sentence-Transformers + Cosine / Prototype Classifier | **Next** | — | — | — | — | — |
| **04. Local LLM** | Quantized Local SLM via Ollama (Constrained JSON Schema) | Planned | — | — | — | — | — |
| **05. Hybrid Cascade** | Rules -> Embeddings -> Local LLM Escalation | Planned | — | — | — | — | — |

---

## Strategy 01 & 02 Highlights (`real_wild_ground_truth.csv`, `N = 96` Unique Real-World Files)

### 1. Stage 1 Real-World Blind Ablation & Operational Safety (`n = 85` Unique Files, 7 Categories)
- **Strategy 01 (Rule-Based Engine, `seed_rules.yaml`):**
  - **Config A (Metadata & Filename Regex Only):** `75.29%` Overall (`64/85`) | `100.00%` Auto-Move (`35/35`) | `58.82%` Review | `0` Unflagged Wrong | `79.64%` Macro F1
  - **Config B (+ Native Parsing):** **`76.47%`** Overall (`65/85`) | **`92.73%`** Auto-Move (`51/55`) | `35.29%` Review | `4` Unflagged Wrong | **`80.79%`** Macro F1
  - **Config C (+ Conditional `RapidOCR`):** `75.29%` Overall (`64/85`) | `91.23%` Auto-Move (`52/57`) | **`32.94%`** Review | `5` Unflagged Wrong | `79.64%` Macro F1
- **Strategy 02B (Multi-Modal TF-IDF + LSA + Structural Stats, `CalibratedLinearSVC` trained on `n = 149` SHA-256-disjoint real files):**
  - **Config A (Metadata-Only):** `62.35%` Overall (`53/85`) | `88.89%` Auto-Move (`40/45`) | `47.06%` Review | `5` Unflagged Wrong | `72.21%` Macro F1
  - **Config B (+ Native Parsing):** `83.53%` Overall (`71/85`) | **`96.72%`** Auto-Move (`59/61`) | `28.24%` Review | **`2` (`2.35%`)** Unflagged Wrong | `84.93%` Macro F1
  - **Config C (+ Conditional `RapidOCR`, Raw Stems):** `87.06%` Overall (`74/85`) | **`95.31%`** Auto-Move (`61/64`) | `24.71%` Review | **`3` (`3.53%`)** Unflagged Wrong | `86.90%` Macro F1
  - **Config C (+ Artifact-Scrubbed Stems & Zero-Peeking 5-Fold CV Cutoffs):** **`90.59%`** Overall (`77/85`) | **`93.75%`** Auto-Move (`75/80`) | **`5.88%`** Review | `5` Unflagged Wrong | **`89.35%`** Macro F1
- **Methodological Rigor & Generalization Audit (`experiments/audit_strategy02_rigor.py`):**
  - **Filename vs. Body Ablation:** Scrubbing repository artifact tokens (`sample`, `altacv`, `template`) eliminated spurious filename shortcuts (fixing both `"sample"` utility bills misclassified as CVs) and raised blind accuracy from `87.06%` -> **`90.59%`**. Removing filename features entirely still yielded **`88.24%`** (`95.24%` Auto-Move), confirming strong reliance on extracted document text.
  - **Leave-One-Source-Out (LOSO) Generalization:** Holding out entire source domains (`downloads_untouched` -> `65.38%`, `github.com` -> `32.43%`, `irs.gov` / `arxiv.org` / `pypi.org` -> `0.00%`) demonstrated that bag-of-words TF-IDF struggles when unseen domains introduce novel vocabulary—directly motivating **Strategy 03 (Pretrained Dense Sentence Embeddings)**.

### 2. Stage 2 Disjoint Few-Shot Adaptation (`k = 2` Support -> `m = 7` Unseen Queries) & Non-Regression Audit
- **Strategy 01 (Runtime Regex Induction):** Improved unseen domain accuracy (`Travel_Bookings`, `Medical_Health`) from `0.00%` -> **`42.86%`**, but caused **5 regressions** on the 85-file blind test (`75.29%` -> `74.12%`, `-1.17%` delta; unflagged wrong moves `5` -> `10`) due to over-generalized token matching on IRS tax forms.
- **Strategy 02B (Calibrated Linear Retraining):** Improved unseen domain accuracy from `0.00%` -> **`28.57%`** with **`100.00%` routed safely to manual review** (`0` silent misroutes), and preserved baseline accuracy with **zero regressions** on the 85-file blind test (`0` unflagged wrong moves).

---


---

## 12-Persona External Blind Benchmark (`datasets/test_set_v1`, `n = 590` Unique Files, 14-Category Taxonomy)
Evaluated strictly as a black-box benchmark (zero test-set inspection; all training sets verified SHA-256-disjoint):

| Model | Scoring Mode | Strict Acc | Lenient Acc (95% CI) | Auto-Move Acc | Review Queue | Unflagged Wrong | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy 01** (`DynamicRuleClassifier`, `96366b3`) | `--legacy` (7-cat) | `60.0%` | `60.0%` (`56.0?63.9%`) | `50.8%` (`321`) | `45.6%` | `158` (`26.8%`) | `50.1%` *(8 buckets)* |
| **Strategy 02B** (`best_ml_classifier.joblib`) | `--legacy` (7-cat) | `58.1%` | `58.1%` (`54.1?62.1%`) | `44.0%` (`291`) | `50.7%` | `163` (`27.6%`) | `46.3%` *(8 buckets)* |
| **Strategy 02C** (`ml_classifier_14cat_v2.joblib`, `586dcc7`) | **Native 14-cat** | **`66.3%`** | **`81.2%`** (`77.8?84.1%`) | **`86.7%`** (`496`) | **`15.9%`** | **`66` (`11.2%`)** | **`67.5%`** *(14 cats)* |

See [`docs/02c_purpose_first_ml_analysis.md`](docs/02c_purpose_first_ml_analysis.md) for the full group, source, flag, and error breakdown motivating **Strategy 03 (Multilingual Dense Embeddings)**.

--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy 02B** (est_ml_classifier.joblib) | --legacy (7-cat) | 58.1% | 58.1% (54.1–62.1%) | 44.0% (291) | 50.7% | 163 (27.6%) | 46.3% *(8 buckets)* |
| **Strategy 02C** (ml_classifier_14cat_v2.joblib, 586dcc7) | **Native 14-cat** | **66.3%** | **81.2%** (77.8–84.1%) | **86.7%** (496) | **15.9%** | **66 (11.2%)** | **67.5%** *(14 cats)* |

See [docs/02c_purpose_first_ml_analysis.md](docs/02c_purpose_first_ml_analysis.md) for the full group, source, flag, and error breakdown motivating **Strategy 03 (Multilingual Dense Embeddings)**.
---

## Key Architectural Highlights
- **3-Tier Lazy Extraction Cascade (`shared/scanner.py`):** Skips pure binaries/media (`0 ms`), parses native digital documents via `PyMuPDF`, `python-docx`, `openpyxl`, and `python-pptx`, and triggers `RapidOCR` *only* on scanned PDFs (`< 30` selectable chars) or document-hinted images.
- **3-Stage Exact Deduplication:** Groups by byte size -> partial edge SHA-256 -> full SHA-256 (`100.00%` Precision/Recall on non-circular seeded duplicate audit) while preserving canonical filenames.
- **Transactional Rollback with Post-Move Edit Protection (`shared/executor.py`):** Logs `sha256_at_move` in SQLite (`operation_journal`) and verifies destination file integrity before reverting batches so post-move user edits are never overwritten.
- **Privacy-Preserving Evaluation:** Personal raw files (`datasets/sandbox_raw/`, `datasets/real_wild_downloads/`, `datasets/ml_train_real/`) and local SQLite journals are excluded via `.gitignore`; sanitized ground-truth labels and reproducible benchmark metrics are versioned in `datasets/` and `experiments/results/`.

---

## Repository Structure
```text
├── docs/
│   ├── 01_rule_based_analysis.md          # Strategy 01 architectural review, operational safety & audit report
│   └── 02_ml_analysis.md                  # Strategy 02A vs 02B multi-modal ML & rigor audit analysis
├── shared/
│   ├── models.py                          # FileRecord, ClassificationResult & ActionItem contracts
│   ├── scanner.py                         # 3-tier extraction cascade, conditional RapidOCR & SHA-256 deduplicator
│   ├── base_classifier.py                 # Shared classifier interface across Strategies 01-05
│   ├── planner.py                         # Confidence-tiered dry-run preview & conflict triage
│   └── executor.py                        # Atomic filesystem executor & SHA-256-verified SQLite rollback
├── strategies/
│   ├── 01_rule_based/
│   │   ├── seed_rules.yaml                # Frozen priority-weighted domain rules & generic-extension guards
│   │   ├── runtime_rules.yaml             # Dynamically induced workspace rules
│   │   └── dynamic_classifier.py          # Two-tier scorer, tie-conflict detector & regex rule inducer
│   └── 02_ml_classification/
│       ├── ml_classifier.py               # 6-branch artifact-scrubbed TF-IDF + LSA + structural ML classifier
│       └── artifacts/
│           └── best_ml_classifier.joblib  # Serialized CalibratedLinearSVC pipeline
├── experiments/
│   ├── evaluate_real_wild_blind.py        # Locked 96-file Strategy 01 blind, safety & non-regression harness
│   ├── evaluate_strategy02_ml.py          # Strategy 02A vs 02B real-world benchmark harness
│   ├── audit_strategy02_rigor.py          # Filename ablation, Leave-One-Source-Out & CV cutoff audit
│   └── results/
│       ├── strategy01_real_wild_blind_results.json
│       ├── strategy02_ml_benchmark_results.json
│       └── strategy02_rigor_audit.json
└── tests/
    ├── test_fault_tolerance.py            # Corrupt PDF, encrypted PDF & out-of-taxonomy fallback tests
    └── test_executor_rollback.py          # Post-move edit & rename rollback safety tests


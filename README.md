# AI-Driven Downloads Folder Organizer — Comparative Strategy Benchmark

An empirical systems and machine learning research project evaluating **five architectural paradigms** for automated, safe, and content-aware organization of unstructured local `Downloads` directories.

## Research Objective
Compare deterministic, statistical, semantic, and generative classification pipelines on a **locked, non-templated real-world ground-truth corpus** (`N = 96` unique files deduplicated by SHA-256: `n = 85` blind test files across 7 core domains + `n = 11` disjoint files for $k=2$ few-shot adaptation) across **Overall Accuracy**, **Auto-Move Accuracy**, **Manual Review Queue %**, **Unflagged Wrong Moves**, **Macro F1**, and **Latency**.

| Strategy | Paradigm | Status | Overall Acc (`n=85`) | Auto-Move Acc | Review Queue | Unflagged Wrong | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01. Rule-Based Engine** | Two-Tier YAML Rules + Native Parsers + Conditional `RapidOCR` | **Completed** | **`76.47%`** *(Cfg B)* / **`75.29%`** *(Cfg C)* | **`92.73%`** *(Cfg B)* / **`91.23%`** *(Cfg C)* | **`35.29%`** *(Cfg B)* / **`32.94%`** *(Cfg C)* | **`4` (`4.71%`)** *(Cfg B)* / **`5` (`5.88%`)** *(Cfg C)* | **`80.79%`** *(Cfg B)* / **`79.64%`** *(Cfg C)* |
| **02. Traditional ML** | Multi-Modal TF-IDF + Structural Stats + Logistic Regression | **In Progress** | — | — | — | — | — |
| **03. Dense Embeddings** | Local Sentence-Transformers + Cosine / Prototype Classifier | Planned | — | — | — | — | — |
| **04. Local LLM** | Quantized Local SLM via Ollama (Constrained JSON Schema) | Planned | — | — | — | — | — |
| **05. Hybrid Cascade** | Rules -> Embeddings -> Local LLM Escalation | Planned | — | — | — | — | — |

---

## Strategy 01 Highlights (`real_wild_ground_truth.csv`, `N = 96` Unique Real-World Files)

### 1. Stage 1 Real-World Blind Ablation & Operational Safety (`n = 85` Unique Files, 7 Categories)
Evaluated strictly with frozen universal `seed_rules.yaml` (`runtime_rules.yaml` empty):

| Configuration | Overall Accuracy | Auto-Move Accuracy | Manual Review Queue | Unflagged Wrong Moves | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A:** Metadata & Filename Regex Only | `75.29%` (`64/85`) | `100.00%` (`35/35`) | `58.82%` (`50/85`) | **`0` (`0.00%`)** | `79.64%` |
| **Config B:** + Native Document Parsing (`PyMuPDF`, Office XML) | **`76.47%`** (`65/85`) | **`92.73%`** (`51/55`) | `35.29%` (`30/85`) | `4` (`4.71%`) | **`80.79%`** |
| **Config C:** + Conditional `RapidOCR` | `75.29%` (`64/85`) | `91.23%` (`52/57`) | **`32.94%`** (`28/85`) | `5` (`5.88%`) | `79.64%` |

### 2. Stage 2 Disjoint Few-Shot Adaptation (`k = 2` Support -> `m = 7` Unseen Queries) & Non-Regression Audit
- **Disjoint Unseen-Domain Adaptation (`Travel_Bookings` & `Medical_Health`):** Inducing runtime rules from $k=2$ labeled support PDFs per new category improved unseen query accuracy from **`0.00%`** (`85.71%` review queue) to **`42.86%`** (`75.00%` Auto-Move Accuracy, `42.86%` review queue, `1` unflagged wrong move).
- **Post-Adaptation 85-File Non-Regression Check:** Re-running the 85-file blind test with the learned `Medical_Health` and `Travel_Bookings` runtime rules active resulted in **5 regressions** (`75.29%` -> `74.12%`, `-1.17%` delta) and increased unflagged wrong moves from `5` to `10` (`Auto-Move Accuracy`: `91.23%` -> `83.87%`), because `rt_medical_health_001` matched generic document tokens inside IRS tax forms (`f1040.pdf`, `f941.pdf`, `fss4.pdf`, `fw9.pdf`) and `E-book of OOP using C++ (E Balagurusamy).pdf`. This empirical finding directly motivates statistical/semantic representations in Strategies 02–05.

---

## Key Architectural Highlights
- **3-Tier Lazy Extraction Cascade (`shared/scanner.py`):** Skips pure binaries/media (`0 ms`), parses native digital documents via `PyMuPDF`, `python-docx`, `openpyxl`, and `python-pptx`, and triggers `RapidOCR` *only* on scanned PDFs (`< 30` selectable chars) or document-hinted images.
- **3-Stage Exact Deduplication:** Groups by byte size -> partial edge SHA-256 -> full SHA-256 (`100.00%` Precision/Recall on non-circular seeded duplicate audit) while preserving canonical filenames.
- **Transactional Rollback with Post-Move Edit Protection (`shared/executor.py`):** Logs `sha256_at_move` in SQLite (`operation_journal`) and verifies destination file integrity before reverting batches so post-move user edits are never overwritten.
- **Privacy-Preserving Evaluation:** Personal raw files (`datasets/sandbox_raw/`, `datasets/real_wild_downloads/`) and local SQLite journals are excluded via `.gitignore`; sanitized ground-truth labels and reproducible benchmark metrics are versioned in `datasets/` and `experiments/results/`.

---

## Repository Structure
```text
├── docs/
│   └── 01_rule_based_analysis.md          # Architectural design review, operational safety & Strategy 01 audit report
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
│       └── ml_classifier.py               # Multi-modal TF-IDF + structural stats classifier
├── experiments/
│   ├── evaluate_real_wild_blind.py        # Locked 96-file real-world blind, safety & non-regression harness
│   ├── evaluate_strategy02_ml.py          # Strategy 02A vs 02B real-world benchmark harness
│   └── results/
│       └── strategy01_real_wild_blind_results.json
└── tests/
    ├── test_fault_tolerance.py            # Corrupt PDF, encrypted PDF & out-of-taxonomy fallback tests
    └── test_executor_rollback.py          # Post-move edit & rename rollback safety tests

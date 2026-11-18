@"
# AI-Driven Downloads Folder Organizer — Comparative Strategy Benchmark

An empirical systems and machine learning research project evaluating **five architectural paradigms** for automated, safe, and content-aware organization of unstructured local ``Downloads`` directories.

## Research Objective
Compare deterministic, statistical, semantic, and generative classification pipelines on a real-world ground-truth corpus (``n = 64`` mixed documents, scanned challans, installers, archives, spreadsheets, code, and media) across **Macro F1**, **Accuracy**, **Inference Latency**, **OCR Overhead**, and **Manual Review Burden**.

| Strategy | Paradigm | Status | Overall Accuracy | Macro F1 | Manual Review Queue |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **01. Rule-Based Engine** | Priority YAML Rules + Native Parsers + Conditional ``RapidOCR`` | **Completed** | **``84.38%``** | **``73.79%``** | **``26.6%``** |
| **02. Traditional ML** | TF-IDF + Logistic Regression / LinearSVC / Random Forest | Next | — | — | — |
| **03. Dense Embeddings** | Local Sentence-Transformers + Cosine / MLP Classifier | Planned | — | — | — |
| **04. Local LLM** | Quantized Local SLM via Ollama (Constrained JSON Schema) | Planned | — | — | — |
| **05. Hybrid Cascade** | Rules -> Embeddings -> Local LLM Escalation | Planned | — | — | — |

---

## Key Architectural Highlights
* **3-Tier Lazy Extraction Cascade (``shared/scanner.py``):** Skips pure binaries/media (``0 ms``), parses native digital documents via ``PyMuPDF``, ``python-docx``, ``openpyxl``, and ``python-pptx`` (``1.70 s`` across 64 files), and triggers ``RapidOCR`` *only* on scanned PDFs (``< 30`` selectable chars) or document-hinted images—boosting ``Finance_Invoices`` recall from ``10%`` to ``80%``.
* **3-Stage Exact Deduplication:** Groups by byte size -> partial edge SHA-256 -> full SHA-256 (``100%`` Precision/Recall) while preserving canonical filenames.
* **Transactional Rollback with Post-Move Edit Protection (``shared/executor.py``):** Logs ``sha256_at_move`` in SQLite (``operation_journal``) and verifies destination file integrity before reverting batches so post-move user edits are never overwritten.
* **Privacy-Preserving Evaluation:** Personal raw files (``datasets/sandbox_raw/``) and local SQLite journals are excluded via ``.gitignore``; sanitized ground-truth labels and reproducible benchmark metrics are versioned in ``datasets/ground_truth.csv`` and ``experiments/results/``.

---

## Repository Structure
```text
├── docs/
│   └── 01_rule_based_analysis.md          # Architectural design review, tool comparison & Strategy 01 ablation report
├── shared/
│   ├── models.py                          # FileRecord, ClassificationResult & ActionItem contracts
│   ├── scanner.py                         # 3-tier extraction cascade, conditional RapidOCR & SHA-256 deduplicator
│   ├── base_classifier.py                 # Shared classifier interface across Strategies 01-05
│   ├── planner.py                         # Confidence-tiered dry-run preview & conflict triage
│   └── executor.py                        # Atomic filesystem executor & SHA-256-verified SQLite rollback
├── strategies/
│   └── 01_rule_based/
│       ├── rules.yaml                     # Priority-weighted domain rules & generic-extension guards
│       └── classifier.py                  # Deterministic scorer & tie-conflict detector
├── experiments/
│   ├── evaluate_strategy_01.py            # Ablation benchmark harness (DropIt baseline vs. Native vs. OCR)
│   └── results/
│       └── strategy_01_benchmark.json     # Reproducible metrics & error taxonomy
└── tests/
    ├── test_fault_tolerance.py            # Corrupt PDF, encrypted PDF & out-of-taxonomy fallback tests
    └── test_executor_rollback.py          # Post-move edit & rename rollback safety tests

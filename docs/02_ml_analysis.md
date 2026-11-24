# Strategy 02: Multi-Modal TF-IDF + LSA + Structural Features & Supervised ML Benchmark

## 1. Architectural Design (`strategies/02_ml_classification/ml_classifier.py`)

To overcome Strategy 01's brittle manual regex maintenance, unseen extension blindness (`*.whl`), and lack of global Inverse Document Frequency (IDF) weighting, **Strategy 02** formulates automated file organization as a calibrated, multi-modal supervised classification pipeline (`TraditionalMLClassifier`) built on `scikit-learn`.

### 1.1 Six-Branch `FeatureUnion` Representation
Each `FileRecord` is transformed into a unified sparse+dense feature vector combining lexical, subword, latent semantic, and file-structure signals:

1. **Extension One-Hot Encoder (`ext_ohe`, weight `2.4`):** Captures deterministic container, media, and binary extensions (`.whl`, `.tar.gz`, `.csv`, `.jpg`) with `handle_unknown="ignore"`.
2. **Normalized Filename Word N-Grams (`stem_word_tfidf`, `(1, 2)`, weight `1.3`):** Tokenizes delimiters (`_`, `-`, `.`, parentheses, brackets) into sublinear TF-IDF word n-grams.
3. **Filename Character N-Grams (`stem_char_tfidf`, `char_wb (3, 5)`, weight `1.1`):** Captures morphological roots, version tags, and compound identifiers without requiring manual regex rules.
4. **Extracted Body Word N-Grams (`body_word_tfidf`, `(1, 2)`, `max_features=7000`, weight `1.5`):** Applies sublinear TF-IDF over the first `12,000` characters of extracted native/OCR document text.
5. **Latent Semantic Analysis (`body_lsa_svd`, `TruncatedSVD(n_components=32)`, weight `1.0`):** Projects the body TF-IDF matrix into a 32-dimensional orthogonal semantic subspace scaled via `MinMaxScaler` to capture classical word co-occurrence structure across prose documents.
6. **Domain-Invariant Structural & Empty-Text Indicators (`structural_stats`, `MinMaxScaler`, weight `1.6`):** Extracts 13 continuous and binary indicator signals separating dense numeric tables, financial statements, zero-text images/binaries, and prose documents:
   - Continuous statistics: `log1p(size_bytes)`, `log1p(text_len)`, `digit_ratio`, `uppercase_ratio`, `comma_per_line`, `tab_per_line`, `currency_per_1k`
   - Structural guards: `is_binary_skipped`, `has_empty_text`, `is_image_ext`, `is_archive_ext`, `is_installer_ext`, and `is_data_ext`

### 1.2 Probability Calibration & Operational Safety Gate
- **Candidate Estimators Evaluated via 5-Fold Stratified CV:**
  - `CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), method="sigmoid")`
  - `LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000)`
  - `MultinomialNB(alpha=0.25)`
  - `RandomForestClassifier(n_estimators=250, class_weight="balanced")`
  - `VotingClassifier(voting="soft", weights=[2.0, 1.5, 1.0])` combining `LogisticRegression`, `CalibratedLinearSVC`, and `MultinomialNB`
- **Dual Safety Triage:** Predictions are routed to `needs_review = True` whenever `top_confidence < 0.56` or `(top_confidence - second_confidence) <= 0.14` (recording `conflict_with` for interactive disambiguation).

---

## 2. Synthetic-to-Real Domain Shift (`Strategy 02A`) vs. Disjoint Real-World Training (`Strategy 02B`)

To eliminate data leakage while testing genuine out-of-sample generalization, `datasets/build_ml_train_real.py` and `datasets/topup_career_train.py` harvested real-world public documents across the 7 core categories (`N = 149` combined with `sandbox_raw`) and verified **`0` SHA-256 hash overlap** (`assert len(hashes_02b & locked_wild_hashes) == 0`) with the locked `N = 96` blind test corpus (`datasets/real_wild_ground_truth.csv`).

### 2.1 Five-Fold Stratified Cross-Validation on Real-World Training Set (`n = 149`)
| Candidate Estimator | 5-Fold CV Accuracy | 5-Fold Macro F1 | Mean Fit Latency |
| :--- | :---: | :---: | :---: |
| **Calibrated LinearSVC (`C=1.0`, sigmoid)** | **`95.33%` (±3.40%)** | **`92.61%`** | `213.5 ms` |
| **Logistic Regression (`C=1.5`, balanced)** | `94.64%` (±4.51%) | `92.47%` | `535.3 ms` |
| **Soft-Voting Ensemble (`LR + SVC + NB`)** | `93.98%` (±3.25%) | `91.21%` | `768.2 ms` |
| **Multinomial Naive Bayes (`alpha=0.25`)** | `91.27%` (±5.01%) | `87.78%` | `157.0 ms` |
| **Random Forest (`250` trees, balanced)** | `90.64%` (±6.09%) | `86.21%` | `576.3 ms` |

- **Selected Model:** `CalibratedLinearSVC` (`95.33%` CV Accuracy, `92.61%` CV Macro F1, `3,564.3 KB` serialized artifact saved to `strategies/02_ml_classification/artifacts/best_ml_classifier.joblib`).

### 2.2 Locked Real-World Blind Ablation & Safety Metrics (`n = 85` Unique Files)

| Strategy / Configuration | Overall Acc | Auto-Move Acc | Manual Review Queue | Unflagged Wrong Moves | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy 02A (Synthetic Corpus Baseline, Cfg C)** | `60.00%` (`51/85`) | `78.95%` (`45/57`) | `32.94%` (`28/85`) | `12` (`14.12%`) | `60.60%` |
| **Strategy 02B — Config A:** Metadata-Only | `62.35%` (`53/85`) | `88.89%` (`40/45`) | `47.06%` (`40/85`) | `5` (`5.88%`) | `72.21%` |
| **Strategy 02B — Config B:** + Native Parsing | `83.53%` (`71/85`) | **`96.72%` (`59/61`)** | `28.24%` (`24/85`) | **`2` (`2.35%`)** | `84.93%` |
| **Strategy 02B — Config C:** + Conditional `RapidOCR` | **`87.06%` (`74/85`)** | **`95.31%` (`61/64`)** | **`24.71%` (`21/85`)** | **`3` (`3.53%`)** | **`86.90%`** |

### 2.3 Per-Category Performance (`Strategy 02B`, Config C, `n = 85`)
| Category | Support ($n$) | Precision | Recall | F1-Score | Comparison vs. Strategy 01 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Archives_Compressed`** | 8 | `100.00%` | `100.00%` | **`100.00%`** | Matches Strategy 01 (`100.00%`) |
| **`Media_Photos_Graphics`** | 5 | `100.00%` | `100.00%` | **`100.00%`** | Matches Strategy 01 (`100.00%`); solved via `has_empty_text` + `is_image_ext` |
| **`Software_Installers`** | 11 | `100.00%` | `100.00%` | **`100.00%`** | **`+37.50%` F1** over Strategy 01 (`62.50%` $\rightarrow$ `100.00%`) |
| **`Spreadsheets_Data`** | 11 | `100.00%` | `100.00%` | **`100.00%`** | Matches Strategy 01 (`100.00%`) |
| **`Academic_Study`** | 23 | `87.50%` | `91.30%` | **`89.36%`** | **`+17.56%` F1** over Strategy 01 (`71.80%`) |
| **`Finance_Invoices`** | 17 | `71.43%` | `88.24%` | **`78.95%`** | **`+12.28%` F1** over Strategy 01 (`66.67%`) |
| **`Career_Identity`** | 10 | `60.00%` | `30.00%` | **`40.00%`** | Surface TF-IDF overlaps with Academic (`B.Tech`/`University`) & Finance (`INR`) |

---

## 3. Stage 2 Few-Shot Retraining & Non-Regression Check (`k = 2` Support)

1. **Unseen-Domain Few-Shot Adaptation (`m = 7` Queries):** Retraining `CalibratedLinearSVC` with $k=2$ support files per unseen domain (`Medical_Health`, `Travel_Bookings`) raised unseen query accuracy from **`0.00%` $\rightarrow$ `28.57%`**, with **`100.00%` of unseen queries safely routed to manual review** (`0` silent misroutes). Unlike deterministic regex rules, supervised linear models require larger sample counts to carve reliable hyperplanes for brand-new semantic categories.
2. **Post-Retraining 85-File Non-Regression Check:** Re-evaluating the locked 85-file blind test after incorporating the $k=2$ medical/travel support files preserved **`87.06%` overall accuracy (`+0.00%` delta, zero regressions)** and reduced unflagged wrong moves from **`3` $\rightarrow$ `0` (`0.00%`)**, demonstrating strong L2-regularized resistance to catastrophic forgetting.

---

## 4. Key Architectural Takeaways (Motivating Strategy 03 Dense Embeddings)

1. **Outperforms Rule-Based Engine Across Every Core Metric:** Strategy 02B (`Config C`) achieves **`87.06%` overall accuracy** (`+10.59%` over Strategy 01's peak `76.47%`), **`95.31%` Auto-Move Accuracy**, only **`3` unflagged wrong moves (`3.53%`)**, and **`86.90%` Macro F1**, while reducing the manual review queue from **`32.94%` $\rightarrow$ `24.71%`**.
2. **Real-World Distribution Matching Beats Synthetic Data by `+27.06%`:** Replacing templated synthetic training files (`02A`, `60.00%`) with a SHA-256-disjoint real-world corpus (`02B`, `87.06%`) cut unflagged errors from `12` to `3`.
3. **Why Bag-of-Words Hits a Ceiling on `Career_Identity` (`40.00%` F1) and $k=2$ Few-Shot Domains (`28.57%`):** All `11` remaining misclassifications occur on `.pdf` prose documents—specifically `Career_Identity` (`7` errors) and `Academic_Study` / `Finance_Invoices` boundary overlaps (`4` errors)—where resumes, offer letters, admit cards, and university marksheets share surface vocabulary (`University`, `B.Tech`, `INR`, `CGPA`) with lecture notes and bills, and $k=2$ support examples lack sufficient lexical overlap. This directly motivates **Strategy 03 (Pretrained Dense Sentence Embeddings + Prototype/Cosine Classification)** to capture document-level conceptual similarity without large labeled training sets.
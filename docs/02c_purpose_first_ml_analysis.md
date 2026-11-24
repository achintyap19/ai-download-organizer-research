# Strategy 02C: Purpose-First 14-Category Multi-Modal ML Analysis (`test_set_v1`)

## 1. Motivation: Diagnosing Strategy 02B on the 12-Persona Blind Benchmark (`test_set_v1`)
Evaluating the frozen 7-category **Strategy 02B** (`best_ml_classifier.joblib`) on `datasets/test_set_v1` (`n = 590` unique files across 12 personas, scored via `score.py --legacy`) revealed **58.1% overall accuracy** (`46.3%` macro-F1, `44.0%` auto-move accuracy, `163` unflagged wrong moves) and exposed four structural bottlenecks:

1. **Closed-World Taxonomy Mismatch (`219` `NO_LEGACY_EQUIVALENT` files):** `37.1%` of `test_set_v1` belongs to categories absent from the legacy 7-class taxonomy (`Work_Professional`, `Personal_IDs_Records`, `Travel`, `Health_Medical`, `Home_Household`, `Other`), forcing closed-world misclassifications into `Academic_Study` (`44`) and `Finance_Invoices` (`40`).
2. **Container Format Overriding Document Purpose:** Raw `.png`/`.jpg` and `.xlsx`/`.csv` indicator features pushed payment screenshots, photographed bills, and expense spreadsheets into `Media_Photos_Graphics` (`28`) and `Spreadsheets_Data` (`19`).
3. **Filename-Gated Image OCR:** `shared/scanner.py` previously triggered `RapidOCR` only when filenames matched `DOC_IMAGE_HINT_RE`, leaving camera-named document photos (`IMG_...`, `image (4).png`) without extracted text.
4. **Partial Download Handling:** Browser partial downloads (`.crdownload`, `.part`, `.tmp`) were skipped by the scanner rather than routed to `Other` (`needs_review=True`).

---

## 2. Architectural Upgrades in Strategy 02C (`commit 586dcc7`)
Trained strictly on SHA-256-disjoint seen datasets (`n = 574` records mapped from `ml_train_real`, `real_wild_downloads`, `sandbox_raw`, `locked_blind_test`, and `TAXONOMY.md` anchor descriptions; `1` overlapping hash automatically excluded before training):

1. **Universal Image & Extended Format Extraction (`shared/scanner.py`):** Enabled `RapidOCR` across all raster images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`), added `.ipynb` cell and `.parquet` schema/sample extraction, and tagged `.crdownload`/`.part`/0-byte files as `TEMP_PARTIAL`/`EMPTY_FILE` -> `Other` (`needs_review=True`).
2. **Purpose-Over-Format Extension Gating (`ml_classifier_v2.py`):** Whenever extracted or OCR text has `>= 25` characters, raw image and spreadsheet extension indicators are replaced by `.purpose_text`, forcing classification by document content rather than file extension.
3. **7-Branch Multi-Modal Feature Union + Domain Signal Densities:** Combines effective extension OHE, noise-scrubbed filename stem TF-IDF, document header TF-IDF (first 280 chars), body word `(1,2)` TF-IDF, body `char_wb` `(3,5)` TF-IDF, 36-component TruncatedSVD, and 21 structural + domain keyword density features.
4. **Deterministic Container Guardrails + Soft-Voting Calibration:** Routes pure installers, archives, audio/video containers, and textless images deterministically while classifying the 11 semantic document/purpose classes via a calibrated `LogisticRegression` + `CalibratedLinearSVC` + `MultinomialNB` soft-voting ensemble (`94.1%` accuracy, `91.2%` macro-F1 on 5-fold CV of the seen corpus).

---

## 3. Locked Benchmark Results (`datasets/test_set_v1`, `n = 590` Unique Files)

| Metric | Strategy 02B (`--legacy` 7-cat) | Strategy 02C (Native 14-cat, `586dcc7`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Strict Accuracy** | `58.1%` | **`66.3%`** | **`+8.2%`** |
| **Lenient Accuracy** | `58.1%` | **`81.2%`** *(95% CI: `77.8–84.1%`)* | **`+23.1%`** |
| **Macro-F1** | `46.3%` *(8 legacy buckets)* | **`67.5%`** *(all 14 categories)* | **`+21.2%`** |
| **Auto-Move Accuracy** | `44.0%` (`291` auto-moved) | **`86.7%`** (`496` auto-moved) | **`+42.7%`** |
| **Review Queue %** | `50.7%` | **`15.9%`** | **`-34.8%`** |
| **Unflagged Wrong Moves** | `163` (`27.6%`) | **`66`** (`11.2%`) | **`-97` errors (`-60%`)** |
| **Real Downloads (`n = 76`)** | `75.0%` | **`86.8%`** (`95.5%` auto-move) | **`+11.8%`** |
| **Real Photos (`n = 41`)** | `92.7%` | **`100.0%`** (`100.0%` auto-move) | **`+7.3%`** |
| **Uninformative Names (`n = 228`)** | `66.2%` | **`89.0%`** (`90.2%` auto-move) | **`+22.8%`** |
| **Misleading Names (`n = 25`)** | `40.0%` | **`84.0%`** (`85.0%` auto-move) | **`+44.0%`** |

---

## 4. Residual Error Analysis -> Motivation for Strategy 03 (Dense Embeddings)
1. **Low-Support Class Starvation (`Work_Professional` `R = 5.7%`, `Home_Household` `R = 2.9%`):** Despite `class_weight="balanced"`, bag-of-words TF-IDF favored high-support training classes (`Study_Learning` `n = 98`, `Finance_Money` `n = 90`) over `Work_Professional` (`n = 10`) and `Home_Household` (`n = 6`), causing `36` `Work_Professional -> Study_Learning` and `14` `Home_Household -> Study_Learning` confusions.
2. **Short OCR Snippets on Photos (`26.5%` strict / `91.2%` lenient):** Short OCR fragments without exact training token matches still fell back to `Media_Images` (`16` `Finance_Money -> Media_Images`, `7` `Health_Medical -> Media_Images`).
3. **Hindi / Hinglish Generalization (`37.5%` strict / `67.5%` lenient, `n = 40`):** Sparse n-grams struggle across Devanagari and code-mixed Hindi/English phrasing—motivating **Strategy 03** using a multilingual dense sentence-transformer (`intfloat/multilingual-e5-small` / `paraphrase-multilingual-MiniLM-L12-v2`) with balanced 14-category semantic prototypes.

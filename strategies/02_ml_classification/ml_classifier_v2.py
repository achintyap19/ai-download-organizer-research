from __future__ import annotations

import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.models import ClassificationResult, FileRecord

TOKEN_SPLIT_RE = re.compile(r"[_\-\.\(\)\[\]\s]+")
NOISE_STEM_RE = re.compile(
    r"(?i)\b(?:img|dsc|pxl|vid|aud|ptt|wa\d+|whatsapp|image|photo|screenshot|screen\s*shot|scan|scanned|doc|document|file|untitled|download|copy|final|new)\b|\b\d{4,}\b"
)

TEMP_EXTS = {".crdownload", ".part", ".tmp", ".download"}
INSTALLER_EXTS = {".exe", ".msi", ".msix", ".deb", ".rpm", ".appimage", ".dmg", ".pkg", ".apk", ".whl", ".jar", ".vsix"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".xz", ".bz2"}
MEDIA_AV_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic"}
DATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
CODE_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".cc", ".c", ".h", ".cs", ".go", ".rs", ".sql", ".sh", ".ps1", ".bat", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".ipynb"}

# Purpose-specific domain signal patterns (aligned with TAXONOMY.md)
RE_FINANCE = re.compile(r"[\$\₹\€\£]|\b(?:invoice|gst|gstin|cgst|sgst|tax|itr|form\s*16|tds|receipt|upi|neft|imps|utr|challan|payslip|salary|hra|pf|emi|loan|statement|balance|subtotal|amount\s*payable|premium|policy\s*no|budget|expense)\b", re.I)
RE_STUDY = re.compile(r"\b(?:lecture|syllabus|semester|mid[\s\-]?term|end[\s\-]?sem|question\s*paper|answer\s*key|assignment|homework|lab\s*record|marksheet|transcript|sgpa|cgpa|admit\s*card|hall\s*ticket|convocation|abstract|references|arxiv|theorem|algorithm|chapter)\b", re.I)
RE_CAREER = re.compile(r"\b(?:resume|curriculum\s*vitae|cover\s*letter|hiring\s*manager|offer\s*letter|appointment\s*letter|relieving\s*letter|experience\s*letter|job\s*description|responsibilities|qualifications|stipend|probation|ctc|shortlist|interview|candidate)\b", re.I)
RE_WORK = re.compile(r"\b(?:proposal|executive\s*summary|scope\s*of\s*work|sow|nda|non[\s\-]?disclosure|minutes\s*of\s*meeting|mom|agenda|action\s*items|employee\s*handbook|sop|standard\s*operating|roadmap|sprint|timesheet|memorandum|whitepaper|deliverables|sla)\b", re.I)
RE_IDENTITY = re.compile(r"\b(?:aadhaar|uidai|permanent\s*account\s*number|pan\s*card|passport|driving\s*licence|driver\s*license|voter\s*id|epic|birth\s*certificate|domicile|caste\s*certificate|income\s*certificate|registration\s*certificate|chassis|puc|affidavit|notary|power\s*of\s*attorney)\b", re.I)
RE_TRAVEL = re.compile(r"\b(?:pnr|boarding\s*pass|e[\s\-]?ticket|flight|airline|indigo|air\s*india|vistara|irctc|train\s*no|berth|coach|hotel\s*booking|check[\s\-]?in|check[\s\-]?out|itinerary|visa|embassy|consulate|baggage)\b", re.I)
RE_MEDICAL = re.compile(r"\b(?:pathology|laboratory\s*report|blood\s*count|hemoglobin|platelets|prescription|rx|diagnosis|dosage|paracetamol|discharge\s*summary|hospital|radiology|mri|ct\s*scan|vaccination|cowin|health\s*claim|cashless|patient)\b", re.I)
RE_HOUSEHOLD = re.compile(r"\b(?:rent\s*agreement|lease\s*agreement|landlord|tenant|security\s*deposit|housing\s*society|maintenance\s*notice|user\s*manual|warranty|serial\s*number|parent\s*teacher|ptm|report\s*card|recipe|ingredients|preheat|property\s*tax)\b", re.I)
RE_CODE = re.compile(r"\b(?:import\s+|from\s+\w+\s+import|def\s+\w+|class\s+\w+|function\s*\(|const\s+\w+\s*=|select\s+.+\s+from|create\s+table|version:\s*['\"]?\d|docker|kubernetes|api_key|localhost)\b", re.I)


def _clean_text(r: FileRecord) -> str:
    return (r.extracted_text or "").strip()


def _has_meaningful_text(r: FileRecord) -> bool:
    return len(_clean_text(r)) >= 25


def _sanitize_stem(stem: str) -> str:
    s = TOKEN_SPLIT_RE.sub(" ", stem.lower()).strip()
    s = NOISE_STEM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _effective_extension(r: FileRecord) -> str:
    """
    Purpose-over-Format Gating (TAXONOMY.md Overlap Rules 1-3):
    If an image or spreadsheet has >=25 chars of extracted/OCR text, suppress raw container
    extension bias so the classifier routes by document purpose rather than container format.
    """
    ext = (r.extension or "").lower()
    if _has_meaningful_text(r) and (ext in IMAGE_EXTS or ext in DATA_EXTS):
        return ".purpose_text"
    return ext


def _extract_domain_vector(r: FileRecord) -> List[float]:
    ext = (r.extension or "").lower()
    text = _clean_text(r)[:12000]
    stem = _sanitize_stem(r.stem)
    combined = f"{stem}\n{text}"
    t_len = len(text)
    has_text = 1.0 if t_len >= 25 else 0.0
    lines = max(1, text.count("\n") + 1)
    digits = sum(1 for c in text if c.isdigit())
    uppers = sum(1 for c in text if c.isupper())
    commas = text.count(",")
    tabs = text.count("\t")

    def density(pattern: re.Pattern) -> float:
        hits = len(pattern.findall(combined))
        return min(15.0, float(hits))

    is_img_no_text = 1.0 if (ext in IMAGE_EXTS and not has_text) else 0.0
    is_data_no_text = 1.0 if (ext in DATA_EXTS and not has_text) else 0.0

    return [
        math.log1p(max(0, r.size_bytes)),
        math.log1p(t_len),
        has_text,
        (digits / t_len) if t_len > 0 else 0.0,
        (uppers / t_len) if t_len > 0 else 0.0,
        min(50.0, commas / lines) if t_len > 0 else 0.0,
        min(50.0, tabs / lines) if t_len > 0 else 0.0,
        is_img_no_text,
        is_data_no_text,
        1.0 if ext in CODE_EXTS else 0.0,
        1.0 if r.extraction_status == "OCR_USED" else 0.0,
        1.0 if r.extraction_status in {"PASSWORD_PROTECTED", "CORRUPT_OR_UNREADABLE", "EMPTY_FILE"} else 0.0,
        density(RE_FINANCE),
        density(RE_STUDY),
        density(RE_CAREER),
        density(RE_WORK),
        density(RE_IDENTITY),
        density(RE_TRAVEL),
        density(RE_MEDICAL),
        density(RE_HOUSEHOLD),
        density(RE_CODE),
    ]


class PurposeFieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field: str):
        self.field = field

    def fit(self, X: Sequence[FileRecord], y=None):
        return self

    def transform(self, X: Sequence[FileRecord]):
        if self.field == "effective_ext":
            return [[_effective_extension(r)] for r in X]
        if self.field == "sanitized_stem":
            return [_sanitize_stem(r.stem) for r in X]
        if self.field == "header":
            return [_clean_text(r)[:280] for r in X]
        if self.field == "body":
            return [_clean_text(r)[:12000] for r in X]
        if self.field == "domain_stats":
            return np.array([_extract_domain_vector(r) for r in X], dtype=np.float64)
        raise ValueError(f"Unknown field: {self.field}")


class PurposeFirstMLClassifier:
    """
    Strategy 02C: 14-Category Purpose-First Multi-Modal ML Classifier.
    Combines deterministic container/partial guardrails, purpose-over-format extension gating,
    4-channel TF-IDF (header, body word, body char-wb, sanitized stem), domain signal densities,
    and calibrated probability + margin rejection.
    """

    def __init__(
        self,
        model_type: str = "soft_voting_ensemble",
        confidence_threshold: float = 0.45,
        ambiguity_margin: float = 0.10,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []

    def _build_estimator(self, calib_cv: int = 3):
        lr = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1500,
            random_state=self.random_state,
        )
        svc = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=self.random_state),
            cv=calib_cv,
            method="sigmoid",
        )
        nb = MultinomialNB(alpha=0.2)
        if self.model_type == "logistic_regression":
            return lr
        if self.model_type == "calibrated_linearsvc":
            return svc
        return VotingClassifier(
            estimators=[("lr", lr), ("svc", svc), ("nb", nb)],
            voting="soft",
            weights=[2.2, 1.8, 0.8],
        )

    def _build_pipeline(self, calib_cv: int = 3) -> Pipeline:
        features = FeatureUnion(
            transformer_list=[
                (
                    "ext_ohe",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("effective_ext")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                ),
                (
                    "stem_word",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("sanitized_stem")),
                        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=3500)),
                    ]),
                ),
                (
                    "header_word",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("header")),
                        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=4000)),
                    ]),
                ),
                (
                    "body_word",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("body")),
                        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=9000)),
                    ]),
                ),
                (
                    "body_char",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("body")),
                        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=8000)),
                    ]),
                ),
                (
                    "body_lsa",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("body")),
                        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=5000)),
                        ("svd", TruncatedSVD(n_components=36, random_state=self.random_state)),
                        ("scale", MinMaxScaler()),
                    ]),
                ),
                (
                    "domain_stats",
                    Pipeline([
                        ("extract", PurposeFieldExtractor("domain_stats")),
                        ("scale", MinMaxScaler()),
                    ]),
                ),
            ],
            transformer_weights={
                "ext_ohe": 1.6,
                "stem_word": 1.4,
                "header_word": 1.8,
                "body_word": 1.7,
                "body_char": 1.2,
                "body_lsa": 1.0,
                "domain_stats": 1.9,
            },
        )
        return Pipeline([
            ("features", features),
            ("clf", self._build_estimator(calib_cv=calib_cv)),
        ])

    @staticmethod
    def _deterministic_route(r: FileRecord) -> Optional[Tuple[str, float, bool, str]]:
        ext = (r.extension or "").lower()
        status = r.extraction_status or ""

        if ext in TEMP_EXTS or status == "TEMP_PARTIAL" or r.size_bytes == 0 or status == "EMPTY_FILE":
            return ("Other", 0.99, True, "guardrail:temp_or_empty")
        if ext in INSTALLER_EXTS:
            return ("Software_Installers", 0.99, False, "guardrail:installer_ext")
        if ext in ARCHIVE_EXTS:
            return ("Archives_Compressed", 0.99, False, "guardrail:archive_ext")
        if ext in MEDIA_AV_EXTS:
            return ("Media_Audio_Video", 0.99, False, "guardrail:media_av_ext")

        # Raster image with no meaningful OCR text -> Media_Images (unless filename screams doc)
        if ext in IMAGE_EXTS and not _has_meaningful_text(r):
            stem = _sanitize_stem(r.stem)
            if not any(k in stem for k in ("invoice", "receipt", "bill", "passport", "aadhaar", "pan", "ticket", "boarding", "prescription", "report", "resume", "cv", "certificate")):
                return ("Media_Images", 0.92, False, "guardrail:non_text_image")

        # Corrupted / password-protected PDF with completely uninformative filename -> Other (review)
        if status in {"PASSWORD_PROTECTED", "CORRUPT_OR_UNREADABLE"} and not _sanitize_stem(r.stem):
            return ("Other", 0.65, True, "guardrail:locked_or_corrupt_uninformative")

        return None

    def fit(self, records: Sequence[FileRecord], labels: Sequence[str]) -> "PurposeFirstMLClassifier":
        # Deterministic container classes are handled by _deterministic_route(); train statistical model on the 11 semantic/purpose classes
        det_classes = {"Software_Installers", "Archives_Compressed", "Media_Audio_Video"}
        train_recs = []
        train_lbls = []
        for r, y in zip(records, labels):
            ext = (r.extension or "").lower()
            if y in det_classes or ext in TEMP_EXTS or ext in INSTALLER_EXTS or ext in ARCHIVE_EXTS or ext in MEDIA_AV_EXTS:
                continue
            train_recs.append(r)
            train_lbls.append(y)

        # Ensure every remaining class has >=3 samples so CalibratedClassifierCV(cv=3) is always well-posed
        counts = Counter(train_lbls)
        for cls_name, cnt in list(counts.items()):
            if cnt < 3:
                cls_recs = [r for r, y in zip(train_recs, train_lbls) if y == cls_name]
                for i in range(3 - cnt):
                    train_recs.append(cls_recs[i % len(cls_recs)])
                    train_lbls.append(cls_name)

        self.pipeline = self._build_pipeline(calib_cv=3)
        self.pipeline.fit(train_recs, train_lbls)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def classify(self, record: FileRecord) -> ClassificationResult:
        det = self._deterministic_route(record)
        if det is not None:
            cat, conf, review, matched = det
            return ClassificationResult(
                file_path=record.path,
                category=cat,
                confidence=round(conf, 4),
                tags=[cat.lower(), "strategy_02c"],
                matched_by=matched,
                needs_review=review,
                conflict_with=None,
            )

        if self.pipeline is None:
            raise RuntimeError("Classifier must be fitted before calling classify().")

        probs = self.pipeline.predict_proba([record])[0]
        ranked = np.argsort(probs)[::-1]
        top_idx = int(ranked[0])
        top_cat = self.classes_[top_idx]
        top_conf = float(probs[top_idx])

        second_cat = self.classes_[int(ranked[1])] if len(ranked) > 1 else None
        second_conf = float(probs[int(ranked[1])]) if len(ranked) > 1 else 0.0
        margin = top_conf - second_conf

        needs_review = (top_conf < self.confidence_threshold) or (margin < self.ambiguity_margin)
        if (record.extraction_status or "") in {"PASSWORD_PROTECTED", "CORRUPT_OR_UNREADABLE"}:
            needs_review = True

        return ClassificationResult(
            file_path=record.path,
            category=top_cat,
            confidence=round(top_conf, 4),
            tags=[top_cat.lower(), f"ml_v2_{self.model_type}"],
            matched_by=f"ml_v2:{self.model_type}",
            needs_review=needs_review,
            conflict_with=second_cat if (second_cat and margin < self.ambiguity_margin) else None,
        )

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
                "ambiguity_margin": self.ambiguity_margin,
                "random_state": self.random_state,
                "pipeline": self.pipeline,
                "classes_": self.classes_,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: Path) -> "PurposeFirstMLClassifier":
        payload = joblib.load(model_path)
        obj = cls(
            model_type=payload["model_type"],
            confidence_threshold=payload["confidence_threshold"],
            ambiguity_margin=payload["ambiguity_margin"],
            random_state=payload["random_state"],
        )
        obj.pipeline = payload["pipeline"]
        obj.classes_ = payload["classes_"]
        return obj


def train_and_validate() -> None:
    corpus_path = REPO_ROOT / "datasets/train_14cat_corpus.pkl"
    data = pickle.loads(corpus_path.read_bytes())
    records: List[FileRecord] = data["records"]
    labels: List[str] = data["labels"]

    # 5-fold stratified CV strictly on the seen corpus
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = [""] * len(records)
    oof_confs = [0.0] * len(records)
    oof_margins = [0.0] * len(records)

    for train_idx, val_idx in skf.split(records, labels):
        X_tr = [records[i] for i in train_idx]
        y_tr = [labels[i] for i in train_idx]
        clf = PurposeFirstMLClassifier(model_type="soft_voting_ensemble")
        clf.fit(X_tr, y_tr)
        for i in val_idx:
            res = clf.classify(records[i])
            oof_preds[i] = res.category
            oof_confs[i] = res.confidence

    overall_acc = accuracy_score(labels, oof_preds)
    macro_f1 = f1_score(labels, oof_preds, average="macro")
    print(f"5-Fold CV on Seen 14-Category Corpus (n={len(records)}): Accuracy={overall_acc*100:.1f}%, Macro-F1={macro_f1*100:.1f}%")

    # Fit final model on 100% of seen corpus and freeze artifact
    final_clf = PurposeFirstMLClassifier(
        model_type="soft_voting_ensemble",
        confidence_threshold=0.42,
        ambiguity_margin=0.08,
    )
    final_clf.fit(records, labels)
    out_model = REPO_ROOT / "strategies/02_ml_classification/artifacts/ml_classifier_14cat_v2.joblib"
    final_clf.save(out_model)
    print(f"Saved frozen Strategy 02C artifact -> {out_model}")


if __name__ == "__main__":
    train_and_validate()

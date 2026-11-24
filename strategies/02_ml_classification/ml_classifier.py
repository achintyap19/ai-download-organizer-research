from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from shared.models import ClassificationResult, FileRecord

TOKEN_SPLIT_RE = re.compile(r"[_\-\.\(\)\[\]\s]+")
CURRENCY_RE = re.compile(r"[\$\₹\€\£]|\b(?:usd|inr|eur|gbp|rs\.?|tax|invoice|total|balance)\b", re.I)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"}
INSTALLER_EXTS = {".exe", ".msi", ".whl", ".deb", ".rpm", ".dmg", ".apk"}
DATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


def _normalize_stem(stem: str) -> str:
    cleaned = TOKEN_SPLIT_RE.sub(" ", stem.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _extract_structural_vector(r: FileRecord) -> List[float]:
    ext = (r.extension or "").lower()
    text = (r.extracted_text or "")[:12000]
    t_len = len(text)
    lines = max(1, text.count("\n") + 1)
    digits = sum(1 for c in text if c.isdigit())
    uppers = sum(1 for c in text if c.isupper())
    commas = text.count(",")
    tabs = text.count("\t")
    curr_hits = len(CURRENCY_RE.findall(text)) if t_len > 0 else 0

    return [
        math.log1p(max(0, r.size_bytes)),
        math.log1p(t_len),
        (digits / t_len) if t_len > 0 else 0.0,
        (uppers / t_len) if t_len > 0 else 0.0,
        min(50.0, commas / lines) if t_len > 0 else 0.0,
        min(50.0, tabs / lines) if t_len > 0 else 0.0,
        min(20.0, (curr_hits * 1000.0) / t_len) if t_len > 0 else 0.0,
        1.0 if r.extraction_status == "SKIPPED_BINARY" else 0.0,
        1.0 if t_len == 0 else 0.0,
        1.0 if ext in IMAGE_EXTS else 0.0,
        1.0 if ext in ARCHIVE_EXTS else 0.0,
        1.0 if ext in INSTALLER_EXTS else 0.0,
        1.0 if ext in DATA_EXTS else 0.0,
    ]


class RecordFieldExtractor(BaseEstimator, TransformerMixin):
    """Extracts a specific structured field from FileRecord objects inside a scikit-learn Pipeline."""

    def __init__(self, field: str):
        self.field = field

    def fit(self, X: Sequence[FileRecord], y=None):
        return self

    def transform(self, X: Sequence[FileRecord]):
        if self.field == "extension":
            return [[(r.extension or "").lower()] for r in X]
        if self.field == "stem":
            return [_normalize_stem(r.stem) for r in X]
        if self.field == "body":
            return [(r.extracted_text or "")[:12000] for r in X]
        if self.field == "structural":
            return np.array([_extract_structural_vector(r) for r in X], dtype=np.float64)
        raise ValueError(f"Unsupported field: {self.field}")


class TraditionalMLClassifier:
    """Multi-modal TF-IDF + LSA + Structural Statistics + Supervised ML classifier."""

    SUPPORTED_MODELS = (
        "calibrated_linearsvc",
        "logistic_regression",
        "multinomial_nb",
        "random_forest",
        "soft_voting_ensemble",
    )

    def __init__(
        self,
        model_type: str = "logistic_regression",
        confidence_threshold: float = 0.56,
        ambiguity_margin: float = 0.14,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []

    def _build_estimator(self):
        if self.model_type == "calibrated_linearsvc":
            base = LinearSVC(C=1.0, class_weight="balanced", random_state=self.random_state)
            return CalibratedClassifierCV(base, cv=getattr(self, "_calib_cv", 3), method="sigmoid")
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                C=1.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
        if self.model_type == "multinomial_nb":
            return MultinomialNB(alpha=0.25)
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if self.model_type == "soft_voting_ensemble":
            lr = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=self.random_state)
            svc = CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=self.random_state),
                cv=3,
                method="sigmoid",
            )
            nb = MultinomialNB(alpha=0.25)
            return VotingClassifier(
                estimators=[("lr", lr), ("svc", svc), ("nb", nb)],
                voting="soft",
                weights=[2.0, 1.5, 1.0],
            )
        raise ValueError(self.model_type)

    def _build_pipeline(self) -> Pipeline:
        features = FeatureUnion(
            transformer_list=[
                (
                    "ext_ohe",
                    Pipeline([
                        ("extract", RecordFieldExtractor("extension")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                ),
                (
                    "stem_word_tfidf",
                    Pipeline([
                        ("extract", RecordFieldExtractor("stem")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                max_features=3000,
                            ),
                        ),
                    ]),
                ),
                (
                    "stem_char_tfidf",
                    Pipeline([
                        ("extract", RecordFieldExtractor("stem")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                max_features=4500,
                            ),
                        ),
                    ]),
                ),
                (
                    "body_word_tfidf",
                    Pipeline([
                        ("extract", RecordFieldExtractor("body")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                max_features=7000,
                            ),
                        ),
                    ]),
                ),
                (
                    "body_lsa_svd",
                    Pipeline([
                        ("extract", RecordFieldExtractor("body")),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                max_features=5000,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=32, random_state=self.random_state)),
                        ("scale", MinMaxScaler()),
                    ]),
                ),
                (
                    "structural_stats",
                    Pipeline([
                        ("extract", RecordFieldExtractor("structural")),
                        ("scale", MinMaxScaler()),
                    ]),
                ),
            ],
            transformer_weights={
                "ext_ohe": 2.4,
                "stem_word_tfidf": 1.3,
                "stem_char_tfidf": 1.1,
                "body_word_tfidf": 1.5,
                "body_lsa_svd": 1.0,
                "structural_stats": 1.6,
            },
        )
        return Pipeline([
            ("features", features),
            ("clf", self._build_estimator()),
        ])

    def fit(self, records: Sequence[FileRecord], labels: Sequence[str]) -> "TraditionalMLClassifier":
        from collections import Counter
        min_count = min(Counter(labels).values()) if labels else 3
        self._calib_cv = max(2, min(3, min_count))
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(records), list(labels))
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict_proba(self, records: Sequence[FileRecord]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Classifier must be fitted before calling predict_proba.")
        return self.pipeline.predict_proba(list(records))

    def classify(self, record: FileRecord) -> ClassificationResult:
        probs = self.predict_proba([record])[0]
        ranked_idx = np.argsort(probs)[::-1]
        top_idx = int(ranked_idx[0])
        top_cat = self.classes_[top_idx]
        top_conf = float(probs[top_idx])

        second_cat: Optional[str] = None
        second_conf = 0.0
        if len(ranked_idx) > 1:
            second_idx = int(ranked_idx[1])
            second_cat = self.classes_[second_idx]
            second_conf = float(probs[second_idx])

        margin = top_conf - second_conf
        needs_review = (top_conf < self.confidence_threshold) or (margin <= self.ambiguity_margin)
        conflict_with = second_cat if (second_cat and margin <= self.ambiguity_margin) else None

        return ClassificationResult(
            file_path=record.path,
            category=top_cat,
            confidence=round(top_conf, 4),
            tags=[top_cat.lower(), f"ml_{self.model_type}"],
            matched_by=f"ml:{self.model_type}",
            needs_review=needs_review,
            conflict_with=conflict_with,
        )

    def get_top_features_per_class(self, top_k: int = 8) -> Dict[str, List[Tuple[str, float]]]:
        if self.pipeline is None:
            return {}
        union: FeatureUnion = self.pipeline.named_steps["features"]
        struct_names = [
            "log_size", "log_text_len", "digit_ratio", "upper_ratio",
            "comma_per_line", "tab_per_line", "currency_per_1k", "is_binary",
            "has_empty_text", "is_image_ext", "is_archive_ext", "is_installer_ext", "is_data_ext",
        ]
        feature_names: List[str] = []
        for branch_name, branch_pipe in union.transformer_list:
            step = branch_pipe.steps[-1][1]
            if branch_name == "body_lsa_svd":
                feature_names.extend([f"body_lsa_svd:comp_{i}" for i in range(32)])
            elif hasattr(step, "get_feature_names_out"):
                feature_names.extend([f"{branch_name}:{n}" for n in step.get_feature_names_out()])
            else:
                feature_names.extend([f"{branch_name}:{n}" for n in struct_names])

        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "estimators_"):
            lr_est = clf.estimators_[0]
            coef_matrix = lr_est.coef_ if hasattr(lr_est, "coef_") else None
            if coef_matrix is None:
                return {}
        elif hasattr(clf, "coef_"):
            coef_matrix = clf.coef_
        elif hasattr(clf, "calibrated_classifiers_"):
            coefs = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_ if hasattr(cc.estimator, "coef_")]
            if not coefs:
                return {}
            coef_matrix = np.mean(coefs, axis=0)
        else:
            return {}

        top_features: Dict[str, List[Tuple[str, float]]] = {}
        for idx, cls_name in enumerate(self.classes_):
            weights = coef_matrix[idx]
            top_indices = np.argsort(weights)[::-1][:top_k]
            top_features[cls_name] = [(feature_names[i], round(float(weights[i]), 3)) for i in top_indices]
        return top_features

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
    def load(cls, model_path: Path) -> "TraditionalMLClassifier":
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


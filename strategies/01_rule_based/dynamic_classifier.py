from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import yaml

from shared.base_classifier import BaseClassifier
from shared.models import ClassificationResult, FileRecord

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "all", "can", "will", "one",
    "pdf", "docx", "xlsx", "file", "document", "page", "date", "name", "total",
    "report", "copy", "final", "new", "data", "info", "details"
}


@dataclass
class DynamicRule:
    rule_id: str
    category: str
    priority: int = 50
    extensions: List[str] = field(default_factory=list)
    filename_regex: List[str] = field(default_factory=list)
    content_regex: List[str] = field(default_factory=list)
    exclude_filename_regex: List[str] = field(default_factory=list)
    exclude_content_regex: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(
        default_factory=lambda: {"ext": 0.25, "filename": 0.35, "content": 0.40}
    )
    source: str = "seed"


class DynamicRuleClassifier(BaseClassifier):
    def __init__(
        self,
        seed_rules_path: Path,
        runtime_rules_path: Optional[Path] = None,
        review_threshold: float = 0.55,
        tie_margin: float = 0.10,
    ):
        self.seed_rules_path = seed_rules_path
        self.runtime_rules_path = runtime_rules_path
        self.review_threshold = review_threshold
        self.tie_margin = tie_margin
        self.rules: List[DynamicRule] = []
        self.reload_rules()

    @property
    def name(self) -> str:
        return "strategy_01_dynamic_rules"

    def reload_rules(self) -> None:
        self.rules = []
        if self.seed_rules_path.exists():
            self.rules.extend(self._load_yaml(self.seed_rules_path, "seed"))
        if self.runtime_rules_path and self.runtime_rules_path.exists():
            self.rules.extend(self._load_yaml(self.runtime_rules_path, "runtime"))

    @staticmethod
    def _load_yaml(path: Path, source: str) -> List[DynamicRule]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules = []
        for item in raw.get("rules", []):
            rules.append(
                DynamicRule(
                    rule_id=item["id"],
                    category=item["category"],
                    priority=int(item.get("priority", 50)),
                    extensions=[e.lower() for e in item.get("extensions", [])],
                    filename_regex=item.get("filename_regex", []),
                    content_regex=item.get("content_regex", []),
                    exclude_filename_regex=item.get("exclude_filename_regex", []),
                    exclude_content_regex=item.get("exclude_content_regex", []),
                    weights=item.get("weights", {"ext": 0.25, "filename": 0.35, "content": 0.40}),
                    source=source,
                )
            )
        return rules

    def _score_rule(self, rule: DynamicRule, rec: FileRecord) -> Tuple[float, List[str]]:
        # Negative collision guards
        for pat in rule.exclude_filename_regex:
            if re.search(pat, rec.stem):
                return 0.0, []
        if rec.extracted_text:
            for pat in rule.exclude_content_regex:
                if re.search(pat, rec.extracted_text):
                    return 0.0, []

        score = 0.0
        reasons: List[str] = []

        ext_matched = bool(rule.extensions and rec.extension in rule.extensions)
        if ext_matched:
            score += rule.weights.get("ext", 0.25)
            reasons.append(f"ext:{rec.extension}")

        fn_hits = 0
        for pat in rule.filename_regex:
            if re.search(pat, rec.stem):
                fn_hits += 1
                reasons.append(f"fn_re:/{pat}/")
        if fn_hits:
            score += rule.weights.get("filename", 0.35) * min(1.35, 1.0 + 0.15 * (fn_hits - 1))

        body_hits = 0
        if rec.extracted_text:
            for pat in rule.content_regex:
                if re.search(pat, rec.extracted_text):
                    body_hits += 1
                    reasons.append(f"body_re:/{pat}/")
        if body_hits:
            score += rule.weights.get("content", 0.40) * min(1.40, 1.0 + 0.20 * (body_hits - 1))

        # Generic document extension guard
        # 1. Header boost: matches in the first 250 chars (document title/header) carry stronger signal
        if body_hits > 0 and rec.extracted_text:
            header_text = rec.extracted_text[:250]
            if any(re.search(rx, header_text) for rx in rule.content_regex):
                score += 0.12
                reasons.append("header_match")

        # 2. Cross-modal corroboration bonus (filename + body both confirm same category)
        if fn_hits > 0 and body_hits > 0:
            score += 0.10
            reasons.append("fn_body_corroborated")

        # 3. Ambiguous generic document guards
        generic_docs = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"}
        if ext_matched and rec.extension in generic_docs:
            if fn_hits == 0 and body_hits == 0:
                score = min(score, 0.25)
            elif body_hits == 0 and len(rec.extracted_text.strip()) > 20:
                # Filename-only guess unconfirmed by readable body text -> always hold for review
                score = min(score, 0.48)
                reasons.append("unconfirmed_fn_only")

        return round(min(score, 1.0), 4), reasons

    def rank_categories(self, rec: FileRecord) -> List[Tuple[str, float, int, str, List[str]]]:
        best_per_cat: Dict[str, Tuple[float, int, str, List[str]]] = {}
        for rule in self.rules:
            sc, reasons = self._score_rule(rule, rec)
            if sc <= 0.0:
                continue
            prev = best_per_cat.get(rule.category)
            if prev is None or (sc, rule.priority) > (prev[0], prev[1]):
                best_per_cat[rule.category] = (sc, rule.priority, rule.rule_id, reasons)

        ranked = [(cat, v[0], v[1], v[2], v[3]) for cat, v in best_per_cat.items()]
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return ranked

    def classify(self, record: FileRecord) -> ClassificationResult:
        ranked = self.rank_categories(record)
        if not ranked:
            return ClassificationResult(
                file_path=record.path,
                category="Uncategorized_Misc",
                confidence=0.0,
                matched_by="none",
                tags=["no_rule_matched"],
                needs_review=True,
            )

        top_cat, top_score, _, top_rule_id, top_reasons = ranked[0]
        conflict_cat = None
        needs_review = top_score < self.review_threshold or record.extraction_status in {
            "PASSWORD_PROTECTED", "CORRUPT_OR_UNREADABLE"
        }

        if len(ranked) > 1:
            sec_cat, sec_score, _, _, _ = ranked[1]
            if (top_score - sec_score) <= self.tie_margin:
                needs_review = True
                conflict_cat = sec_cat

        return ClassificationResult(
            file_path=record.path,
            category=top_cat,
            confidence=top_score,
            matched_by=f"rule:{top_rule_id}",
            tags=top_reasons,
            needs_review=needs_review,
            conflict_with=conflict_cat,
        )

    def add_runtime_rule(
        self,
        category: str,
        extensions: Optional[List[str]] = None,
        filename_regex: Optional[List[str]] = None,
        content_regex: Optional[List[str]] = None,
        priority: int = 86,
    ) -> DynamicRule:
        existing = {"rules": []}
        if self.runtime_rules_path and self.runtime_rules_path.exists():
            existing = yaml.safe_load(self.runtime_rules_path.read_text(encoding="utf-8")) or {"rules": []}

        rule_id = f"rt_{category.lower()}_{len(existing['rules']) + 1:03d}"
        existing["rules"].append({
            "id": rule_id,
            "category": category,
            "priority": priority,
            "extensions": extensions or [],
            "filename_regex": filename_regex or [],
            "content_regex": content_regex or [],
            "weights": {"ext": 0.15, "filename": 0.35, "content": 0.50},
        })
        if self.runtime_rules_path:
            self.runtime_rules_path.parent.mkdir(parents=True, exist_ok=True)
            self.runtime_rules_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
        self.reload_rules()
        return self.rules[-1]

    @staticmethod
    def induce_runtime_regex(records: List[FileRecord], min_doc_freq: int = 2, top_k: int = 4) -> Tuple[List[str], List[str]]:
        """Synthesizes generalizable regex patterns from a cluster of unseen files at runtime."""
        fn_df: Counter = Counter()
        body_df: Counter = Counter()

        for rec in records:
            fn_toks = {t for t in re.findall(r"[A-Za-z]{3,}", rec.stem.lower()) if t not in STOPWORDS}
            for t in fn_toks:
                fn_df[t] += 1
            if rec.extracted_text:
                body_toks = {t for t in re.findall(r"[A-Za-z]{4,}", rec.extracted_text.lower()) if t not in STOPWORDS}
                for t in body_toks:
                    body_df[t] += 1

        fn_terms = [t for t, c in fn_df.most_common(top_k) if c >= min_doc_freq]
        body_terms = [t for t, c in body_df.most_common(top_k) if c >= min_doc_freq]

        fn_re = [rf"(?i)\b({'|'.join(re.escape(t) for t in fn_terms)})\b"] if fn_terms else []
        body_re = [rf"(?i)\b({'|'.join(re.escape(t) for t in body_terms)})\b"] if body_terms else []
        return fn_re, body_re

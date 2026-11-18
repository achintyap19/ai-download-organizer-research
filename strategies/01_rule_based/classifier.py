import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

from shared.base_classifier import BaseClassifier
from shared.models import ClassificationResult, FileRecord


class RuleBasedClassifier(BaseClassifier):
    """
    Approach 01: Deterministic Priority-Weighted Rule Engine.
    - content_aware=False: Simulates DropIt-style metadata + filename regex matching.
    - content_aware=True : Uses extension + filename regex + extracted text/OCR keywords.
    """

    def __init__(self, rules_path: Path | None = None, content_aware: bool = True):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.yaml"
        with open(rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.content_aware = content_aware
        self.min_confidence: float = config.get("min_confidence_threshold", 0.45)
        self.generic_doc_exts = {ext.lower() for ext in config.get("generic_doc_extensions", [])}
        self.rules: List[Dict[str, Any]] = sorted(
            config.get("rules", []),
            key=lambda r: r.get("priority", 0),
            reverse=True,
        )
        for rule in self.rules:
            pattern = rule.get("filename_regex")
            rule["_compiled_regex"] = re.compile(pattern) if pattern else None
            rule["_ext_set"] = {ext.lower() for ext in rule.get("extensions", [])}
            rule["_keywords"] = [kw.lower() for kw in rule.get("content_keywords", [])]

    @property
    def name(self) -> str:
        return "01_RuleBased_ContentAware" if self.content_aware else "01_RuleBased_DropItBaseline"

    def classify(self, record: FileRecord) -> ClassificationResult:
        scored_candidates: List[Tuple[float, Dict[str, Any], List[str]]] = []
        text_lower = record.extracted_text.lower() if (self.content_aware and record.extracted_text) else ""

        for rule in self.rules:
            score = 0.0
            reasons = []
            semantic_signal_found = False

            if record.extension in rule["_ext_set"]:
                score += 0.45
                reasons.append(f"ext({record.extension})")

            regex = rule["_compiled_regex"]
            if regex and regex.search(record.stem):
                score += 0.35
                reasons.append("filename_regex")
                semantic_signal_found = True

            if text_lower and rule["_keywords"]:
                hits = [kw for kw in rule["_keywords"] if kw in text_lower]
                if hits:
                    score += 0.35 if len(hits) >= 2 else 0.25
                    reasons.append(f"keywords({','.join(hits[:2])})")
                    semantic_signal_found = True

            # Guard: Generic document extensions (.pdf, .docx, .xlsx) alone without any filename
            # or content keyword match should NOT blindly trigger a domain folder.
            if record.extension in self.generic_doc_exts and not semantic_signal_found:
                score = min(score, 0.30)

            if score > 0:
                effective_score = min(1.0, score + (rule.get("priority", 0) * 0.0001))
                scored_candidates.append((effective_score, rule, reasons))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        if scored_candidates and scored_candidates[0][0] >= self.min_confidence:
            best_score, best_rule, best_reasons = scored_candidates[0]
            conflict_with = None
            is_ambiguous = False

            # Detect close ties (runner-up within 0.10 score points)
            if len(scored_candidates) > 1:
                runner_score, runner_rule, _ = scored_candidates[1]
                if (best_score - runner_score) <= 0.10 and runner_score >= self.min_confidence:
                    conflict_with = runner_rule["category"]
                    is_ambiguous = True

            unreadable_flag = record.extraction_status in {"PASSWORD_PROTECTED", "CORRUPT_OR_UNREADABLE"}
            needs_review = (best_score < 0.70) or is_ambiguous or unreadable_flag

            return ClassificationResult(
                file_path=record.path,
                category=best_rule["category"],
                confidence=round(best_score, 4),
                tags=list(best_rule.get("tags", [])),
                suggested_name=record.name,
                matched_by=f"rule:{best_rule['category']} [{'+'.join(best_reasons)}]",
                needs_review=needs_review,
                conflict_with=conflict_with,
            )

        return ClassificationResult(
            file_path=record.path,
            category="Uncategorized_Misc",
            confidence=round(scored_candidates[0][0], 4) if scored_candidates else 0.0,
            tags=["uncategorized"],
            suggested_name=record.name,
            matched_by=f"rule:fallback_uncategorized ({record.extraction_status})",
            needs_review=True,
        )
from pathlib import Path
from typing import Dict, List
from .models import ActionItem, ClassificationResult, FileRecord


def build_action_plan(
    records: List[FileRecord],
    results: List[ClassificationResult],
    destination_root: Path,
    quarantine_duplicates: bool = True,
) -> List[ActionItem]:
    plan: List[ActionItem] = []
    reserved_paths: Dict[Path, int] = {}

    for rec, res in zip(records, results):
        if quarantine_duplicates and rec.is_duplicate:
            effective_category = "_Duplicates_Review"
            tags = list(set(res.tags + ["exact-duplicate"]))
            matched_by = f"sha256_duplicate_of:{rec.duplicate_of.name if rec.duplicate_of else 'unknown'}"
        else:
            effective_category = res.category
            tags = res.tags
            matched_by = res.matched_by

        target_dir = destination_root / effective_category
        base_name = res.suggested_name or rec.name
        candidate = target_dir / base_name

        counter = 1
        stem, suffix = Path(base_name).stem, Path(base_name).suffix
        while candidate.exists() or candidate in reserved_paths:
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        reserved_paths[candidate] = 1
        plan.append(
            ActionItem(
                source_path=rec.path,
                destination_path=candidate,
                category=effective_category,
                confidence=1.0 if rec.is_duplicate else res.confidence,
                tags=tags,
                matched_by=matched_by,
                is_duplicate=rec.is_duplicate,
            )
        )
    return plan

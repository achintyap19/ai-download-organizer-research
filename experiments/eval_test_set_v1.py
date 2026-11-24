import argparse
import csv
import inspect
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (
    REPO_ROOT,
    REPO_ROOT / "strategies" / "01_rule_based",
    REPO_ROOT / "strategies" / "02_ml_classification",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.scanner import scan_directory


def scan_persona(persona_dir: pathlib.Path):
    params = inspect.signature(scan_directory).parameters
    kwargs = {}
    for key in ("ocr_enabled", "use_ocr", "enable_ocr"):
        if key in params:
            kwargs[key] = True
            break
    for key in ("deduplicate", "dedup"):
        if key in params:
            kwargs[key] = False
            break
    return scan_directory(persona_dir, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["01", "02", "02c"], required=True)
    ap.add_argument("--test-root", default="datasets/test_set_v1")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    test_root = REPO_ROOT / args.test_root
    files_dir = test_root / "files"
    out_csv = REPO_ROOT / (args.out_csv or f"experiments/results/strategy{args.strategy}_test_set_v1_predictions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.strategy == "02c":
        import __main__
        from ml_classifier_v2 import PurposeFieldExtractor, PurposeFirstMLClassifier
        __main__.PurposeFieldExtractor = PurposeFieldExtractor
        model_path = REPO_ROOT / "strategies/02_ml_classification/artifacts/ml_classifier_14cat_v2.joblib"
        clf = PurposeFirstMLClassifier.load(model_path)
    elif args.strategy == "02":
        from ml_classifier import TraditionalMLClassifier
        model_path = REPO_ROOT / "strategies/02_ml_classification/artifacts/best_ml_classifier.joblib"
        clf = TraditionalMLClassifier.load(model_path)
    else:
        from rule_classifier import RuleBasedClassifier
        rules_path = REPO_ROOT / "strategies/01_rule_based/rules.yaml"
        clf = RuleBasedClassifier(rules_path)

    rows = []
    t0 = time.perf_counter()
    for persona_dir in sorted(files_dir.iterdir()):
        if not persona_dir.is_dir():
            continue
        records = scan_persona(persona_dir)
        for rec in records:
            res = clf.classify(rec)
            rel = pathlib.Path(rec.path).relative_to(test_root).as_posix()
            rows.append({
                "relative_path": rel,
                "predicted_category": res.category,
                "needs_review": str(res.needs_review).lower(),
            })

    elapsed = time.perf_counter() - t0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "predicted_category", "needs_review"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Strategy {args.strategy}] Predicted {len(rows)} files across 12 personas in {elapsed:.2f}s -> {out_csv}")


if __name__ == "__main__":
    main()

import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from shared.models import FileRecord
from shared.scanner import detect_exact_duplicates, scan_directory

# Load frozen Strategy 01 module
spec = importlib.util.spec_from_file_location(
    "dynamic_classifier",
    Path("strategies/01_rule_based/dynamic_classifier.py"),
)
dyn_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dyn_mod)
DynamicRuleClassifier = dyn_mod.DynamicRuleClassifier

RAW_DIR = Path("datasets/real_wild_downloads")
GT_CSV = Path("datasets/real_wild_ground_truth.csv")
SEED_RULES = Path("strategies/01_rule_based/seed_rules.yaml")
OUT_JSON = Path("experiments/results/strategy01_real_wild_blind_results.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_metrics(y_true: list[str], y_pred: list[str], needs_review: list[bool]) -> dict:
    labels = sorted(set(y_true))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0
    review_rate = sum(1 for r in needs_review if r) / len(needs_review) if needs_review else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {
            "support": sum(1 for t in y_true if t == lbl),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "n": len(y_true),
        "correct": correct,
        "accuracy_pct": round(accuracy * 100, 2),
        "macro_precision_pct": round((sum(precisions) / len(labels)) * 100, 2),
        "macro_recall_pct": round((sum(recalls) / len(labels)) * 100, 2),
        "macro_f1_pct": round((sum(f1s) / len(labels)) * 100, 2),
        "review_queue_pct": round(review_rate * 100, 2),
        "per_class": per_class,
    }


def run_ablation_config(
    config_name: str,
    gt_rows: list[dict],
    extract_text: bool,
    use_ocr: bool,
) -> tuple[dict, list[dict], dict[str, FileRecord]]:
    t0 = time.perf_counter()
    records = scan_directory(RAW_DIR, extract_text=extract_text, use_ocr=use_ocr)
    scan_ms = (time.perf_counter() - t0) * 1000.0

    # Verify SHA-256 uniqueness across all scanned records before scoring
    by_hash: dict[str, FileRecord] = {}
    for rec in records:
        digest = rec.sha256 or sha256_file(rec.path)
        if digest not in by_hash:
            by_hash[digest] = rec

    clf = DynamicRuleClassifier(seed_rules_path=SEED_RULES, runtime_rules_path=None)

    y_true, y_pred, reviews, errors = [], [], [], []
    status_counts = Counter()

    t1 = time.perf_counter()
    for row in gt_rows:
        if row["split"] != "blind_test":
            continue
        rec = by_hash[row["sha256"]]
        status_counts[rec.extraction_status] += 1
        res = clf.classify(rec)
        y_true.append(row["true_category"])
        y_pred.append(res.category)
        reviews.append(res.needs_review)
        if res.category != row["true_category"]:
            errors.append({
                "filename": row["filename"],
                "source_domain": row["source_domain"],
                "true_category": row["true_category"],
                "predicted": res.category,
                "confidence": round(res.confidence, 3),
                "matched_by": res.matched_by,
                "extraction_status": rec.extraction_status,
                "needs_review": res.needs_review,
            })
    infer_ms = (time.perf_counter() - t1) * 1000.0

    metrics = compute_metrics(y_true, y_pred, reviews)
    metrics["config"] = config_name
    metrics["scan_time_ms"] = round(scan_ms, 1)
    metrics["infer_time_ms"] = round(infer_ms, 1)
    metrics["extraction_statuses"] = dict(status_counts)
    return metrics, errors, by_hash


def run_disjoint_stage2_eval(gt_rows: list[dict], by_hash: dict[str, FileRecord]) -> dict:
    support_by_cat: dict[str, list[FileRecord]] = defaultdict(list)
    query_rows: list[dict] = []

    for row in gt_rows:
        if row["split"] == "stage2_support":
            support_by_cat[row["true_category"]].append(by_hash[row["sha256"]])
        elif row["split"] == "stage2_query":
            query_rows.append(row)

    # 1. Baseline: Frozen seed rules BEFORE seeing any support examples
    clf_before = DynamicRuleClassifier(seed_rules_path=SEED_RULES, runtime_rules_path=None)
    y_true = [r["true_category"] for r in query_rows]
    preds_before = [clf_before.classify(by_hash[r["sha256"]]) for r in query_rows]
    before_metrics = compute_metrics(
        y_true,
        [p.category for p in preds_before],
        [p.needs_review for p in preds_before],
    )

    # 2. Induce runtime rules STRICTLY from the k=2 labeled support files per category
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_rt_yaml = Path(tmpdir) / "runtime_rules_eval.yaml"
        clf_after = DynamicRuleClassifier(seed_rules_path=SEED_RULES, runtime_rules_path=temp_rt_yaml)
        induced_rules_log = {}

        for cat, sup_recs in sorted(support_by_cat.items()):
            fn_re, body_re = DynamicRuleClassifier.induce_runtime_regex(sup_recs, min_doc_freq=2, top_k=5)
            rule = clf_after.add_runtime_rule(
                category=cat,
                extensions=[".pdf"],
                filename_regex=fn_re,
                content_regex=body_re,
                priority=86,
            )
            induced_rules_log[cat] = {
                "k_support_files": len(sup_recs),
                "support_filenames": [r.name for r in sup_recs],
                "induced_filename_regex": fn_re,
                "induced_content_regex": body_re,
                "rule_id": getattr(rule, "rule_id", getattr(rule, "name", f"rt_{cat.lower()}")),
            }

        # 3. Evaluate strictly on the DISJOINT unseen query set (m=7 files)
        preds_after = [clf_after.classify(by_hash[r["sha256"]]) for r in query_rows]
        after_metrics = compute_metrics(
            y_true,
            [p.category for p in preds_after],
            [p.needs_review for p in preds_after],
        )

        query_details = []
        for r, pb, pa in zip(query_rows, preds_before, preds_after):
            query_details.append({
                "filename": r["filename"],
                "true_category": r["true_category"],
                "before_pred": pb.category,
                "before_conf": round(pb.confidence, 3),
                "after_pred": pa.category,
                "after_conf": round(pa.confidence, 3),
                "after_matched_by": pa.matched_by,
            })

    return {
        "k_labeled_per_new_category": 2,
        "total_labeled_support_files": sum(len(v) for v in support_by_cat.values()),
        "disjoint_unseen_query_files": len(query_rows),
        "before_adaptation": before_metrics,
        "after_k2_adaptation": after_metrics,
        "induced_rules": induced_rules_log,
        "query_predictions": query_details,
    }


def run_non_circular_duplicate_audit(gt_rows: list[dict]) -> dict:
    # Construct 10 independent copy pairs outside the detector and record ground-truth pairs first
    selected = gt_rows[:10]
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_dir = Path(tmpdir)
        expected_dup_names = set()
        for idx, row in enumerate(selected, 1):
            src = RAW_DIR / row["relative_path"]
            orig_dst = audit_dir / row["filename"]
            copy_name = f"Copy_{idx:02d}_final_{row['filename']}"
            copy_dst = audit_dir / copy_name
            shutil.copy2(src, orig_dst)
            shutil.copy2(src, copy_dst)
            expected_dup_names.add(copy_name)

        audit_records = scan_directory(audit_dir, extract_text=False, use_ocr=False)
        detect_exact_duplicates(audit_records)

        detected_dup_names = {r.name for r in audit_records if r.is_duplicate}
        tp = len(expected_dup_names & detected_dup_names)
        fp = len(detected_dup_names - expected_dup_names)
        fn = len(expected_dup_names - detected_dup_names)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "total_audit_files": len(selected) * 2,
        "seeded_duplicate_pairs": len(selected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": round(prec * 100, 2),
        "recall_pct": round(rec * 100, 2),
        "f1_pct": round(f1 * 100, 2),
    }


def main() -> None:
    with GT_CSV.open("r", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    print("=== RUNNING SINGLE-RUN FROZEN BENCHMARK ON 96 REAL-WORLD UNIQUE FILES ===")
    cfg_a, _, _ = run_ablation_config("Config A: Metadata-Only", gt_rows, extract_text=False, use_ocr=False)
    cfg_b, _, _ = run_ablation_config("Config B: + Native Parsing", gt_rows, extract_text=True, use_ocr=False)
    cfg_c, errors_c, by_hash_c = run_ablation_config("Config C: + Conditional RapidOCR", gt_rows, extract_text=True, use_ocr=True)

    stage2_res = run_disjoint_stage2_eval(gt_rows, by_hash_c)
    dup_audit_res = run_non_circular_duplicate_audit(gt_rows)

    report = {
        "benchmark_dataset": "real_wild_ground_truth.csv (n=96 unique non-template files)",
        "stage1_blind_test_ablation": [cfg_a, cfg_b, cfg_c],
        "stage1_config_c_misclassifications": errors_c,
        "stage2_disjoint_few_shot_adaptation": stage2_res,
        "non_circular_duplicate_audit": dup_audit_res,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- STAGE 1: REAL-WORLD BLIND TEST (n = 85 UNIQUE FILES, 7 CATEGORIES) ---")
    for cfg in [cfg_a, cfg_b, cfg_c]:
        print(
            f"{cfg['config']:34s} | Acc: {cfg['accuracy_pct']:6.2f}% ({cfg['correct']}/{cfg['n']}) | "
            f"Macro F1: {cfg['macro_f1_pct']:6.2f}% | Review: {cfg['review_queue_pct']:5.2f}% | "
            f"Scan: {cfg['scan_time_ms']} ms"
        )

    print("\n--- CONFIG C PER-CATEGORY BREAKDOWN ---")
    for cat, m in cfg_c["per_class"].items():
        print(f"  {cat:22s} (n={m['support']:2d}) | P: {m['precision']:6.2f}% | R: {m['recall']:6.2f}% | F1: {m['f1']:6.2f}%")

    print(f"\n--- CONFIG C MISCLASSIFICATIONS ({len(errors_c)} files) ---")
    for err in errors_c:
        print(
            f"  {err['filename']} [{err['source_domain']}]: True={err['true_category']} -> "
            f"Pred={err['predicted']} (conf={err['confidence']}, via={err['matched_by']}, status={err['extraction_status']})"
        )

    print("\n--- STAGE 2: DISJOINT FEW-SHOT ADAPTATION (k=2 Support -> m=7 Unseen Query) ---")
    print(
        f"Before Adaptation (Frozen Seed): Acc = {stage2_res['before_adaptation']['accuracy_pct']:.2f}% | "
        f"Macro F1 = {stage2_res['before_adaptation']['macro_f1_pct']:.2f}%"
    )
    print(
        f"After k=2 Support Induction:     Acc = {stage2_res['after_k2_adaptation']['accuracy_pct']:.2f}% | "
        f"Macro F1 = {stage2_res['after_k2_adaptation']['macro_f1_pct']:.2f}% "
        f"(Labeling cost: k={stage2_res['k_labeled_per_new_category']} files/domain)"
    )
    for q in stage2_res["query_predictions"]:
        print(
            f"  {q['filename']} | True={q['true_category']} | "
            f"Before={q['before_pred']} ({q['before_conf']}) -> After={q['after_pred']} ({q['after_conf']})"
        )

    print("\n--- NON-CIRCULAR DUPLICATE AUDIT (10 Seeded Pairs) ---")
    print(
        f"Precision: {dup_audit_res['precision_pct']:.2f}% | Recall: {dup_audit_res['recall_pct']:.2f}% | "
        f"F1: {dup_audit_res['f1_pct']:.2f}% (TP={dup_audit_res['true_positives']}, FP={dup_audit_res['false_positives']}, FN={dup_audit_res['false_negatives']})"
    )
    print(f"\nSaved full JSON report to {OUT_JSON}")


if __name__ == "__main__":
    main()

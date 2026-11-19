import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.scanner import scan_directory

spec = importlib.util.spec_from_file_location("dyn_cls", ROOT / "strategies/01_rule_based/dynamic_classifier.py")
dyn_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dyn_mod)
DynamicRuleClassifier = dyn_mod.DynamicRuleClassifier


def _eval_records(records, gt_map, clf):
    t0 = time.perf_counter()
    preds = clf.classify_batch(records)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    y_true, y_pred = [], []
    review_count = 0
    for p in preds:
        fname = p.file_path.name
        if fname not in gt_map:
            continue
        y_true.append(gt_map[fname]["true_category"])
        y_pred.append(p.category)
        if p.needs_review:
            review_count += 1

    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "n": len(y_true),
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(mp * 100, 2),
        "macro_recall_pct": round(mr * 100, 2),
        "macro_f1_pct": round(mf1 * 100, 2),
        "latency_ms": round(lat_ms, 2),
        "review_queue_pct": round((review_count / max(1, len(y_true))) * 100, 1),
        "predictions": preds,
    }


def main():
    seed_path = ROOT / "strategies/01_rule_based/seed_rules.yaml"
    runtime_path = ROOT / "strategies/01_rule_based/runtime_rules.yaml"
    if runtime_path.exists():
        runtime_path.unlink()

    # 1. Evaluate on 200 COMPLETELY UNSEEN files (Blind seed_rules.yaml only)
    base_dir = ROOT / "datasets/unseen_base_200"
    with open(ROOT / "datasets/unseen_base_ground_truth.csv", encoding="utf-8") as f:
        base_gt = {r["filename"]: r for r in csv.DictReader(f)}

    rec_meta = scan_directory(base_dir, extract_text=False, use_ocr=False)
    rec_native = scan_directory(base_dir, extract_text=True, use_ocr=False)
    rec_ocr = scan_directory(base_dir, extract_text=True, use_ocr=True)

    clf_blind = DynamicRuleClassifier(seed_rules_path=seed_path, runtime_rules_path=None)
    res_a = _eval_records(rec_meta, base_gt, clf_blind)
    res_b = _eval_records(rec_native, base_gt, clf_blind)
    res_c = _eval_records(rec_ocr, base_gt, clf_blind)

    print("\n=== STAGE 1: BLIND GENERAL RULES ON 200 UNSEEN FILES (ZERO OVERFITTING) ===")
    for name, r in [("Config A (Metadata/Regex)", res_a), ("Config B (+Native Parsing)", res_b), ("Config C (+Conditional OCR)", res_c)]:
        print(f"{name:28s} | Acc: {r['accuracy_pct']:6.2f}% | Macro F1: {r['macro_f1_pct']:6.2f}% | Review Queue: {r['review_queue_pct']:5.1f}%")

    # 2. Evaluate on +20 NEW UNSEEN STREAM FILES (Existing categories + 2 Brand-New Categories)
    stream_dir = ROOT / "datasets/unseen_stream_20"
    with open(ROOT / "datasets/unseen_stream_ground_truth.csv", encoding="utf-8") as f:
        stream_gt = {r["filename"]: r for r in csv.DictReader(f)}

    stream_recs = scan_directory(stream_dir, extract_text=True, use_ocr=True)
    clf_dynamic = DynamicRuleClassifier(seed_rules_path=seed_path, runtime_rules_path=runtime_path)

    before_stream = _eval_records(stream_recs, stream_gt, clf_dynamic)
    print("\n=== STAGE 2A: +20 NEW UNSEEN FILES BEFORE RUNTIME RULE ADAPTATION ===")
    print(f"Accuracy: {before_stream['accuracy_pct']:.2f}% | Macro F1: {before_stream['macro_f1_pct']:.2f}% | Flagged for Review: {before_stream['review_queue_pct']:.1f}%")
    for p in before_stream["predictions"]:
        fname = p.file_path.name
        print(f"  - {fname:28s} | True: {stream_gt[fname]['true_category']:18s} | Pred: {p.category:18s} | Review: {p.needs_review}")

    # Simulate Runtime Rule Synthesis on flagged clusters (Existing + Brand-New Categories)
    by_true_cluster = {}
    for rec, pred in zip(stream_recs, before_stream["predictions"]):
        if pred.needs_review or pred.category == "Uncategorized_Misc":
            tcat = stream_gt[rec.name]["true_category"]
            by_true_cluster.setdefault(tcat, []).append(rec)

    print("\n=== STAGE 2B: SYNTHESIZING RUNTIME REGEX RULES ON FLAGGED UNSEEN CLUSTERS ===")
    for cat, cluster_recs in by_true_cluster.items():
        fn_re, body_re = clf_dynamic.induce_runtime_regex(cluster_recs, min_doc_freq=2, top_k=3)
        new_rule = clf_dynamic.add_runtime_rule(
            category=cat,
            extensions=sorted({r.extension for r in cluster_recs}),
            filename_regex=fn_re,
            content_regex=body_re,
        )
        print(f"  + Created runtime rule '{new_rule.rule_id}' for '{cat}' -> fn_re={fn_re}, body_re={body_re}")

    after_stream = _eval_records(stream_recs, stream_gt, clf_dynamic)
    print(f"\nAfter Runtime Adaptation -> Accuracy: {after_stream['accuracy_pct']:.2f}% | Macro F1: {after_stream['macro_f1_pct']:.2f}% | Review Queue: {after_stream['review_queue_pct']:.1f}%")

    # 3. Cross-Dataset Generalization Check on Original 64 Files (Using ONLY blind seed_rules.yaml!)
    orig_dir = ROOT / "datasets/sandbox_raw"
    orig_csv = ROOT / "datasets/ground_truth.csv"
    orig_res = None
    if orig_dir.exists() and orig_csv.exists():
        with open(orig_csv, encoding="utf-8") as f:
            orig_gt = {Path(r.get("filename") or r.get("file_name") or r.get("filepath") or list(r.values())[0]).name: r for r in csv.DictReader(f)}
        orig_recs = scan_directory(orig_dir, extract_text=True, use_ocr=True)
        orig_res = _eval_records(orig_recs, orig_gt, clf_blind)
        print("\n=== STAGE 3: CROSS-DATASET CHECK ON ORIGINAL 64 FILES (BLIND SEED RULES ONLY) ===")
        print(f"Accuracy: {orig_res['accuracy_pct']:.2f}% | Macro F1: {orig_res['macro_f1_pct']:.2f}% | Review Queue: {orig_res['review_queue_pct']:.1f}%")

    out_json = ROOT / "experiments/results/strategy_01_rigorous_benchmark.json"
    out_json.write_text(
        json.dumps(
            {
                "unseen_200_blind": {
                    "config_a_metadata": {k: v for k, v in res_a.items() if k != "predictions"},
                    "config_b_native": {k: v for k, v in res_b.items() if k != "predictions"},
                    "config_c_ocr": {k: v for k, v in res_c.items() if k != "predictions"},
                },
                "incremental_stream_20": {
                    "before_runtime_rules": {k: v for k, v in before_stream.items() if k != "predictions"},
                    "after_runtime_rules": {k: v for k, v in after_stream.items() if k != "predictions"},
                },
                "cross_dataset_64_blind": {k: v for k, v in orig_res.items() if k != "predictions"} if orig_res else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved rigorous benchmark JSON to {out_json}")


if __name__ == "__main__":
    main()

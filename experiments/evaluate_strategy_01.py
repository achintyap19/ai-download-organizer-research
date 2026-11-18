import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.scanner import scan_directory


def load_rule_classifier(content_aware: bool):
    strategy_file = ROOT / "strategies/01_rule_based/classifier.py"
    spec = importlib.util.spec_from_file_location("rule_strategy", strategy_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RuleBasedClassifier(content_aware=content_aware)


def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    gt = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            cat = row["true_category"].strip()
            if fname and cat:
                gt[fname] = {
                    "true_category": cat,
                    "is_exact_duplicate": int(row["is_exact_duplicate"]),
                }
    return gt


def evaluate_config(label: str, records, gt: dict[str, dict], content_aware: bool) -> tuple[dict, list[dict]]:
    classifier = load_rule_classifier(content_aware=content_aware)

    tracemalloc.start()
    t0 = time.perf_counter()
    results = classifier.classify_batch(records)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_true, y_pred = [], []
    dup_true, dup_pred = [], []
    needs_review_count = 0
    conflict_count = 0
    misclassified = []

    for rec, res in zip(records, results):
        if rec.name not in gt:
            continue
        true_cat = gt[rec.name]["true_category"]
        y_true.append(true_cat)
        y_pred.append(res.category)
        dup_true.append(gt[rec.name]["is_exact_duplicate"])
        dup_pred.append(int(rec.is_duplicate))

        if res.needs_review:
            needs_review_count += 1
        if res.conflict_with:
            conflict_count += 1
        if res.category != true_cat:
            misclassified.append({
                "filename": rec.name,
                "true": true_cat,
                "pred": res.category,
                "confidence": res.confidence,
                "extraction_status": rec.extraction_status,
                "matched_by": res.matched_by,
            })

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    dup_p = precision_score(dup_true, dup_pred, zero_division=0)
    dup_r = recall_score(dup_true, dup_pred, zero_division=0)

    print(f"\n==================== {label} ====================")
    print(f"Evaluated Files         : {len(y_true)}")
    print(f"Accuracy                : {acc * 100:.2f}%")
    print(f"Macro Precision         : {macro_p * 100:.2f}%")
    print(f"Macro Recall            : {macro_r * 100:.2f}%")
    print(f"Macro F1-Score          : {macro_f1 * 100:.2f}%")
    print(f"Duplicate Precision/Rec : {dup_p * 100:.2f}% / {dup_r * 100:.2f}%")
    print(f"Classify Latency        : {elapsed_ms:.2f} ms ({elapsed_ms / max(1, len(y_true)):.3f} ms/file)")
    print(f"Peak Python RAM         : {peak_mem / 1024 / 1024:.2f} MB")
    print(f"Flagged for Review      : {needs_review_count}/{len(y_true)} ({needs_review_count / max(1, len(y_true)) * 100:.1f}%)")
    print(f"Tie Conflicts Detected  : {conflict_count}")
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    metrics = {
        "configuration": label,
        "evaluated_files": len(y_true),
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(macro_p * 100, 2),
        "macro_recall_pct": round(macro_r * 100, 2),
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "duplicate_precision_pct": round(dup_p * 100, 2),
        "duplicate_recall_pct": round(dup_r * 100, 2),
        "classify_time_ms": round(elapsed_ms, 3),
        "peak_ram_mb": round(peak_mem / 1024 / 1024, 3),
        "manual_review_rate_pct": round(needs_review_count / max(1, len(y_true)) * 100, 2),
        "tie_conflicts": conflict_count,
    }
    return metrics, misclassified


def main():
    raw_dir = ROOT / "datasets/sandbox_raw"
    gt_path = ROOT / "datasets/ground_truth.csv"
    out_json = ROOT / "experiments/results/strategy_01_benchmark.json"

    gt = load_ground_truth(gt_path)
    if not gt:
        print("Error: datasets/ground_truth.csv has no labeled rows.")
        return

    # Pass 1: Scan WITHOUT OCR (Native parsing + SHA-256)
    t_no_ocr0 = time.perf_counter()
    records_no_ocr = scan_directory(raw_dir, extract_text=True, use_ocr=False)
    scan_no_ocr_ms = (time.perf_counter() - t_no_ocr0) * 1000

    # Pass 2: Scan WITH Conditional RapidOCR
    t_ocr0 = time.perf_counter()
    records_with_ocr = scan_directory(raw_dir, extract_text=True, use_ocr=True)
    scan_with_ocr_ms = (time.perf_counter() - t_ocr0) * 1000

    ocr_triggered = sum(1 for r in records_with_ocr if r.extraction_status == "OCR_USED")
    print(f"\n[SCAN TELEMETRY] Files: {len(records_with_ocr)} | Native Scan: {scan_no_ocr_ms:.1f} ms | Scan + Conditional OCR: {scan_with_ocr_ms:.1f} ms (OCR triggered on {ocr_triggered} files)")

    m_dropit, _ = evaluate_config("Config A: DropIt-Style (Metadata + Filename Regex Only)", records_no_ocr, gt, content_aware=False)
    m_native, _ = evaluate_config("Config B: Strategy 01 (Native Parsing + Rules, No OCR)", records_no_ocr, gt, content_aware=True)
    m_ocr, errors_ocr = evaluate_config("Config C: Strategy 01 (Native Parsing + Conditional RapidOCR + Rules)", records_with_ocr, gt, content_aware=True)

    print("\n=== MISCLASSIFIED FILES IN CONFIG C (MOTIVATING STRATEGIES 02-05) ===")
    for err in errors_ocr:
        print(f"- {err['filename']:<35} | True: {err['true']:<20} | Pred: {err['pred']:<20} | Status: {err['extraction_status']}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "scan_telemetry": {
                "total_files": len(records_with_ocr),
                "native_scan_ms": round(scan_no_ocr_ms, 2),
                "scan_with_conditional_ocr_ms": round(scan_with_ocr_ms, 2),
                "ocr_triggered_files": ocr_triggered,
            },
            "results": [m_dropit, m_native, m_ocr],
            "config_c_errors": errors_ocr,
        }, f, indent=2)

    print(f"\nSaved full benchmark results to {out_json}")


if __name__ == "__main__":
    main()
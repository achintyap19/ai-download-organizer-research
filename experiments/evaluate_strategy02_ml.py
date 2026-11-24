import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from shared.models import FileRecord
from shared.scanner import scan_directory

sys.path.insert(0, str(Path("strategies/02_ml_classification").resolve()))
_spec = importlib.util.spec_from_file_location("ml_classifier", Path("strategies/02_ml_classification/ml_classifier.py"))
_ml_mod = importlib.util.module_from_spec(_spec)
sys.modules["ml_classifier"] = _ml_mod
_spec.loader.exec_module(_ml_mod)
TraditionalMLClassifier = _ml_mod.TraditionalMLClassifier

SANDBOX_DIR = Path("datasets/sandbox_raw")
SANDBOX_CSV = Path("datasets/ground_truth.csv")
UNSEEN_BASE_DIR = Path("datasets/unseen_base_200")
UNSEEN_BASE_CSV = Path("datasets/unseen_base_ground_truth.csv")
REAL_TRAIN_DIR = Path("datasets/ml_train_real")
REAL_TRAIN_CSV = Path("datasets/ml_train_real_ground_truth.csv")

WILD_DIR = Path("datasets/real_wild_downloads")
WILD_CSV = Path("datasets/real_wild_ground_truth.csv")

MODEL_OUT = Path("strategies/02_ml_classification/artifacts/best_ml_classifier.joblib")
RESULTS_OUT = Path("experiments/results/strategy02_ml_benchmark_results.json")

LABEL_NORMALIZATION = {
    "Media_Audio_Video": "Media_Photos_Graphics",
    "Media_Screenshots": "Media_Photos_Graphics",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_metrics(y_true: list[str], y_pred: list[str], needs_review: list[bool]) -> dict:
    labels = sorted(set(y_true))
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    review_count = sum(1 for r in needs_review if r)
    review_rate = review_count / n if n else 0.0

    auto_total = sum(1 for r in needs_review if not r)
    auto_correct = sum(1 for t, p, r in zip(y_true, y_pred, needs_review) if (not r) and (t == p))
    unflagged_wrong = sum(1 for t, p, r in zip(y_true, y_pred, needs_review) if (not r) and (t != p))
    flagged_wrong = sum(1 for t, p, r in zip(y_true, y_pred, needs_review) if r and (t != p))
    auto_move_acc = (auto_correct / auto_total) if auto_total > 0 else 0.0

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
        "n": n,
        "correct": correct,
        "accuracy_pct": round(accuracy * 100, 2),
        "auto_moved_count": auto_total,
        "auto_moved_correct": auto_correct,
        "auto_move_accuracy_pct": round(auto_move_acc * 100, 2),
        "review_queue_count": review_count,
        "review_queue_pct": round(review_rate * 100, 2),
        "unflagged_wrong_moves": unflagged_wrong,
        "unflagged_wrong_move_pct": round((unflagged_wrong / n) * 100, 2) if n else 0.0,
        "flagged_wrong_moves": flagged_wrong,
        "macro_precision_pct": round((sum(precisions) / len(labels)) * 100, 2),
        "macro_recall_pct": round((sum(recalls) / len(labels)) * 100, 2),
        "macro_f1_pct": round((sum(f1s) / len(labels)) * 100, 2),
        "per_class": per_class,
    }


def load_corpus_sources(
    sources: list[tuple[Path, Path]],
    forbidden_hashes: set[str],
    extract_text: bool = True,
    use_ocr: bool = True,
) -> tuple[list[FileRecord], list[str], set[str]]:
    seen_hashes: set[str] = set()
    records: list[FileRecord] = []
    labels: list[str] = []

    for folder, csv_path in sources:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            row_map = {
                r["filename"]: LABEL_NORMALIZATION.get(r["true_category"], r["true_category"])
                for r in csv.DictReader(f)
            }
        for rec in scan_directory(folder, extract_text=extract_text, use_ocr=use_ocr):
            if rec.name not in row_map:
                continue
            digest = rec.sha256 or sha256_file(rec.path)
            if digest in forbidden_hashes or digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            records.append(rec)
            labels.append(row_map[rec.name])

    return records, labels, seen_hashes


def evaluate_5fold_cv(records: list[FileRecord], labels: list[str]) -> dict:
    counts = Counter(labels)
    valid_idx = [i for i, lbl in enumerate(labels) if counts[lbl] >= 5]
    cv_records = [records[i] for i in valid_idx]
    cv_labels = [labels[i] for i in valid_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_arr = np.array(cv_labels)
    cv_summary = {}

    for model_type in TraditionalMLClassifier.SUPPORTED_MODELS:
        fold_accs, fold_f1s, train_ms_list, infer_ms_list = [], [], [], []
        for train_idx, val_idx in skf.split(cv_records, y_arr):
            X_tr = [cv_records[i] for i in train_idx]
            y_tr = [cv_labels[i] for i in train_idx]
            X_va = [cv_records[i] for i in val_idx]
            y_va = [cv_labels[i] for i in val_idx]

            clf = TraditionalMLClassifier(model_type=model_type)
            t0 = time.perf_counter()
            clf.fit(X_tr, y_tr)
            train_ms_list.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            preds = [clf.classify(r) for r in X_va]
            infer_ms_list.append((time.perf_counter() - t1) * 1000.0)

            m = compute_metrics(y_va, [p.category for p in preds], [p.needs_review for p in preds])
            fold_accs.append(m["accuracy_pct"])
            fold_f1s.append(m["macro_f1_pct"])

        cv_summary[model_type] = {
            "cv_mean_accuracy_pct": round(float(np.mean(fold_accs)), 2),
            "cv_std_accuracy_pct": round(float(np.std(fold_accs)), 2),
            "cv_mean_macro_f1_pct": round(float(np.mean(fold_f1s)), 2),
            "cv_std_macro_f1_pct": round(float(np.std(fold_f1s)), 2),
            "mean_fit_time_ms": round(float(np.mean(train_ms_list)), 1),
            "mean_infer_time_ms": round(float(np.mean(infer_ms_list)), 2),
        }
    return cv_summary


def eval_on_wild_blind(
    clf: TraditionalMLClassifier,
    wild_rows: list[dict],
    by_hash: dict[str, FileRecord],
) -> tuple[dict, list[dict]]:
    y_true, y_pred, reviews, errs = [], [], [], []
    t0 = time.perf_counter()
    for row in wild_rows:
        if row["split"] != "blind_test":
            continue
        rec = by_hash[row["sha256"]]
        res = clf.classify(rec)
        y_true.append(row["true_category"])
        y_pred.append(res.category)
        reviews.append(res.needs_review)
        if res.category != row["true_category"]:
            errs.append({
                "filename": row["filename"],
                "source_domain": row["source_domain"],
                "true_category": row["true_category"],
                "predicted": res.category,
                "confidence": res.confidence,
                "conflict_with": res.conflict_with,
                "needs_review": res.needs_review,
                "extraction_status": rec.extraction_status,
            })
    infer_ms = (time.perf_counter() - t0) * 1000.0
    m = compute_metrics(y_true, y_pred, reviews)
    m["infer_time_ms"] = round(infer_ms, 2)
    return m, errs


def main() -> None:
    with WILD_CSV.open("r", encoding="utf-8") as f:
        wild_rows = list(csv.DictReader(f))
    locked_wild_hashes = {r["sha256"] for r in wild_rows}

    print("=== SCANNING LOCKED REAL-WORLD BLIND DATASET (n = 96 UNIQUE FILES) ===")
    wild_by_hash_cfg: dict[str, dict[str, FileRecord]] = {}
    scan_times_ms: dict[str, float] = {}
    for cfg_name, ext_text, ocr_flag in [
        ("Config A: Metadata-Only", False, False),
        ("Config B: + Native Parsing", True, False),
        ("Config C: + Conditional RapidOCR", True, True),
    ]:
        t0 = time.perf_counter()
        recs = scan_directory(WILD_DIR, extract_text=ext_text, use_ocr=ocr_flag)
        scan_times_ms[cfg_name] = round((time.perf_counter() - t0) * 1000.0, 1)
        by_h = {}
        for rec in recs:
            d = rec.sha256 or sha256_file(rec.path)
            if d not in by_h:
                by_h[d] = rec
        wild_by_hash_cfg[cfg_name] = by_h

    # --- STRATEGY 02A: Synthetic/Small Corpus Baseline ---
    print("\n=== STRATEGY 02A: BASELINE TRAINING ON SANDBOX + UNSEEN_BASE_200 ===")
    recs_02a, labels_02a, hashes_02a = load_corpus_sources(
        [(SANDBOX_DIR, SANDBOX_CSV), (UNSEEN_BASE_DIR, UNSEEN_BASE_CSV)],
        forbidden_hashes=locked_wild_hashes,
    )
    assert len(hashes_02a & locked_wild_hashes) == 0
    clf_02a = TraditionalMLClassifier(model_type="logistic_regression").fit(recs_02a, labels_02a)
    m_02a_cfg_c, _ = eval_on_wild_blind(clf_02a, wild_rows, wild_by_hash_cfg["Config C: + Conditional RapidOCR"])
    print(
        f"  Strategy 02A (n_train={len(recs_02a)}) on Blind Test (Config C): "
        f"Overall = {m_02a_cfg_c['accuracy_pct']:6.2f}% ({m_02a_cfg_c['correct']}/{m_02a_cfg_c['n']}) | "
        f"Auto-Move = {m_02a_cfg_c['auto_move_accuracy_pct']:6.2f}% | "
        f"Review = {m_02a_cfg_c['review_queue_pct']:5.2f}% | "
        f"Unflagged Wrong = {m_02a_cfg_c['unflagged_wrong_moves']}"
    )

    # --- STRATEGY 02B: Structural Features + Real-World Disjoint Corpus ---
    print("\n=== STRATEGY 02B: REAL-WORLD DISJOINT TRAINING CORPUS (SANDBOX + ML_TRAIN_REAL) ===")
    recs_02b, labels_02b, hashes_02b = load_corpus_sources(
        [(SANDBOX_DIR, SANDBOX_CSV), (REAL_TRAIN_DIR, REAL_TRAIN_CSV)],
        forbidden_hashes=locked_wild_hashes,
    )
    assert len(hashes_02b & locked_wild_hashes) == 0, "LEAKAGE DETECTED!"
    print(f"Loaded {len(recs_02b)} SHA-256-disjoint real-world training files across {len(set(labels_02b))} categories.")
    print("Training class distribution:", dict(sorted(Counter(labels_02b).items())))

    print("\n--- 5-Fold Stratified CV Model Selection (Strategy 02B) ---")
    cv_results = evaluate_5fold_cv(recs_02b, labels_02b)
    for model_type, stats in cv_results.items():
        print(
            f"  {model_type:22s} | CV Acc: {stats['cv_mean_accuracy_pct']:6.2f}% (±{stats['cv_std_accuracy_pct']:4.2f}%) | "
            f"Macro F1: {stats['cv_mean_macro_f1_pct']:6.2f}% | Fit: {stats['mean_fit_time_ms']:6.1f} ms"
        )

    best_model_type = max(
        cv_results.keys(),
        key=lambda k: (cv_results[k]["cv_mean_macro_f1_pct"], cv_results[k]["cv_mean_accuracy_pct"]),
    )
    print(f"\nSelected Best Model via 5-Fold CV: {best_model_type}")

    best_clf = TraditionalMLClassifier(model_type=best_model_type)
    t_fit = time.perf_counter()
    best_clf.fit(recs_02b, labels_02b)
    full_fit_ms = (time.perf_counter() - t_fit) * 1000.0
    best_clf.save(MODEL_OUT)
    model_size_kb = round(MODEL_OUT.stat().st_size / 1024.0, 1)

    print("\n--- Locked Real-World Blind Ablation & Safety Metrics (Strategy 02B, n = 85) ---")
    ablation_reports = []
    errors_c: list[dict] = []
    for cfg_name in [
        "Config A: Metadata-Only",
        "Config B: + Native Parsing",
        "Config C: + Conditional RapidOCR",
    ]:
        m, errs = eval_on_wild_blind(best_clf, wild_rows, wild_by_hash_cfg[cfg_name])
        m["config"] = cfg_name
        m["scan_time_ms"] = scan_times_ms[cfg_name]
        ablation_reports.append(m)
        if "Config C" in cfg_name:
            errors_c = errs
        print(
            f"  {cfg_name:34s} | Overall: {m['accuracy_pct']:6.2f}% ({m['correct']}/{m['n']}) | "
            f"Auto-Move: {m['auto_move_accuracy_pct']:6.2f}% ({m['auto_moved_correct']}/{m['auto_moved_count']}) | "
            f"Review: {m['review_queue_pct']:5.2f}% | Unflagged Wrong: {m['unflagged_wrong_moves']} | "
            f"Macro F1: {m['macro_f1_pct']:6.2f}%"
        )

    cfg_c_report = ablation_reports[-1]
    print("\n--- STRATEGY 02B CONFIG C PER-CATEGORY BREAKDOWN ---")
    for cat, m in cfg_c_report["per_class"].items():
        print(f"  {cat:22s} (n={m['support']:2d}) | P: {m['precision']:6.2f}% | R: {m['recall']:6.2f}% | F1: {m['f1']:6.2f}%")

    print(f"\n--- STRATEGY 02B CONFIG C MISCLASSIFICATIONS ({len(errors_c)} files) ---")
    for err in errors_c:
        print(
            f"  {err['filename']} [{err['source_domain']}]: True={err['true_category']} -> "
            f"Pred={err['predicted']} (conf={err['confidence']}, review={err['needs_review']})"
        )

    print("\n=== STAGE 2 UNSEEN-DOMAIN RETRAINING (k=2 Support -> m=7 Unseen Queries) ===")
    by_hash_c = wild_by_hash_cfg["Config C: + Conditional RapidOCR"]
    support_recs, support_labels, query_rows = [], [], []
    for row in wild_rows:
        if row["split"] == "stage2_support":
            support_recs.append(by_hash_c[row["sha256"]])
            support_labels.append(row["true_category"])
        elif row["split"] == "stage2_query":
            query_rows.append(row)

    y_q_true = [r["true_category"] for r in query_rows]
    preds_before = [best_clf.classify(by_hash_c[r["sha256"]]) for r in query_rows]
    stage2_before = compute_metrics(
        y_q_true,
        [p.category for p in preds_before],
        [p.needs_review for p in preds_before],
    )

    retrained_clf = TraditionalMLClassifier(model_type=best_model_type)
    retrained_clf.fit(recs_02b + support_recs, labels_02b + support_labels)
    preds_after = [retrained_clf.classify(by_hash_c[r["sha256"]]) for r in query_rows]
    stage2_after = compute_metrics(
        y_q_true,
        [p.category for p in preds_after],
        [p.needs_review for p in preds_after],
    )
    print(
        f"  Before Retraining (0 unseen examples): Overall = {stage2_before['accuracy_pct']:6.2f}% | "
        f"Auto-Move = {stage2_before['auto_move_accuracy_pct']:6.2f}% | Review = {stage2_before['review_queue_pct']:5.2f}%"
    )
    print(
        f"  After k=2 Support Retraining:          Overall = {stage2_after['accuracy_pct']:6.2f}% | "
        f"Auto-Move = {stage2_after['auto_move_accuracy_pct']:6.2f}% | Review = {stage2_after['review_queue_pct']:5.2f}%"
    )
    for r, pb, pa in zip(query_rows, preds_before, preds_after):
        print(
            f"    {r['filename']} | True={r['true_category']} | "
            f"Before={pb.category} ({pb.confidence}) -> After={pa.category} ({pa.confidence})"
        )

    # Post-Retraining Non-Regression Check on 85-File Blind Set
    m_02b_post_blind, _ = eval_on_wild_blind(retrained_clf, wild_rows, by_hash_c)
    print("\n=== POST-RETRAINING NON-REGRESSION CHECK ON 85-FILE BLIND SET ===")
    print(
        f"  Pre-Retraining Overall: {cfg_c_report['accuracy_pct']:6.2f}% -> "
        f"Post-Retraining Overall: {m_02b_post_blind['accuracy_pct']:6.2f}% "
        f"(Delta: {m_02b_post_blind['accuracy_pct'] - cfg_c_report['accuracy_pct']:+.2f}%) | "
        f"Unflagged Wrong: {cfg_c_report['unflagged_wrong_moves']} -> {m_02b_post_blind['unflagged_wrong_moves']}"
    )

    full_report = {
        "strategy_02a_synthetic_corpus_baseline": {
            "n_train": len(recs_02a),
            "blind_test_config_c": m_02a_cfg_c,
        },
        "strategy_02b_real_disjoint_corpus": {
            "n_train": len(recs_02b),
            "sha256_overlap_with_blind_test": len(hashes_02b & locked_wild_hashes),
            "training_class_counts": dict(sorted(Counter(labels_02b).items())),
            "five_fold_cv_model_selection": cv_results,
            "selected_model": best_model_type,
            "full_fit_time_ms": round(full_fit_ms, 1),
            "serialized_model_size_kb": model_size_kb,
            "top_discriminative_features": best_clf.get_top_features_per_class(top_k=6),
            "stage1_real_wild_blind_ablation": ablation_reports,
            "stage1_config_c_misclassifications": errors_c,
            "stage2_unseen_domain_adaptation": {
                "before_retraining": stage2_before,
                "after_k2_retraining": stage2_after,
            },
            "post_retraining_85file_non_regression": m_02b_post_blind,
        },
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved serialized model to {MODEL_OUT} ({model_size_kb} KB)")
    print(f"Saved full Strategy 02A vs 02B benchmark JSON to {RESULTS_OUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import importlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from experiments.evaluate_strategy02_ml import load_corpus_sources, sha256_file
from shared.models import FileRecord

TraditionalMLClassifier = importlib.import_module(
    "strategies.02_ml_classification.ml_classifier"
).TraditionalMLClassifier

ARTIFACT_STOPWORDS = {
    "sample", "samples", "template", "templates", "example", "examples",
    "altacv", "friggeri", "awesome", "rendercv", "moderncv", "deedy",
    "jakes", "unlocked", "edited", "copy", "v1", "v2", "latest",
}


def sanitize_stem_record(rec: FileRecord) -> FileRecord:
    """Returns a copy of FileRecord with source-template artifact tokens scrubbed from stem."""
    tokens = re.split(r"[_\-\.\(\)\[\]\s]+", rec.stem.lower())
    cleaned = [t for t in tokens if t and t not in ARTIFACT_STOPWORDS]
    new_stem = " ".join(cleaned) if cleaned else "document"
    return replace(rec, name=f"{new_stem}{rec.extension}", stem=new_stem)


def fit_with_weights(
    clf: TraditionalMLClassifier,
    X: Sequence[FileRecord],
    y: Sequence[str],
    weight_overrides: Dict[str, float] | None = None,
) -> TraditionalMLClassifier:
    """Fits TraditionalMLClassifier with custom FeatureUnion branch weights applied before fitting."""
    if weight_overrides:
        orig_build = clf._build_pipeline

        def _custom_build():
            pipe = orig_build()
            pipe.named_steps["features"].transformer_weights.update(weight_overrides)
            return pipe

        clf._build_pipeline = _custom_build  # type: ignore[method-assign]
    clf.fit(X, y)
    return clf


def load_source_domain_by_sha256(csv_paths: Sequence[Path]) -> Dict[str, str]:
    """Maps sha256 -> source_domain from ground-truth CSVs."""
    mapping: Dict[str, str] = {}
    for p in csv_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                digest = (row.get("sha256") or "").strip().lower()
                src = (row.get("source_domain") or "").strip()
                if digest and src:
                    mapping[digest] = src
    return mapping


def eval_predictions(
    clf: TraditionalMLClassifier,
    records: Sequence[FileRecord],
    y_true: Sequence[str],
) -> Dict[str, float]:
    preds = [clf.classify(r) for r in records]
    y_pred = [p.category for p in preds]
    overall = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    auto_correct = 0
    auto_total = 0
    review_count = 0
    unflagged_wrong = 0

    for yt, pred in zip(y_true, preds):
        if pred.needs_review:
            review_count += 1
        else:
            auto_total += 1
            if pred.category == yt:
                auto_correct += 1
            else:
                unflagged_wrong += 1

    auto_acc = (auto_correct / auto_total * 100.0) if auto_total > 0 else 0.0
    review_pct = review_count / len(y_true) * 100.0
    return {
        "n": len(y_true),
        "overall_acc": round(overall, 2),
        "auto_move_acc": round(auto_acc, 2),
        "auto_moved": auto_total,
        "review_pct": round(review_pct, 2),
        "unflagged_wrong": unflagged_wrong,
        "macro_f1": round(macro_f1, 2),
    }


def main() -> None:
    core_cats = {
        "Academic_Study",
        "Archives_Compressed",
        "Career_Identity",
        "Finance_Invoices",
        "Media_Photos_Graphics",
        "Software_Installers",
        "Spreadsheets_Data",
    }

    raw_blind_recs, raw_blind_labels, _ = load_corpus_sources(
        [(Path("datasets/real_wild_downloads"), Path("datasets/real_wild_ground_truth.csv"))],
        forbidden_hashes=set(),
        extract_text=True,
        use_ocr=True,
    )
    blind_pairs = [(r, l) for r, l in zip(raw_blind_recs, raw_blind_labels) if l in core_cats]
    blind_recs = [x[0] for x in blind_pairs]
    blind_labels = [x[1] for x in blind_pairs]
    blind_hashes = {r.sha256 for r in blind_recs}

    raw_train_recs, raw_train_labels, _ = load_corpus_sources(
        [
            (Path("datasets/sandbox_raw"), Path("datasets/ground_truth.csv")),
            (Path("datasets/ml_train_real"), Path("datasets/ml_train_real_ground_truth.csv")),
        ],
        forbidden_hashes=blind_hashes,
        extract_text=True,
        use_ocr=True,
    )
    train_pairs = [(r, l) for r, l in zip(raw_train_recs, raw_train_labels) if l in core_cats]
    train_recs = [x[0] for x in train_pairs]
    train_labels = [x[1] for x in train_pairs]

    print(f"Loaded Disjoint Train (n={len(train_recs)}) & Locked Blind (n={len(blind_recs)})")

    # =========================================================================
    # CHECK 1: FILENAME ARTIFACT ABLATION (FILENAME VS. BODY TEXT DEPENDENCE)
    # =========================================================================
    print("\n=== CHECK 1: FILENAME ARTIFACT & BODY-ONLY ABLATION (n = 85 BLIND) ===")

    clf_orig = fit_with_weights(TraditionalMLClassifier(model_type="calibrated_linearsvc"), train_recs, train_labels)
    m_orig = eval_predictions(clf_orig, blind_recs, blind_labels)

    train_clean = [sanitize_stem_record(r) for r in train_recs]
    blind_clean = [sanitize_stem_record(r) for r in blind_recs]

    clf_clean = fit_with_weights(TraditionalMLClassifier(model_type="calibrated_linearsvc"), train_clean, train_labels)
    m_clean = eval_predictions(clf_clean, blind_clean, blind_labels)

    clf_low_stem = fit_with_weights(
        TraditionalMLClassifier(model_type="calibrated_linearsvc"),
        train_clean,
        train_labels,
        {"stem_word_tfidf": 0.5, "stem_char_tfidf": 0.4, "body_word_tfidf": 1.8, "body_lsa_svd": 1.2},
    )
    m_low_stem = eval_predictions(clf_low_stem, blind_clean, blind_labels)

    clf_no_stem = fit_with_weights(
        TraditionalMLClassifier(model_type="calibrated_linearsvc"),
        train_clean,
        train_labels,
        {"stem_word_tfidf": 0.0, "stem_char_tfidf": 0.0, "body_word_tfidf": 1.8, "body_lsa_svd": 1.2},
    )
    m_no_stem = eval_predictions(clf_no_stem, blind_clean, blind_labels)

    clf_no_body = fit_with_weights(
        TraditionalMLClassifier(model_type="calibrated_linearsvc"),
        train_clean,
        train_labels,
        {"body_word_tfidf": 0.0, "body_lsa_svd": 0.0},
    )
    m_no_body = eval_predictions(clf_no_body, blind_clean, blind_labels)

    for label, res in [
        ("1A. Raw Stems (Current Baseline)", m_orig),
        ("1B. Artifact-Scrubbed Stems (no 'sample'/'altacv')", m_clean),
        ("1C. Low Filename Weight (0.5/0.4) + Scrubbed Stems", m_low_stem),
        ("1D. ZERO Filename Features (Ext + Body + Struct)", m_no_stem),
        ("1E. ZERO Body Features (Ext + Scrubbed Stem Only)", m_no_body),
    ]:
        print(
            f"  {label:<50} | Overall: {res['overall_acc']:6.2f}% | "
            f"Auto-Move: {res['auto_move_acc']:6.2f}% | Review: {res['review_pct']:5.2f}% | "
            f"Unflagged Wrong: {res['unflagged_wrong']} | Macro F1: {res['macro_f1']:6.2f}%"
        )

    print("\n  Utility bill 'sample' check under 1B (Artifact-Scrubbed Stems):")
    for r_raw, r_cln, yt in zip(blind_recs, blind_clean, blind_labels):
        if "sample" in r_raw.name.lower() and yt == "Finance_Invoices":
            p_before = clf_orig.classify(r_raw)
            p_after = clf_clean.classify(r_cln)
            print(
                f"    {r_raw.name}: Before={p_before.category} ({p_before.confidence}, rev={p_before.needs_review}) "
                f"-> After={p_after.category} ({p_after.confidence}, rev={p_after.needs_review})"
            )

    # =========================================================================
    # CHECK 2: LEAVE-ONE-SOURCE-OUT (LOSO) CROSS-DOMAIN GENERALIZATION
    # =========================================================================
    print("\n=== CHECK 2: LEAVE-ONE-SOURCE-OUT (LOSO) GENERALIZATION ACROSS POOLED CORPUS ===")
    src_map = load_source_domain_by_sha256([
        Path("datasets/ml_train_real_ground_truth.csv"),
        Path("datasets/real_wild_ground_truth.csv"),
        Path("datasets/ground_truth.csv"),
    ])

    pooled: List[Tuple[FileRecord, str, str]] = []
    seen: set[str] = set()
    for rec, cat in list(zip(train_recs, train_labels)) + list(zip(blind_recs, blind_labels)):
        digest = (rec.sha256 or sha256_file(rec.path)).lower()
        if digest in seen:
            continue
        seen.add(digest)
        src = src_map.get(digest, "sandbox_synthetic")
        pooled.append((sanitize_stem_record(rec), cat, src))

    by_source: Dict[str, List[Tuple[FileRecord, str, str]]] = defaultdict(list)
    for item in pooled:
        by_source[item[2]].append(item)

    loso_results = {}
    for holdout_src, test_bucket in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        if len(test_bucket) < 5:
            continue
        train_bucket = [x for x in pooled if x[2] != holdout_src]
        tr_x = [x[0] for x in train_bucket]
        tr_y = [x[1] for x in train_bucket]
        te_x = [x[0] for x in test_bucket]
        te_y = [x[1] for x in test_bucket]

        clf_loso = fit_with_weights(TraditionalMLClassifier(model_type="calibrated_linearsvc"), tr_x, tr_y)
        res = eval_predictions(clf_loso, te_x, te_y)
        loso_results[holdout_src] = res
        cats_present = dict(Counter(te_y))
        print(
            f"  Holdout Source: {holdout_src:<22} (n={res['n']:2d}) | Overall: {res['overall_acc']:6.2f}% | "
            f"Auto-Move: {res['auto_move_acc']:6.2f}% | Review: {res['review_pct']:5.2f}% | "
            f"Unflagged Wrong: {res['unflagged_wrong']} | Classes: {cats_present}"
        )

    # =========================================================================
    # CHECK 3: STRICT TRAINING-ONLY 5-FOLD CV SELECTION OF WEIGHTS & CUTOFFS
    # =========================================================================
    print("\n=== CHECK 3: STRICT TRAINING-ONLY 5-FOLD CV SELECTION (ZERO BLIND PEEKING) ===")
    weight_presets = {
        "balanced_multimodal": {
            "ext_ohe": 2.0, "stem_word_tfidf": 1.0, "stem_char_tfidf": 0.8,
            "body_word_tfidf": 1.5, "body_lsa_svd": 1.0, "structural_stats": 1.5,
        },
        "content_heavy": {
            "ext_ohe": 2.0, "stem_word_tfidf": 0.6, "stem_char_tfidf": 0.5,
            "body_word_tfidf": 1.8, "body_lsa_svd": 1.2, "structural_stats": 1.6,
        },
        "uniform_unweighted": {
            "ext_ohe": 1.0, "stem_word_tfidf": 1.0, "stem_char_tfidf": 1.0,
            "body_word_tfidf": 1.0, "body_lsa_svd": 1.0, "structural_stats": 1.0,
        },
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_preset_name = None
    best_cv_f1 = -1.0
    best_oof_probs = None
    classes_sorted = sorted(core_cats)

    for pname, weights in weight_presets.items():
        oof_probs = np.zeros((len(train_clean), len(classes_sorted)), dtype=np.float64)
        fold_accs = []
        fold_f1s = []
        for tr_idx, va_idx in skf.split(train_clean, train_labels):
            tr_r = [train_clean[i] for i in tr_idx]
            tr_l = [train_labels[i] for i in tr_idx]
            va_r = [train_clean[i] for i in va_idx]
            va_l = [train_labels[i] for i in va_idx]

            c = fit_with_weights(
                TraditionalMLClassifier(model_type="calibrated_linearsvc"),
                tr_r,
                tr_l,
                weights,
            )

            probs = c.predict_proba(va_r)
            col_map = [c.classes_.index(cls) for cls in classes_sorted]
            aligned = probs[:, col_map]
            oof_probs[va_idx] = aligned
            preds = [classes_sorted[int(np.argmax(row))] for row in aligned]
            fold_accs.append(accuracy_score(va_l, preds))
            fold_f1s.append(f1_score(va_l, preds, average="macro", zero_division=0))

        mean_acc = float(np.mean(fold_accs)) * 100.0
        mean_f1 = float(np.mean(fold_f1s)) * 100.0
        print(f"  CV Preset '{pname:<20}' | 5-Fold Train Acc: {mean_acc:6.2f}% | Macro F1: {mean_f1:6.2f}%")
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_preset_name = pname
            best_oof_probs = oof_probs

    best_thresh = 0.50
    best_margin = 0.10
    best_review_pct = 101.0
    best_train_auto_acc = 0.0

    for th in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for mg in [0.08, 0.12, 0.15, 0.20]:
            auto_c, auto_t, rev_c = 0, 0, 0
            for yt, row in zip(train_labels, best_oof_probs):
                order = np.argsort(row)[::-1]
                p1, p2 = float(row[order[0]]), float(row[order[1]])
                pred_cls = classes_sorted[int(order[0])]
                if p1 < th or (p1 - p2) <= mg:
                    rev_c += 1
                else:
                    auto_t += 1
                    if pred_cls == yt:
                        auto_c += 1
            auto_acc = (auto_c / auto_t * 100.0) if auto_t > 0 else 0.0
            rev_pct = rev_c / len(train_labels) * 100.0
            if auto_acc >= 97.5 and rev_pct < best_review_pct:
                best_review_pct = rev_pct
                best_thresh = th
                best_margin = mg
                best_train_auto_acc = auto_acc

    print(
        f"\n  Selected Strictly via Train OOF CV: preset='{best_preset_name}', "
        f"conf_thresh={best_thresh:.2f}, margin={best_margin:.2f} "
        f"(Train OOF Auto-Move={best_train_auto_acc:.2f}%, Review={best_review_pct:.2f}%)"
    )

    final_clf = fit_with_weights(
        TraditionalMLClassifier(
            model_type="calibrated_linearsvc",
            confidence_threshold=best_thresh,
            ambiguity_margin=best_margin,
        ),
        train_clean,
        train_labels,
        weight_presets[best_preset_name],
    )
    final_blind = eval_predictions(final_clf, blind_clean, blind_labels)

    print(
        f"  SINGLE LOCKED BLIND RUN (n=85) WITH CV-SELECTED PARAMS: "
        f"Overall: {final_blind['overall_acc']:.2f}% | Auto-Move: {final_blind['auto_move_acc']:.2f}% | "
        f"Review: {final_blind['review_pct']:.2f}% | Unflagged Wrong: {final_blind['unflagged_wrong']} | "
        f"Macro F1: {final_blind['macro_f1']:.2f}%"
    )

    out_path = Path("experiments/results/strategy02_rigor_audit.json")
    out_path.write_text(
        json.dumps(
            {
                "filename_ablation": {
                    "raw_stems": m_orig,
                    "artifact_scrubbed_stems": m_clean,
                    "low_filename_weight": m_low_stem,
                    "zero_filename_features": m_no_stem,
                    "zero_body_features": m_no_body,
                },
                "leave_one_source_out": loso_results,
                "cv_selected_params": {
                    "preset": best_preset_name,
                    "weights": weight_presets[best_preset_name],
                    "confidence_threshold": best_thresh,
                    "ambiguity_margin": best_margin,
                    "train_oof_auto_move_acc": round(best_train_auto_acc, 2),
                    "train_oof_review_pct": round(best_review_pct, 2),
                },
                "single_locked_blind_result": final_blind,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved full audit report to {out_path}")


if __name__ == "__main__":
    main()


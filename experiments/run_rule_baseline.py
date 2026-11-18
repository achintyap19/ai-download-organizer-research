import argparse
import importlib.util
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.executor import TransactionalExecutor
from shared.planner import build_action_plan
from shared.scanner import scan_directory


def load_rule_classifier(content_aware: bool = True):
    strategy_file = ROOT / "strategies/01_rule_based/classifier.py"
    spec = importlib.util.spec_from_file_location("rule_strategy", strategy_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RuleBasedClassifier(content_aware=content_aware)


def main():
    parser = argparse.ArgumentParser(description="Run Approach 01: Rule-Based Baseline")
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/sandbox_raw")
    parser.add_argument("--dest", type=Path, default=ROOT / "datasets/sandbox_organized")
    parser.add_argument("--db", type=Path, default=ROOT / "datasets/journal.db")
    parser.add_argument("--apply", action="store_true", help="Execute moves (default is dry-run)")
    parser.add_argument("--rollback", type=str, help="Rollback a specific batch_id")
    args = parser.parse_args()

    executor = TransactionalExecutor(args.db)
    if args.rollback:
        print(f"[ROLLBACK] Restored {executor.rollback_batch(args.rollback)} files.")
        return

    tracemalloc.start()
    t0 = time.perf_counter()

    records = scan_directory(args.source, extract_text=True)
    classifier = load_rule_classifier(content_aware=True)
    results = classifier.classify_batch(records)
    plan = build_action_plan(records, results, args.dest, quarantine_duplicates=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    dup_count = sum(1 for r in records if r.is_duplicate)
    print(f"\n=== {classifier.name} DRY-RUN ({len(plan)} files | {dup_count} exact duplicates | {elapsed_ms:.2f} ms | Peak RAM: {peak_mem / 1024 / 1024:.2f} MB) ===")
    for item in plan:
        print(f"{item.source_path.name:<35} -> {item.category:<22} ({item.confidence:.2f}) | {item.matched_by}")

    if args.apply:
        batch_id = executor.apply_plan(plan, dry_run=False)
        print(f"\n[APPLIED] Batch ID: {batch_id} (Undo with: python experiments/run_rule_baseline.py --rollback {batch_id})")


if __name__ == "__main__":
    main()

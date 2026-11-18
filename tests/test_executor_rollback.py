from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.executor import TransactionalExecutor
from shared.models import ActionItem


def test_rollback_protects_post_move_edited_and_renamed_files(tmp_path: Path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    db_path = tmp_path / "journal.db"

    f_clean = downloads / "lecture_notes.pdf"
    f_edited = downloads / "assignment_draft.docx"
    f_renamed = downloads / "invoice_jan.pdf"

    f_clean.write_text("clean content", encoding="utf-8")
    f_edited.write_text("version 1", encoding="utf-8")
    f_renamed.write_text("invoice data", encoding="utf-8")

    plan = [
        ActionItem(f_clean, downloads / "Academic_Study" / "lecture_notes.pdf", "Academic_Study", 0.9, [], "rule"),
        ActionItem(f_edited, downloads / "Academic_Study" / "assignment_draft.docx", "Academic_Study", 0.9, [], "rule"),
        ActionItem(f_renamed, downloads / "Finance_Invoices" / "invoice_jan.pdf", "Finance_Invoices", 0.9, [], "rule"),
    ]

    executor = TransactionalExecutor(db_path=db_path)
    batch_id = executor.apply_plan(plan)

    # Simulate user modifying assignment_draft.docx and renaming invoice_jan.pdf after move
    (downloads / "Academic_Study" / "assignment_draft.docx").write_text("version 2 edited by user", encoding="utf-8")
    (downloads / "Finance_Invoices" / "invoice_jan.pdf").rename(downloads / "Finance_Invoices" / "invoice_2026_final.pdf")

    summary = executor.rollback_batch(batch_id, force_modified=False)

    assert summary["restored"] == 1
    assert summary["skipped_modified"] == 1
    assert summary["skipped_missing"] == 1
    assert f_clean.exists()
    assert not f_edited.exists()  # User's edited file was safely preserved!
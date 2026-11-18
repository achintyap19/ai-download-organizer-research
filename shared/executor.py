from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sqlite3
from typing import Dict, List
from .models import ActionItem


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TransactionalExecutor:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operation_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    destination_path TEXT NOT NULL,
                    sha256_at_move TEXT NOT NULL,
                    category TEXT NOT NULL,
                    matched_by TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)

    def apply_plan(self, plan: List[ActionItem]) -> str:
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        with sqlite3.connect(self.db_path) as conn:
            for item in plan:
                if not item.source_path.exists():
                    continue
                item.destination_path.parent.mkdir(parents=True, exist_ok=True)
                file_hash = _compute_sha256(item.source_path)
                shutil.move(str(item.source_path), str(item.destination_path))
                item.status = "MOVED"
                conn.execute(
                    """
                    INSERT INTO operation_journal
                    (batch_id, timestamp, source_path, destination_path, sha256_at_move, category, matched_by, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        datetime.now(timezone.utc).isoformat(),
                        str(item.source_path),
                        str(item.destination_path),
                        file_hash,
                        item.category,
                        item.matched_by,
                        "MOVED",
                    ),
                )
        return batch_id

    def rollback_batch(self, batch_id: str, force_modified: bool = False) -> Dict[str, int]:
        """
        Verifies destination SHA-256 integrity before reverting a batch:
        - Unmodified files move back cleanly.
        - Post-move modified files are protected unless force_modified=True.
        - Renamed/missing destination files are skipped cleanly without crashing the batch.
        """
        summary = {"restored": 0, "skipped_modified": 0, "restored_modified": 0, "skipped_missing": 0}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, source_path, destination_path, sha256_at_move
                FROM operation_journal
                WHERE batch_id = ? AND status = 'MOVED'
                ORDER BY id DESC
                """,
                (batch_id,),
            ).fetchall()

            for row_id, src_str, dst_str, expected_hash in rows:
                src, dst = Path(src_str), Path(dst_str)
                if not dst.exists():
                    summary["skipped_missing"] += 1
                    conn.execute("UPDATE operation_journal SET status = ? WHERE id = ?", ("SKIPPED_MISSING_AT_DEST", row_id))
                    continue

                current_hash = _compute_sha256(dst)
                is_modified = current_hash != expected_hash

                if is_modified and not force_modified:
                    summary["skipped_modified"] += 1
                    conn.execute("UPDATE operation_journal SET status = ? WHERE id = ?", ("SKIPPED_MODIFIED_CONTENT", row_id))
                    continue

                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
                new_status = "ROLLED_BACK_MODIFIED_CONTENT" if is_modified else "ROLLED_BACK"
                conn.execute("UPDATE operation_journal SET status = ? WHERE id = ?", (new_status, row_id))
                if is_modified:
                    summary["restored_modified"] += 1
                else:
                    summary["restored"] += 1

        return summary
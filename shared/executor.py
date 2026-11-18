from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
from typing import List
import uuid
from .models import ActionItem


class TransactionalExecutor:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    destination_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    matched_by TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def apply_plan(self, plan: List[ActionItem], dry_run: bool = True) -> str:
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        if dry_run:
            return batch_id
        with sqlite3.connect(self.db_path) as conn:
            for item in plan:
                if not item.source_path.exists():
                    continue
                item.destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.source_path), str(item.destination_path))
                item.status = "APPLIED"
                conn.execute(
                    """
                    INSERT INTO operation_journal
                    (batch_id, timestamp, source_path, destination_path, category, confidence, matched_by, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (batch_id, datetime.now(timezone.utc).isoformat(), str(item.source_path),
                     str(item.destination_path), item.category, item.confidence, item.matched_by, "APPLIED"),
                )
        return batch_id

    def rollback_batch(self, batch_id: str) -> int:
        restored = 0
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, source_path, destination_path FROM operation_journal WHERE batch_id = ? AND status = 'APPLIED' ORDER BY id DESC",
                (batch_id,),
            ).fetchall()
            for row_id, src_str, dst_str in rows:
                src, dst = Path(src_str), Path(dst_str)
                if dst.exists() and not src.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(src))
                    conn.execute("UPDATE operation_journal SET status = 'ROLLED_BACK' WHERE id = ?", (row_id,))
                    restored += 1
        return restored

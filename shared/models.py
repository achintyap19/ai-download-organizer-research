from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class FileRecord:
    """Represents a scanned file, extraction status, text snippet, and duplicate state."""
    path: Path
    name: str
    stem: str
    extension: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    sha256: Optional[str] = None
    extracted_text: str = ""
    extraction_status: str = "NOT_EXTRACTED"  # NATIVE_TEXT, OCR_USED, PASSWORD_PROTECTED, CORRUPT, SKIPPED_BINARY
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None


@dataclass
class ClassificationResult:
    """Standardized output contract shared across all classification strategies."""
    file_path: Path
    category: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    suggested_name: Optional[str] = None
    matched_by: str = "unknown"
    needs_review: bool = False
    conflict_with: Optional[str] = None


@dataclass
class ActionItem:
    """Represents a planned filesystem mutation for dry-run preview and SQLite rollback."""
    source_path: Path
    destination_path: Path
    category: str
    confidence: float
    tags: List[str]
    matched_by: str
    is_duplicate: bool = False
    status: str = "PENDING"
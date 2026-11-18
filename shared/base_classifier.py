from abc import ABC, abstractmethod
from typing import List
from .models import ClassificationResult, FileRecord


class BaseClassifier(ABC):
    """Abstract interface ensuring fair, interchangeable benchmarking across strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def classify(self, record: FileRecord) -> ClassificationResult:
        pass

    def classify_batch(self, records: List[FileRecord]) -> List[ClassificationResult]:
        return [self.classify(r) for r in records]

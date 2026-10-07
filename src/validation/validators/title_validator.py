"""Soft-boosts candidates whose supporting detections were found via a
titled pattern (e.g. 'Dr. Suresh Kumar') - a strong signal of personhood."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator
from src.detection.regex_detector import PERSON_CUE_PATTERN, PERSON_MARKER_PATTERNS


class TitleValidator(BaseValidator):
    name = "title"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        patterns = {d.metadata.get("pattern") for d in candidate.candidate.source_detections}

        if "titled" in patterns:
            return ValidationResult(
                self.name, passed=True, score=0.30,
                message="Preceded by a recognized title/honorific",
            )
        # Kinship word, sign-off, salutation or speaker label (2026-10-01) -
        # the same kind of evidence as a title. See regex_detector.py.
        if patterns & PERSON_MARKER_PATTERNS:
            cues = sorted({d.metadata.get("cue", "") for d in candidate.candidate.source_detections
                           if d.metadata.get("pattern") == PERSON_CUE_PATTERN})
            return ValidationResult(
                self.name, passed=True, score=0.30,
                message=f"Marked as a person by its context ({', '.join(cues)})",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No title evidence")

"""Rejects candidates where the same token repeats (e.g. OCR/typo
artifacts like 'John John John') - not a plausible person name.

A word used twice with another between is allowed (2026-10-01): some real
names repeat one ("Rao Gopal Rao", a Telugu actor in a school textbook).
Adjacent repeats ("John John") and three uses stay rejected.

Initials are not words (2026-10-07): the same letter twice is an ordinary
pair of initials ("V. V. Subrahmanyam", "J. J. Thomson", "A. A. Milne") -
a school textbook's interviewer was rejected as a repeat. Only full words
are compared; a span of nothing but initials is InitialValidator's to reject."""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.latin import LATIN_UPPER
from src.validation.base_validator import BaseValidator

_INITIAL_RE = re.compile(rf"^[{LATIN_UPPER}]\.?$")


class RepetitionValidator(BaseValidator):
    name = "repetition"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        raw = candidate.candidate.normalized_text.split()
        tokens = [None if _INITIAL_RE.match(t) else t.lower() for t in raw]

        adjacent = any(a is not None and a == b for a, b in zip(tokens, tokens[1:]))
        if adjacent or any(t is not None and tokens.count(t) >= 3 for t in tokens):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Candidate contains repeated tokens",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No repetition")

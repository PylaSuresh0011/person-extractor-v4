"""
src.candidate.explicit_aliases
=================================
Aliases a document states (textbook error brief P6, 2026-10-01):

  "Nandamuri Taraka Ramarao (NTR)"            - an acronym of the name's
                                                initials, in brackets after it
  "Manikda [as Ray was affectionately called" - "<alias> as <name> was
                                                (affectionately) called/known"
  "Arthur Lemiere Hendriks ... (known as      - "known as" / "affectionately
   Micky Hendriks ...)"                         called" <alias>, after a name
                                                in the same sentence

The name must be one this document accepted. Each pair (alias, name) goes
into the result (ExtractionResult.aliases) for the combined lists' aliases
column (alias_resolution.py), and the alias's mentions get a
`document_name` detection vouched for by the name - an acronym alias ("NTR")
is then not rejected as an acronym (StructureValidator).
"""

from __future__ import annotations

import re

from src.core.models import CandidateResult, Decision
from src.detection.regex_detector import _NAME_SEQ

_WORDS_THEN_ACRONYM = re.compile(r"(?P<words>(?:[A-Z][a-z]+[ \t]+){0,5}[A-Z][a-z]+)[ \t]*\n?[ \t]*\((?P<acronym>[A-Z]{2,5})\)")
_AS_CALLED = re.compile(
    rf"(?P<alias>{_NAME_SEQ})[ \t]*[\[(,][ \t]*as[ \t]+(?P<name>{_NAME_SEQ})\s+was\s+"
    r"(?:(?:affectionately|fondly|popularly|lovingly)\s+)?(?:called|known)\b")
_KNOWN_AS = re.compile(
    rf"\b(?:known\s+as|referred\s+to\s+as|nicknamed|(?:affectionately|fondly|popularly|lovingly)\s+called)\s+"
    rf"[‘'\"“]?(?P<alias>{_NAME_SEQ})")
# How far back a "known as" alias looks for the name it belongs to.
_KNOWN_AS_REACH = 200


def _words(name: str) -> list[str]:
    return [w for w in name.replace(".", ". ").split() if w[:1].isupper() and not w.endswith(".")]


def _is_initials_of(acronym: str, name: str) -> bool:
    words = _words(name)
    return len(words) == len(acronym) and all(w[0] == a for w, a in zip(words, acronym))


def find_explicit_aliases(text: str, candidates: list[CandidateResult]) -> list[tuple[str, str]]:
    """(alias, name) pairs, the name an accepted candidate's text."""
    accepted = [c for c in candidates if c.state.decision == Decision.ACCEPTED]
    accepted_texts = {c.candidate.normalized_text for c in accepted}
    pairs: list[tuple[str, str]] = []
    # The capitalised words just before "(NTR)", as many as its letters, are
    # the name when they are its initials and an accepted name covers them -
    # whatever span that mention itself got.
    accepted_spans = [(c.candidate.start, c.candidate.end) for c in accepted]
    for m in _WORDS_THEN_ACRONYM.finditer(text):
        acronym = m.group("acronym")
        spans = [(m.start("words") + w.start(), m.start("words") + w.end())
                 for w in re.finditer(r"\S+", m.group("words"))][-len(acronym):]
        name = " ".join(text[s:e] for s, e in spans)
        if len(spans) != len(acronym) or not _is_initials_of(acronym, name):
            continue
        name_start, name_end = spans[0][0], spans[-1][1]
        if name in accepted_texts or any(s < name_end and name_start < e for s, e in accepted_spans):
            pairs.append((acronym, name))
    for m in _AS_CALLED.finditer(text):
        name = m.group("name")
        if name in accepted_texts or any(t.split()[-1] == name for t in accepted_texts if len(t.split()) > 1):
            pairs.append((m.group("alias"), name))
    by_end = sorted((c.candidate.end, c.candidate.normalized_text) for c in accepted if len(c.candidate.normalized_text.split()) > 1)
    for m in _KNOWN_AS.finditer(text):
        before = [name for end, name in by_end if m.start() - _KNOWN_AS_REACH <= end <= m.start()]
        if before and m.group("alias") != before[-1]:
            pairs.append((m.group("alias"), before[-1]))
    return list(dict.fromkeys(pairs))

"""
src.candidate.list_propagation
=================================
List propagation: when at least half of the items of a coordinated list are
names this document already accepted, the other items are names too -
"Ghantasala (music), M.L. Vasantha Kumari, Leela, Suseela and Madhavapeddi
(playback)", "Plato and Xenophon", "producers Nagireddy and Chakrapani".
Added 2026-10-01 from a school textbook's error brief (P7): credits and
cast lists were its largest group of missed people.

An item gets a `person_cue` detection with cue "list" (see
regex_detector.py) - evidence of a person, like a title. It still goes
through the full validation pipeline: a place or an organisation in a list
of people is rejected as usual. Only rejections for lack of evidence
(corroboration, grammar) are re-judged with the new evidence.

A list: name-shaped items (1-4 capitalised words or initials, each maybe
followed by a role in brackets) joined by ",", "and" or "&", a line break
allowed after a separator - with an "and"/"&" or at least three items:
"Finally, Ravi went" is no list (found on the benchmark: "Finally",
"Separately" and "Impossible" were accepted that way). An item that is an
ordinary word is never propagated either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.core.models import CandidateResult, Decision, Detection, DetectorName
from src.detection.regex_detector import _CITED_AUTHOR_RE, _NAME_SEQ, CITATION_CUE, PERSON_CUE_PATTERN

LIST_CUE = "list"
LIST_CONFIDENCE = 0.75
MIN_CONFIRMED_SHARE = 0.5
MIN_ITEMS_WITHOUT_AND = 3
# Rejections that only mean "not enough evidence" - the list is new evidence.
EVIDENCE_REJECTIONS = ("corroboration:", "grammar:")

_ITEM_RE = re.compile(rf"\b{_NAME_SEQ}")
_ROLE = re.compile(r"[ \t]*\([^()\n]{1,25}\)")
_AND = re.compile(r"\band\b|&")
_SEPARATOR = re.compile(r"[ \t]*(?:,[ \t]*(?:(?:and|&)(?=\s))?|(?:and|&)(?=\s))[ \t]*\n?[ \t]*")


@dataclass(frozen=True, slots=True)
class ListItem:
    start: int
    end: int
    text: str


def find_lists(text: str) -> list[list[ListItem]]:
    """Every run of 2+ name-shaped items joined by list separators."""
    lists: list[list[ListItem]] = []
    current: list[ListItem] = []
    has_and = False
    previous_end = -1

    def close() -> None:
        if len(current) >= MIN_ITEMS_WITHOUT_AND or (len(current) >= 2 and has_and):
            lists.append(current)

    for m in _ITEM_RE.finditer(text):
        item = ListItem(m.start(), m.end(), m.group())
        between = text[previous_end:item.start] if current else ""
        if current and between and _SEPARATOR.fullmatch(between):
            current.append(item)
            has_and = has_and or bool(_AND.search(between))
        else:
            close()
            current, has_and = [item], False
        role = _ROLE.match(text, item.end)
        previous_end = role.end() if role else item.end
    close()
    return lists


def list_detection(item: ListItem) -> Detection:
    return Detection(
        text=item.text, start=item.start, end=item.end, page_index=0,
        detector=DetectorName.REGEX, confidence=LIST_CONFIDENCE,
        metadata={"pattern": PERSON_CUE_PATTERN, "cue": LIST_CUE},
    )


def items_to_propagate(
    text: str, candidates: list[CandidateResult], is_ordinary_word: Callable[[str], bool] | None = None,
) -> list[ListItem]:
    """Items of lists whose items are at least half accepted names, that are
    not accepted themselves. An item counts as accepted when an accepted
    candidate overlaps it and ends where it ends ("Sri K. Venkata" and
    "K. Venkata" are the same item)."""
    accepted_ends: dict[int, list[int]] = {}
    # Where the initials of a cited author begin ("Hawton, K. Suicide in
    # prisoners", 2026-10-12): they belong to that author - no item of the
    # list, though "K. Suicide" has the shape of one.
    cited_initials: set[int] = set()
    for c in candidates:
        if c.state.decision == Decision.ACCEPTED:
            accepted_ends.setdefault(c.candidate.end, []).append(c.candidate.start)
        if any(d.metadata.get("cue") == CITATION_CUE for d in c.candidate.source_detections):
            author = _CITED_AUTHOR_RE.match(text, c.candidate.start)
            if author:
                cited_initials.add(author.start("initials"))
    out: list[ListItem] = []
    for items in find_lists(text):
        confirmed = [i for i in items if any(s < i.end for s in accepted_ends.get(i.end, ()))]
        if confirmed and len(confirmed) >= MIN_CONFIRMED_SHARE * len(items):
            out.extend(i for i in items if i not in confirmed and i.start not in cited_initials and not (
                is_ordinary_word is not None and len(i.text.split()) == 1 and is_ordinary_word(i.text)))
    return out

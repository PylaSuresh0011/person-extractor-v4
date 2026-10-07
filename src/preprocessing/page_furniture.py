"""
src.preprocessing.page_furniture
==================================
Removes page furniture from paged documents (PDFs) before any name is
looked for: running headers and footers, and stray watermark letters.

Found on a 274-page textbook (2026-10-01): its footer "English Class X /
Free distribution by T.S. Government 2019-20" came out of pypdf as one
line with the page number glued on ("31Free distribution by T.S.
Government 2019-20English Class X 31"), and "SCERT TELANGANA" closed
every page. The footer alone produced the extractor's most frequent
non-name, "English" (329 mentions).

A line counts as furniture when, with digits and spacing ignored (page
numbers change, and the same footer is spaced differently from page to
page), it appears on at least 20% of the pages - and on at least 3 - or
when it holds nothing but one or two capital letters (a vertical watermark
read letter by letter). Such lines are blanked, not removed, so every
other line keeps its line number in the reports.
"""

from __future__ import annotations

import re
from collections import Counter

MIN_SHARE_OF_PAGES = 0.2
MIN_PAGES = 3
_DIGITS = re.compile(r"\d+")
_WATERMARK_LINE = re.compile(r"\s*[A-Z]{1,2}\s*")
_NOT_LETTER = re.compile(r"[^A-Za-z]+")
# A line this long is keyed by its letters in any order: the text layer also
# glues a two-part footer in either order ("English Class X ... 2019-20" on
# some pages, "Free distribution ... 2019-20English Class X" on others -
# 40 of the textbook's pages). Two different lines with the same 12+
# letters are not a realistic collision; shorter lines keep their order.
_MIN_LETTERS_ANY_ORDER = 12


def _line_key(line: str) -> str:
    # Digits and spacing ignored: page numbers change, and the text layer
    # spaces the same footer differently ("Class XFree" / "Class X Free").
    letters = _NOT_LETTER.sub("", line)
    if len(letters) >= _MIN_LETTERS_ANY_ORDER:
        return "".join(sorted(letters))
    return _DIGITS.sub("#", "".join(line.split()))


def blank_page_furniture(pages: list[str]) -> list[str]:
    """The same pages with repeated header/footer lines and watermark-letter
    lines blanked. Fewer than MIN_PAGES pages: returned unchanged."""
    if len(pages) < MIN_PAGES:
        return pages
    seen_on = Counter(key for page in pages for key in {_line_key(l) for l in page.split("\n") if l.strip()})
    threshold = max(MIN_PAGES, MIN_SHARE_OF_PAGES * len(pages))
    repeated = {key for key, n in seen_on.items() if n >= threshold}
    return ["\n".join("" if line.strip() and (_line_key(line) in repeated or _WATERMARK_LINE.fullmatch(line))
                      else line for line in page.split("\n"))
            for page in pages]

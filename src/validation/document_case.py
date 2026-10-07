"""
src.validation.document_case
===============================
How a document itself capitalises a word - the "lowercase test" of the
textbook error brief (2026-10-01, P2/P8). A word the document also writes in
lowercase ("read", "story") is an ordinary word whose capital came from its
position; a word it never writes in lowercase ("Sumathi", "Relangi") behaves
like a name in this document.

Only words of prose count: "marcus.delaney@gmail.com" is no lowercase use of
"Marcus".
"""

from __future__ import annotations

import re


def lowercase_uses(word: str, text: str) -> int:
    """How often the document writes `word` in lowercase as a word of prose."""
    return len(re.findall(rf"(?<![\w@./\\-]){re.escape(word.lower())}(?![\w@/\\-]|\.\w)", text))


def mid_sentence_capitalised_uses(word: str, text: str) -> int:
    """How often it is capitalised right after a lowercase word, a comma or a
    semicolon - where only a name (or a proper noun) is capitalised."""
    return len(re.findall(rf"(?<=[a-z,;][ \t]){re.escape(word)}(?![A-Za-z])", text))


def capitalised_uses(word: str, text: str) -> int:
    """How often the document writes `word` exactly as given, as a whole word."""
    return len(re.findall(rf"(?<![\w@./\\-]){re.escape(word)}(?![\w@/\\-]|\.\w)", text))


class CaseCache:
    """Per-document memo of lowercase_uses - validators run once per
    mention, and the count is a scan of the whole document."""

    def __init__(self) -> None:
        self._lowercase: dict[str, int] = {}
        self._capitalised: dict[str, int] = {}

    def clear(self) -> None:
        self._lowercase.clear()
        self._capitalised.clear()

    def only_ever_capitalised(self, word: str, text: str) -> bool:
        """The document writes the word at least twice and never in
        lowercase: here it behaves as a proper noun, whatever a word list
        says. The frequency list of ordinary words (common_english_words.txt,
        the 10,000 most frequent words on the web) holds names too -
        "shakespeare", "william", "luther" - because people write about
        them often."""
        if word not in self._capitalised:
            self._capitalised[word] = capitalised_uses(word, text)
        return self._capitalised[word] >= 2 and self.lowercase_uses(word, text) == 0

    def lowercase_uses(self, word: str, text: str) -> int:
        if word not in self._lowercase:
            self._lowercase[word] = lowercase_uses(word, text)
        return self._lowercase[word]

"""
src.candidate.alias_resolution
=================================
Entity resolution for combined name lists: the forms one person appears
under - "Kalam", "Abdul Kalam", "Dr. Kalam"; "Mrs. Slater", "Amelia",
"Mrs. Amelia Slater" - become one row with the longest form as its name and
the others in an `aliases` column. Added 2026-10-01 from the textbook error
brief (P6: 42 duplicate entries).

Two names are one person when, with titles, case, accents and punctuation
ignored:
  - they are the same ("Dr. Kalam" / "Kalam");
  - one's words are part of the other's ("Kalam" / "Abdul Kalam", "Murthy"
    / "Mrs. Murthy") - only when every longer name containing it is the
    same person ("Murthy" next to "Sudha Murthy" and "Narayana Murthy"
    stays on its own), and the two are found close together: the same file,
    within NEAR_PAGES pages (or blocks of LINES_PER_PAGE lines). In a
    274-page anthology "Albert" (a play) is not "Albert Einstein" and
    "Mary" (a film) is not "Mary Kom";
  - initials stand for the given names ("R.K. Agnihotri" / "Rama Kant
    Agnihotri");
  - one word of 6+ letters has one letter more ("Shastry" / "Sastry",
    "Bachchan" / "Bacchan", "Amebedkar" / "Ambedkar") - a different letter
    is a different name ("Sarada" / "Sarala");
  - the document says so ("Nandamuri Taraka Ramarao (NTR)", "Manikda [as
    Ray was affectionately called]") - pairs the extractor finds, passed in.
A man's title and a woman's never match ("Mr. Slater" / "Mrs. Slater"), and
a generational suffix makes another person ("Henry Ford" / "Henry Ford II").

merge=False (the default since 2026-10-01) keeps one row per form and only
fills `aliases` with the other forms of the same person. Merging rows lowers
F1 as the textbook brief measures it - each merged duplicate was a correct
entry, so precision (correct entries / entries) drops - and the few wrong
merges ("Raju" into "M. Raju") cost recall: on the textbook, accepted+review
F1 84.0% -> 82.8% for duplicates 52 -> 21. merge=True does it anyway.

Used on combined lists only (the GUI's results, cli.py's batch file);
per-file reports keep one row per surface form.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Small generic gazetteer of titles and honorifics (brief P6).
# Royal and religious titles too (2026-10-01, round 3): "King Mahendra" is
# Mahendra, never the same person as a bare "King" (Martin Luther King).
_MALE = frozenset({"mr", "sri", "shri", "king", "emperor", "prince", "tsar", "czar", "sultan", "maharaja", "nawab",
                   "pope", "pastor", "swami"})
_FEMALE = frozenset({"mrs", "ms", "miss", "smt", "srimati", "madam", "queen", "empress", "princess", "rani",
                     "maharani", "begum"})
_TITLES = _MALE | _FEMALE | frozenset({
    "dr", "prof", "professor", "fr", "sir", "lord", "colonel", "col", "capt", "rev", "mahatma", "pandit",
    "president", "general", "archduke", "viceroy", "sheikh", "sheik", "maulana", "acharya", "sant", "guru",
})
_PUNCT = re.compile(r"[.,'’()\[\]\-]")
NEAR_PAGES = 2
LINES_PER_PAGE = 60


def place(source: str, location) -> tuple[str, int]:
    """Where a mention is, coarsely: its file and page - for text without
    pages, a block of LINES_PER_PAGE lines."""
    if location.page_number is not None:
        return source, location.page_number
    return source, location.line_number // LINES_PER_PAGE


def name_key(name: str) -> tuple[tuple[str, ...], str]:
    """(words without the titles in front, gender of those titles: "m", "f"
    or ""). Only leading titles go - "King" ends "Martin Luther King" as his
    surname - and a title on its own stays as the name ("King")."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    folded = re.sub(r"['’]s\b", "", folded)
    words = _PUNCT.sub(" ", folded).split()
    n = 0
    while n < len(words) - 1 and words[n] in _TITLES:
        n += 1
    titles = words[:n]
    gender = "m" if any(w in _MALE for w in titles) else "f" if any(w in _FEMALE for w in titles) else ""
    return tuple(words[n:]), gender


def _edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similar_word(a: str, b: str) -> bool:
    """Equal, or the longer has one letter more and is otherwise the same."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) < 6 or len(long) != len(short) + 1:
        return False
    return any(long[:k] + long[k + 1:] == short for k in range(len(long)))


def _same_name(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Equal, initials for given names, or one word spelt slightly differently."""
    if len(a) != len(b) or not a:
        return False
    if a == b:
        return True
    if len(a) > 1 and any(len(w) == 1 for w in a[:-1] + b[:-1]) and _similar_word(a[-1], b[-1]) and all(
            x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y))
            for x, y in zip(a[:-1], b[:-1])):
        return True
    differing = [(x, y) for x, y in zip(a, b) if x != y]
    return len(differing) == 1 and _similar_word(*differing[0])


def _near(a: dict, b: dict) -> bool:
    """Found close together - see NEAR_PAGES. Rows without places (built
    by older code) are not held to it."""
    if "places" not in a or "places" not in b:
        return True
    pages_by_file: dict[str, list[int]] = {}
    for source, page in b["places"]:
        pages_by_file.setdefault(source, []).append(page)
    return any(abs(page - other) <= NEAR_PAGES
               for source, page in a["places"] for other in pages_by_file.get(source, ()))


def _buckets(word: str) -> list[str]:
    """The word itself and, for a word long enough to have spelling
    variants, its first two and last three letters."""
    return [word, "<" + word[:2], ">" + word[-3:]] if len(word) >= 6 else [word]


def _more_specific(short: tuple[tuple[str, ...], str], long: tuple[tuple[str, ...], str]) -> bool:
    """long names the same person more fully: more words containing short's,
    or the same words with a man's or woman's title short lacks."""
    return _part_of(short[0], long[0]) or (short[0] == long[0] and not short[1] and bool(long[1]))


_GENERATIONAL = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "vi"})


def _part_of(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """Every word of the shorter name is in the longer one, or spelt nearly
    the same ("Ambedkar" / "Babasaheb Amebedkar") - and the longer one adds
    no generational suffix ("Henry Ford II" is not "Henry Ford")."""
    if len(short) >= len(long) or any(w in _GENERATIONAL for w in long if w not in short):
        return False
    return all(w in long or any(_similar_word(w, x) for x in long) for w in short)


class _Clusters:
    """Union-find that also keeps the title genders in each cluster."""

    def __init__(self, genders: list[str]) -> None:
        self.parent = list(range(len(genders)))
        self.genders = [{g} if g else set() for g in genders]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj
            self.genders[rj] |= self.genders[ri]

    def compatible(self, i: int, j: int) -> bool:
        """No man's title on one side and woman's on the other - across
        everything each one is already joined with."""
        return len(self.genders[self.find(i)] | self.genders[self.find(j)]) <= 1


def resolve_aliases(entries: list[dict], explicit_aliases: Iterable[tuple[str, str]] = (),
                    merge: bool = False) -> list[dict]:
    """entries: combined rows {"display", "occurrences", "decision", "files",
    ...}. explicit_aliases: (alias, full name) pairs the documents state.
    Each row gets "aliases": the other forms of its person, most frequent
    first. merge=True: one row per person instead - the fullest form's row,
    occurrences summed, "accepted" if any form was, files merged."""
    if not entries:
        return []
    keys = [name_key(e["display"]) for e in entries]
    clusters = _Clusters([gender for _, gender in keys])
    compatible = clusters.compatible

    by_key: dict[tuple[str, ...], list[int]] = {}
    for i, (words, _) in enumerate(keys):
        if words:
            by_key.setdefault(words, []).append(i)
    # The same name - unless only one has a man's or woman's title, which
    # makes it the more specific one ("Mrs. Murthy" / "Murthy": part-of below).
    for i, (words, gender) in enumerate(keys):
        for j in by_key.get(words, ()):
            if j != i and keys[j][1] == gender:
                clusters.union(i, j)

    for alias, full in explicit_aliases:
        ia, ifull = by_key.get(name_key(alias)[0], []), by_key.get(name_key(full)[0], [])
        if ia and ifull:
            clusters.union(ia[0], ifull[0])

    # Only names sharing a word, or a long word's start or end, can match.
    index: dict[str, set[int]] = {}
    for i, (words, _) in enumerate(keys):
        for w in words:
            for bucket in _buckets(w):
                index.setdefault(bucket, set()).add(i)

    def related(i: int) -> set[int]:
        out: set[int] = set()
        for w in keys[i][0]:
            for bucket in _buckets(w):
                out |= index.get(bucket, set())
        out.discard(i)
        return out

    # Initials and spelling variants - only an unambiguous match.
    for i, (words, _) in enumerate(keys):
        matches = {clusters.find(j) for j in related(i)
                   if len(keys[j][0]) == len(words) and keys[j][0] != words
                   and _same_name(words, keys[j][0]) and compatible(i, j)}
        if len(matches) == 1:
            clusters.union(i, matches.pop())

    # Part of a longer name - shortest last, so "Abdul Kalam" has joined the
    # full name before "Kalam" looks for the longer names containing it; a
    # second pass for names the first pass joined ("Mr. Slater" to "Henry
    # Slater" makes "Mrs. Slater" unambiguous).
    order = sorted(range(len(entries)), key=lambda i: -len(keys[i][0]))
    for _ in range(2):
        for i in order:
            words = keys[i][0]
            if not words:
                continue
            supersets = {clusters.find(j) for j in related(i)
                         if _more_specific(keys[i], keys[j]) and compatible(i, j) and _near(entries[i], entries[j])}
            if len(supersets) == 1:
                clusters.union(i, supersets.pop())

    groups: dict[int, list[int]] = {}
    for i in range(len(entries)):
        groups.setdefault(clusters.find(i), []).append(i)
    if not merge:
        out = []
        for i, entry in enumerate(entries):
            row = {k: v for k, v in entry.items() if k != "places"}
            others = sorted((j for j in groups[clusters.find(i)] if j != i), key=lambda j: -entries[j]["occurrences"])
            row["aliases"] = [entries[j]["display"] for j in others]
            out.append(row)
        return sorted(out, key=lambda e: (-e["occurrences"], e["display"]))
    out = []
    for members in groups.values():
        # The fullest form names the row: most words, then a title, then most found.
        members.sort(key=lambda i: (-len(keys[i][0]), -bool(keys[i][1]), -entries[i]["occurrences"]))
        head = dict(entries[members[0]])
        head["files"] = list(head.get("files", []))
        head.pop("places", None)
        others = sorted(members[1:], key=lambda i: -entries[i]["occurrences"])
        head["aliases"] = [entries[i]["display"] for i in others]
        for i in others:
            head["occurrences"] += entries[i]["occurrences"]
            if entries[i].get("decision") == "accepted":
                head["decision"] = "accepted"
            for f in entries[i].get("files", []):
                if f not in head["files"]:
                    head["files"].append(f)
        out.append(head)
    return sorted(out, key=lambda e: (-e["occurrences"], e["display"]))

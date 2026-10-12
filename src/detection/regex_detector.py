"""
src.detection.regex_detector
==============================
Regex-based person-name detector.

Two complementary patterns:
1. Titled names: an honorific/title followed by 1-4 capitalized tokens
   ("Dr. Suresh Kumar", "Mrs. Jane A. Doe").
2. Bare capitalized sequences: 2-4 consecutive capitalized tokens with no
   title ("John Smith"). Lower base confidence since bare capitalization
   alone is a weaker signal (also matches place/org names) - the
   Validation Firewall is responsible for filtering those, not the regex
   detector itself.

Both patterns use [ \\t] (never \\s) between tokens so a match can NEVER
span a newline - i.e. never merge two different lines/rows into one fake
"name". (This was a real bug found and fixed during V3 adversarial
testing: \\s+ let a match bleed across a table row boundary.) The
whitespace run between tokens is further capped at 1-2 characters (see
_SAFE_SEP) so a match also can't bridge 3+ spaces of deliberate table-
column padding on the SAME line, which \\t-vs-\\s alone doesn't prevent.

The titled pattern also allows STACKED titles ("Prof. Dr. Lalita Kher")
to all be consumed as the title prefix, not just the first one - without
that, the second title word ("Dr.") looks exactly like a valid name
token and gets wrongly captured as if it were the person's actual name,
splitting the real name off as a separate, unlabeled bare match.

A trailing possessive ('s / 's) is stripped from the matched span before
it becomes a Detection, same as dictionary_detector.py already does for
its own matches. Found via real testing: without this, "Wrenna
Solkiewicz's number changed" produced a SEPARATE candidate normalized to
"Wrenna Solkiewicz's" - a different aggregation key than plain "Wrenna
Solkiewicz" - fragmenting one real person's mentions into two weaker,
independently-scored report entries instead of contributing to the same
occurrence count.
"""

from __future__ import annotations

import re
from typing import Callable

from src.core.models import Detection, DetectorName
from src.detection.base_detector import BaseDetector
from src.detection.dictionary_detector import _strip_possessive
from src.preprocessing.latin import LATIN_LETTERS, LATIN_UPPER

# Token: capitalized word, allows hyphens/apostrophes (O'Brien, Smith-Jones).
# The accented-letter ranges (Latin-1 Supplement + Latin Extended-A) exist
# because the plain-ASCII-only version of this pattern silently couldn't
# match accented Latin names at all - not "match them oddly", literally
# zero detections, since a single non-ASCII character (e.g. the ç in
# "Francois") breaks the token match outright and the whole name vanishes
# with no trace anywhere in the pipeline. Deliberately still ASCII-first
# in the class (most real-world names are pure ASCII) with the accented
# ranges appended, not a full \w-based Unicode rewrite - keeps the
# character set to Latin-script names specifically, consistent with
# LanguageValidator's non-Latin-script hard reject elsewhere in this
# pipeline.
#
# The letter classes now come from src/preprocessing/latin.py, shared with
# StructureValidator/DictionaryDetector - which had stayed ASCII-only and
# were rejecting/fragmenting every accented name this detector found. The
# old hand-typed first-letter class [A-ZÀ-ÖØ-Þ] also missed Ł/Š/Č/Ž and
# Romanian/Vietnamese capitals (see latin.py's docstring).
_LATIN_LETTER = LATIN_LETTERS
# Joined initials ("K.N. Anandan", "S.V. Ranga Rao", 2026-10-01) are one token:
# without the first alternative "K.N." matched as nothing and the name
# was cut to "N. Anandan" or missed.
_NAME_TOKEN = rf"(?:(?:[{LATIN_UPPER}]\.){{2,4}}|[{LATIN_UPPER}][{_LATIN_LETTER}'\-]*\.?)"

_TITLES = (
    r"Dr|Mr|Mrs|Ms|Miss|Prof|Professor|Rev|Fr|Sir|Madam|Capt|Col|"
    r"Gen|Lt|Sgt|Hon|Judge|Sen|Rep|Amb|"
    # Indian honorifics (2026-10-01): "Sri Navuluri Peraiah, SA",
    # "Smt. B. Syama Sundari" on a textbook's credit pages.
    r"Sri|Shri|Smt|Srimati|"
    # Religious titles (round 3, 2026-10-01): "Pastor Niemoeller".
    r"Pastor|Swami|Guru|Bishop|Father"
)

# A single non-title token, used to build the separator: two name tokens
# may be joined by [ \t]{1,2} ONLY if the whitespace is not immediately
# preceded by a genuine sentence-ending period/!/? - the match must
# never cross a real sentence boundary (same principle that already
# prevents crossing a newline). The one deliberate exception: a period
# directly after a SINGLE capital letter ("J.", "K.") is a mid-name
# initial, not a sentence end, and must still be allowed to join to the
# next token ("Cortland J. Dahl" must not be truncated to "Cortland J").
#
# Capped at 1-2 whitespace characters (not unbounded [ \t]+) so a match
# can tolerate one accidental extra space (real chat data really does
# have this - e.g. "Donald  Richard" with a stray double space) without
# also being able to bridge 3+ spaces of deliberate column padding in
# tabular/report-style text ("Kavya Bhandari    Design       Active").
# 3+ spaces between words is essentially never natural prose spacing;
# treating it as a hard boundary (same principle as never crossing a
# newline) stops a name from swallowing adjacent table-column values.
_SAFE_SEP = rf"(?:(?<=\b[{LATIN_UPPER}]\.)[ \t]{{1,2}}|(?<![.!?])[ \t]{{1,2}})"

_TITLED_NAME_RE = re.compile(
    rf"\b(?:{_TITLES})\.?[ \t]{{1,2}}(?:(?:{_TITLES})\.?[ \t]{{1,2}})*"
    rf"(?:{_NAME_TOKEN}{_SAFE_SEP}){{0,3}}{_NAME_TOKEN}"
)

_BARE_NAME_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:{_SAFE_SEP}{_NAME_TOKEN}){{1,3}}\b"
)

# Person cues (2026-10-01, textbook error brief P7): words around a name that
# say it is a person, the way a title does. The match is the name only.
# Patterns treated as that kind of evidence: "titled" and "person_cue".
PERSON_CUE_PATTERN = "person_cue"
PERSON_MARKER_PATTERNS = frozenset({"titled", PERSON_CUE_PATTERN})
_NAME_SEQ = rf"{_NAME_TOKEN}(?:{_SAFE_SEP}{_NAME_TOKEN}){{0,3}}"
_IN_LAW = r"(?:brother|sister|son|daughter|father|mother)[ \t]*-[ \t]*in[ \t]*-[ \t]*law"
_KINSHIP = (rf"(?:{_IN_LAW}|wife|husband|sons?|daughters?|brothers?|sisters?|cousins?|father|mother|grandson|"
            rf"granddaughter|nephew|niece|married|parents[ \t]+(?:were|are))")
_PERSON_CUE_RES = (
    # "his wife Sumanthamma", "married Sumathi", "daughters Rashmitha (3) and
    # Ashwitha", "brother -in-law Jallaluddin" - lowercase relation words
    # only, so "Mother Teresa" stays one name - and the label form of a
    # biographical sketch, "Father: Frank Hawking".
    ("kinship", re.compile(
        rf"(?:\b{_KINSHIP}[ \t]*,?|\b(?:Wife|Husband|Spouse|Father|Mother|Son|Daughter)[ \t]*:)(?:[ \t]+|[ \t]*\n[ \t]*)"
        rf"(?P<name>{_NAME_SEQ})(?:[ \t]*\([^()\n]{{1,20}}\))?(?:[ \t]*(?:,|and)[ \t]+(?P<name2>{_NAME_SEQ}))?")),
    # "Yours faithfully,\nRamu" - the name on the same line or the next.
    ("sign_off", re.compile(
        rf"\bYours[ \t]+(?:faithfully|sincerely|truly|lovingly|affectionately|obediently)[ \t]*,?[ \t]*\n?[ \t]*"
        rf"(?P<name>{_NAME_SEQ})")),
    # "Dear Ravi," / "My dear Sita!" - not "The Dear Departed" (a title).
    ("salutation", re.compile(rf"(?<![Tt]he )\b[Dd]ear[ \t]+(?P<name>{_NAME_SEQ})(?=[ \t]*[,!\n])")),
    # Round 3 (2026-10-01, error brief v2 N7/N12):
    # "Sun-Yat-Sen (1866-1925)" - a life span in brackets.
    ("lifespan", re.compile(rf"(?P<name>{_NAME_SEQ})[ \t]*\((?:b\.[ \t]*)?1?\d{{3}}[ \t]*[-–][ \t]*\d{{2,4}}\)")),
    # "Ho Chi Minh (who spent time in France", "Nasser, who".
    ("who", re.compile(rf"(?P<name>{_NAME_SEQ})[ \t]*(?:\(|,)[ \t]*who\b")),
    # "economists like JM Keynes", "the dictator Franco", "poet Rabindranath Tagore".
    ("role", re.compile(
        r"\b(?:leaders?|dictators?|emperors?|novelists?|writers?|poets?|historians?|economists?|scientists?|"
        r"philosophers?|journalists?|activists?|reformers?)(?:[ \t]+(?:like|such[ \t]+as))?[ \t]+"
        rf"(?P<name>{_NAME_SEQ})")),
    # "George Orwell wrote", "Keynes argued".
    ("agent_verb", re.compile(rf"(?P<name>{_NAME_SEQ})[ \t]+(?:said|says|wrote|writes|argued|argues|declared)\b")),
    # An author line: "- Cedric Mount", "– Pydimarri Venkata Subba Rao".
    ("byline", re.compile(rf"(?m)^[ \t]*[-–—][ \t]*(?P<name>{_NAME_SEQ})[ \t]*$")),
)
# Citation cues (2026-10-12): the way a paper names the people it cites. A
# reference list writes each author as surname, initials - one or more of
# them, joined by ";", "," or "and", and closed by the year: "Abdolmohammadi,
# M; Harris, J and Smith, K (2002), ...", "Fergusson, D.M., Woodward, L.J. &
# Horwood, L.J. (1997)." - and "et al." after a name stands for more people.
# In the three research papers of a case folder, 115 of the 139 cited authors
# the scan did not accept stood in that form; a lone surname no list holds
# was never even a candidate. The match is the surname.
#
# The layout has to be the document's own: at least MIN_REFERENCE_ENTRIES
# entries closed by a year - one author alone with the year in brackets, since
# "Sorry, I 2000 pay" in a chat is an amount. In a document that has them,
# an author list is also one that runs to three authors or to "et al."
# ("Fazel, S., Cartwright, J., Norman-Nott, A. & Hawton, K. Suicide in
# prisoners"), and one author before a bare year ("Rohrman, B. 1986.").
CITATION_CUE = "citation"
MIN_REFERENCE_ENTRIES = 3
# The surname may carry a suffix ("Kinney Jr, W.R"). Initials are one to four
# letters set apart by dots, spaces or a hyphen, or two letters written
# together - "ADAM" in "(... Monitoring Programme, ADAM, 2000)" is an acronym.
_CITED_AUTHOR_RE = re.compile(
    rf"(?<![{_LATIN_LETTER}'\-])(?P<name>[{LATIN_UPPER}][{_LATIN_LETTER}'\-]+(?:[ \t](?:Jr\.?|Sr\.?|II|III|IV))?),[ \t]{{0,2}}"
    rf"(?P<initials>[{LATIN_UPPER}](?:[{LATIN_UPPER}]|(?:(?:\.[ \t]?-?|[ \t]|-)[{LATIN_UPPER}]){{1,3}})?\.?)"
    rf"(?![{_LATIN_LETTER}])")
_AUTHOR_JOIN_RE = re.compile(r"\s{0,3}(?:[;,]\s{0,3})?(?:and\s{1,3}|&\s{0,3})?")
_YEAR = r"(?:1[6-9]|20)\d\d[a-z]?"
# A PDF's text layer can space a digit off inside the brackets: "(1 999)", "(2008 )".
_YEAR_IN_BRACKETS = r"\(\s?(?:1\s?[6-9]|2\s?0)\s?\d\s?\d[a-z]?\s?\)"
_ET_AL_TAIL = r",?\s{0,3}et\s{1,3}al\b\.?"
_YEAR_IN_BRACKETS_RE = re.compile(rf"(?:{_ET_AL_TAIL})?\s{{0,3}},?\s{{0,3}}{_YEAR_IN_BRACKETS}")
_YEAR_AFTER_AUTHORS_RE = re.compile(
    rf"(?:{_ET_AL_TAIL})?\s{{0,3}}[.,]?\s{{0,3}}(?:{_YEAR_IN_BRACKETS}|{_YEAR}(?!\d))")
_ET_AL_AFTER_AUTHORS_RE = re.compile(rf"(?:\s{{0,3}},\s{{0,3}}{_NAME_SEQ})?{_ET_AL_TAIL}")
_ET_AL_RE = re.compile(rf"(?P<name>{_NAME_SEQ})[ \t]+et[ \t]+al\b")
# "Nasser of Egypt, Sukarno of Indonesia" - a name "of" a country (kept
# when is_place says the right-hand side is one).
_OF_COUNTRY_RE = re.compile(rf"(?P<name>{_NAME_SEQ})[ \t]+of[ \t]+(?P<place>[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?)")
# "Malcolm X" - a name and a single X (not "Class X", "Unit X": ordinary words).
_NAME_X_RE = re.compile(r"\b(?P<name>[A-Z][a-z]{2,}[ \t]X)(?![\w.])")
# "Sarath : Oh! What happened?" - a name opening a line, a colon, then speech.
# Kept only for a label used at least twice (a play or dialogue repeats its
# speakers), and never on a line of "|"-delimited fields.
_SPEAKER_RE = re.compile(rf"(?m)^[ \t]*(?P<name>{_NAME_SEQ})[ \t]*:[ \t]*(?=[\"“'‘(]?[A-Z])(?P<speech>[^\n]*)")
MIN_SPEAKER_LINES = 2
# A speaker's lines must read as speech: a lowercase word in at least one,
# no second "Key:" and no code characters in any - not "GestureMode: MULTI_TOUCH
# DeviceType: TOUCH_SCREEN" or 'Ssp: "PatternMatcher{LITERAL: ...' from an
# Android dump (real case scan, 2026-10-01).
_SPEECH_WORD = re.compile(r"(?<![\w.@/])[a-z]{2,}(?![\w.@/])")
_ANOTHER_LABEL = re.compile(r"\b\w+[ \t]*:[ \t]")
_CODE_CHARS = re.compile(r"[{}<>=\[\]]")
_CONTRACTION_END = re.compile(r"['’](?:t|s|m|ve|re|ll|d)$")


class RegexDetector(BaseDetector):
    name = DetectorName.REGEX

    TITLED_CONFIDENCE = 0.90
    BARE_CONFIDENCE = 0.55
    CUE_CONFIDENCE = 0.85

    def __init__(self, is_ordinary_word: Callable[[str], bool] | None = None,
                 is_place: Callable[[str], bool] | None = None,
                 is_organization: Callable[[str], bool] | None = None) -> None:
        self.is_place = is_place
        # A cue before ordinary words is no person cue: "Dear Student,",
        # "Mother : Money, money" (a play's speaker label, not "Father: Frank
        # Hawking"), speaker labels "Study Skills :" - skipped when this check
        # is given (DetectorManager passes the knowledge base's). Contractions
        # ("Mother : Didn't you hear?") always are. Nor is one before an
        # organisation's name and ordinary words (2026-10-12): "Dear Zomato
        # Customer Service," made that phrase a person, 26 times in a case.
        self.is_ordinary_word = is_ordinary_word
        self.is_organization = is_organization

    def detect(self, text: str, page_index: int) -> list[Detection]:
        if not text:
            return []

        detections: list[Detection] = []
        claimed_spans: list[tuple[int, int]] = []

        for match in _TITLED_NAME_RE.finditer(text):
            raw = match.group().strip()
            stemmed, trim = _strip_possessive(raw)
            detections.append(
                Detection(
                    text=stemmed,
                    start=match.start(),
                    end=match.end() - trim,
                    page_index=page_index,
                    detector=self.name,
                    confidence=self.TITLED_CONFIDENCE,
                    metadata={"pattern": "titled"},
                )
            )
            claimed_spans.append((match.start(), match.end()))

        for match in _BARE_NAME_RE.finditer(text):
            span = (match.start(), match.end())
            if self._overlaps(span, claimed_spans):
                continue  # already covered by a stronger titled match
            raw = match.group().strip()
            stemmed, trim = _strip_possessive(raw)
            detections.append(
                Detection(
                    text=stemmed,
                    start=match.start(),
                    end=match.end() - trim,
                    page_index=page_index,
                    detector=self.name,
                    confidence=self.BARE_CONFIDENCE,
                    metadata={"pattern": "bare"},
                )
            )

        for cue, pattern in _PERSON_CUE_RES:
            for match in pattern.finditer(text):
                for group in ("name", "name2"):
                    if (group in pattern.groupindex and match.group(group)
                            and not self._ordinary_label(match.group(group))):
                        detections.append(self._cue_detection(match, group, cue, page_index))
        if self.is_place is not None:
            for m in _OF_COUNTRY_RE.finditer(text):
                if self.is_place(m.group("place")) and not self._no_person_of_a_place(m.group("name")):
                    detections.append(self._cue_detection(m, "name", "of_country", page_index))
        for m in _NAME_X_RE.finditer(text):
            if not self._ordinary_label(m.group("name").split()[0]):
                detections.append(self._cue_detection(m, "name", "name_x", page_index))
        lines_by_label: dict[str, list[re.Match]] = {}
        for m in _SPEAKER_RE.finditer(text):
            if not self._ordinary_label(m.group("name")):
                lines_by_label.setdefault(m.group("name"), []).append(m)
        for lines in lines_by_label.values():
            speeches = [m.group("speech") for m in lines]
            if (len(lines) >= MIN_SPEAKER_LINES and not any(_looks_like_data(s) for s in speeches)
                    and any(_SPEECH_WORD.search(s) for s in speeches)):
                detections.extend(self._cue_detection(m, "name", "speaker", page_index) for m in lines)
        detections.extend(self._cue_detection(m, "name", CITATION_CUE, page_index) for m in _cited_people(text))

        return detections

    def _ordinary_label(self, label: str) -> bool:
        words = label.split()
        if any(_CONTRACTION_END.search(w) for w in words):
            return True
        return self.is_ordinary_word is not None and all(
            self.is_ordinary_word(w.rstrip("."))
            or (self.is_organization is not None and self.is_organization(w.rstrip("."))) for w in words)

    def _no_person_of_a_place(self, name: str) -> bool:
        """ "X of <country>" names a person when X is a name ("Nasser of
        Egypt") - not an abbreviation ("Govt. of India"), a phrase with an
        ordinary word in it ("Population Pyramid of India") or a plural
        ("the Garhwal Himalayas of Uttarakhand"), 2026-10-12."""
        words = name.split()
        last = words[-1]
        return (name.endswith(".")
                or (len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")))
                or (self.is_ordinary_word is not None and any(self.is_ordinary_word(w.rstrip(".")) for w in words)))

    def _cue_detection(self, match: re.Match, group: str, cue: str, page_index: int) -> Detection:
        raw = match.group(group)
        stemmed, trim = _strip_possessive(raw)
        # A sentence's full stop is not part of the name ("married Sumathi.");
        # an initial's dot is ("Tom Watson Sr.", "K.").
        if stemmed.endswith(".") and len(stemmed.split()[-1]) > 3 and stemmed.split()[-1][:-1] not in ("Sr", "Jr"):
            stemmed, trim = stemmed[:-1], trim + 1
        return Detection(
            text=stemmed, start=match.start(group), end=match.end(group) - trim, page_index=page_index,
            detector=self.name, confidence=self.CUE_CONFIDENCE, metadata={"pattern": PERSON_CUE_PATTERN, "cue": cue},
        )

    @staticmethod
    def _overlaps(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
        start, end = span
        return any(not (end <= o_start or start >= o_end) for o_start, o_end in others)


def _cited_people(text: str) -> list[re.Match]:
    """Matches whose "name" group is a cited person: a name before "et al.",
    and every author of a reference-list entry - see CITATION_CUE."""
    cited = list(_ET_AL_RE.finditer(text))
    entries: list[list[re.Match]] = []
    for author in _CITED_AUTHOR_RE.finditer(text):
        if entries and _AUTHOR_JOIN_RE.fullmatch(text, entries[-1][-1].end(), author.start()):
            entries[-1].append(author)
        else:
            entries.append([author])
    closed = [entry for entry in entries
              if (_YEAR_AFTER_AUTHORS_RE if len(entry) > 1 else _YEAR_IN_BRACKETS_RE).match(text, entry[-1].end())]
    if len(closed) >= MIN_REFERENCE_ENTRIES:
        cited.extend(author for entry in entries for author in entry
                     if len(entry) >= 3 or _YEAR_AFTER_AUTHORS_RE.match(text, entry[-1].end())
                     or (len(entry) > 1 and _ET_AL_AFTER_AUTHORS_RE.match(text, entry[-1].end())))
    return cited


def _looks_like_data(speech: str) -> bool:
    """A record field or a dump line, not speech - see _SPEECH_WORD."""
    return "|" in speech or _ANOTHER_LABEL.search(speech) is not None or _CODE_CHARS.search(speech) is not None

"""
src.preprocessing.typeset
===========================
Normalises text extracted from typeset documents (PDFs) before names are
looked for. PDF text layers glue and split words in ways a person reading
the page never sees; on a 274-page textbook (2026-10-01) this cut or hid
names such as "G.Gopal Reddy", "S.V . Ranga Rao" and "Abel Merry-/weather".

  - initials: "G.Gopal" -> "G. Gopal", "S.V . Ranga" -> "S.V. Ranga",
    "Dr.A.L. Khanna" -> "Dr. A.L. Khanna"
  - a word hyphenated across a line break is joined: "Merry-\\nweather" ->
    "Merryweather\\n" (the line break moves after the word, so the line
    count is unchanged)
  - a run of underscores (a fill-in blank) becomes one short placeholder
  - initials written without dots before a name (round 3, 2026-10-01):
    "IK Gujral" -> "I.K. Gujral", "L K Advani" -> "L.K. Advani" - only before
    a name word (is_name_word) and never for a known acronym ("US", "BJP")

PDFs only - chat exports and logs are left exactly as they are.
"""

from __future__ import annotations

import re

_SPACE_BEFORE_INITIAL_DOT = re.compile(r"\b([A-Z]) +\.(?= *[A-Z])")
_INITIAL_GLUED_TO_WORD = re.compile(r"\b([A-Z])\.(?=[A-Z][a-z])")
# A title glued to an initial or a name: "Dr.A.L. Khanna", "Dr.Homen Thangjam".
_TITLE_GLUED_TO_INITIAL = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Sri|Smt|Shri)\.(?=[A-Z](?:\.|[a-z]))")
_HYPHENATED_LINE_BREAK = re.compile(r"([A-Za-z])-\n([a-z]+)")
_BLANK = re.compile(r"_{3,}")
# Initials without dots, glued ("IK", "VP", "JMK") or spaced ("L K", "M G"),
# then a name word.
_GLUED_INITIALS = re.compile(r"(?<![A-Za-z.&])([A-Z]{2,3})[ \t]+(?=([A-Z][a-z]{2,})\b)")
_SPACED_INITIALS = re.compile(r"(?<![A-Za-z.&])([A-Z])[ \t]([A-Z])(?:[ \t]([A-Z]))?[ \t]+(?=([A-Z][a-z]{2,})\b)")
# Common acronyms that come before capitalised words - small generic list.
ACRONYMS = frozenset("""AD BC CE BCE UN US UK USA EU SC ST OBC PG RSS TDP PGT BJP NDA UPA CPI CPM INC DMK MP MLA MLC
PM CM IAS IPS IFS GDP GNP NGO TV FM AM RBI SBI IT IIT IIM AI UGC DNA HIV ATM SMS TB TT NSS NCC PHC CSR ICT
GST VAT ISRO NASA WHO ILO IMF WTO SAARC ASEAN NATO USSR UAE JNU HMT DRDO ONGC LIC BSF CRPF RTI IPC CRPC
PIL MRP OK ID BA MA MSC BSC PHD MBA MBBS IQ HR SP DSP SI""".split())


_TITLE = r"(?:Dr|Mr|Mrs|Ms|Smt|Sri|Shri|Fr|Prof|Professor|Sir|Colonel|Capt|Gen)\.?"
_NAME_WORD = r"[A-Z][a-z]+(?:['’][a-z]+)?"
# A role word before a single name ("the emperor, Bao / Dai", round 3,
# 2026-10-01) - the name is a person's, so the next line may continue it.
_ROLE = r"(?:[Ee]mperor|[Kk]ing|[Qq]ueen|[Ll]eader|[Pp]resident|[Gg]eneral|[Dd]ictator|[Pp]remier|[Mm]inister|Pastor|Swami)"
# A name at the end of a line - a title and name words, two or more name
# words, or a role word and a name - and, after the line break(s), the next
# line's first word: a name word followed on its line by a lowercase word or
# by , ; :
_WRAPPED_NAME = re.compile(
    rf"(?:\b{_TITLE}[ \t]+(?:[A-Z]\.[ \t]*)*{_NAME_WORD}(?:[ \t]+{_NAME_WORD})*"
    rf"|\b{_ROLE},?[ \t]+{_NAME_WORD}"
    rf"|\b{_NAME_WORD}(?:[ \t]+{_NAME_WORD})+)"
    rf"(?P<gap>[ \t]*\n\s*)(?P<next>{_NAME_WORD})(?=[ \t]*[,;:]|[ \t]+[a-z])")
_SENTENCE_STARTERS = frozenset(
    "The This That These Those He She It They We You I A An In On At By For From With His Her Its Their Our My "
    "Your When If As But And Or So Then There Here What Who Why How After Before Once While Although Because "
    "Some Many Most All Every Each No Not One Two Yes Dear Thanking Yours "
    # Figure captions and tables start a line of their own, never a name's end.
    "Fig Figure Table Map Graph Chart Source Note".split())


# A line of running text that ends on a name word, and the next line opening
# with one to three name words that the sentence then goes on from (a
# lowercase word, or , ; : . ) after them): "…in Hyderabad. Mohammad / Quli
# Qutub Shah, the fifth Sultan", "…me and Zaki (Syed / Moinuddin Zaki) from".
_WRAPPED_IN_RUNNING_TEXT = re.compile(
    rf"\b(?P<last>{_NAME_WORD})(?P<gap>[ \t]*\n[ \t]*)(?P<next>{_NAME_WORD})"
    rf"(?=(?:[ \t]+{_NAME_WORD}){{0,2}}(?:[ \t]*[,;:.)]|[ \t]+[a-z]))")
# How wide a line must be, against the wider of its two neighbours, to count
# as a full line of its paragraph.
_FULL_LINE_SHARE = 0.8
_FIRST_LETTER = re.compile(r"[^\W\d_]")


def _starts_lowercase(line: str) -> bool:
    letter = _FIRST_LETTER.search(line)
    return bool(letter) and letter.group().islower()


def _wrap_of_running_text(text: str, line_start: int, line_end: int, next_start: int) -> bool:
    """The line break after text[line_start:line_end] is where a paragraph's
    line ran out of room: the line is about as wide as the lines around it,
    and the lines around it read as prose - one of them starts with a
    lowercase word. A heading or a short last line is narrower than its
    neighbour; in verse and in lists every line starts with a capital
    ("So let it be with Caesar. The noble Brutus / Hath told you Caesar was
    ambitious:")."""
    prev_start = text.rfind("\n", 0, max(line_start - 1, 0)) + 1 if line_start else 0
    prev_line = text[prev_start:line_start - 1] if line_start else ""
    next_end = text.find("\n", next_start)
    next_end = len(text) if next_end == -1 else next_end
    after_end = text.find("\n", next_end + 1)
    after_line = text[next_end + 1:len(text) if after_end == -1 else after_end]
    head_line, next_line = text[line_start:line_end], text[next_start:next_end]
    if len(head_line.rstrip()) < _FULL_LINE_SHARE * max(len(prev_line.rstrip()), len(next_line.rstrip())):
        return False
    return any(_starts_lowercase(line) for line in (prev_line, head_line, after_line))


def join_wrapped_names(text: str, is_ordinary_word=None, is_place=None) -> str:
    """'Dr. Vikram\\nSarabhai of the Dept.' -> 'Dr. Vikram Sarabhai of the Dept.'
    The line break (and any blank lines between, e.g. across a page break)
    becomes one space, and the rest of the gap moves to just after the
    joined word - the text keeps its exact length, so the line index the
    pipeline built BEFORE this join still points reports at the real lines.
    Not joined when the next line's first word is a sentence starter or,
    with is_ordinary_word given, an ordinary word: on credit pages a name
    line is followed by "Director, SCERT", not by more of the name.

    A second pass (2026-10-07) joins a name that the wrap of a paragraph's
    line cut anywhere - after one word, or before several: where a line of
    running text ran out of room (_wrap_of_running_text), the words on both
    sides of the break are neighbours in one sentence, exactly as they would
    be on one line. On three textbooks this is how "Mohammad / Quli Qutub
    Shah", "Syed / Moinuddin Zaki", "Benito / Mussolini", "Bava / Mahaliya"
    and "Indira / Gandhi" were cut in two. Neither word may be an ordinary
    word or a sentence starter; the word before the break may not be a
    possessive ("Ghatothkacha's / Manadi Sodara Prema" - a possessive ends
    the name); and the next line may not open with a place (is_place): "Dr.
    Homen Thangjam / Manipur University," is a name and, below it, where he
    works."""
    def _join(m: re.Match) -> str:
        word = m.group("next")
        if word in _SENTENCE_STARTERS or (is_ordinary_word is not None and is_ordinary_word(word)):
            return m.group(0)
        gap = m.group("gap")
        head = m.group(0)[:m.start("gap") - m.start()]
        return head + " " + word + " " * (len(gap) - 1)

    def _join_in_running_text(m: re.Match) -> str:
        last, word = m.group("last"), m.group("next")
        if any(w in _SENTENCE_STARTERS or (is_ordinary_word is not None and is_ordinary_word(w)) for w in (last, word)):
            return m.group(0)
        if last.endswith(("'s", "’s")) or (is_place is not None and is_place(word)):
            return m.group(0)
        line_start = m.string.rfind("\n", 0, m.start()) + 1
        if not _wrap_of_running_text(m.string, line_start, m.end("last"), m.start("next")):
            return m.group(0)
        return last + " " + word + " " * (len(m.group("gap")) - 1)

    text = _WRAPPED_NAME.sub(_join, text)
    return _WRAPPED_IN_RUNNING_TEXT.sub(_join_in_running_text, text)


def _dot_initials(text: str, is_name_word) -> str:
    def glued(m: re.Match) -> str:
        if m.group(1) in ACRONYMS or (is_name_word is not None and not is_name_word(m.group(2))):
            return m.group(0)
        return "".join(f"{c}." for c in m.group(1)) + " "

    def spaced(m: re.Match) -> str:
        letters = [c for c in m.groups()[:3] if c]
        if "".join(letters) in ACRONYMS or (is_name_word is not None and not is_name_word(m.group(4))):
            return m.group(0)
        return "".join(f"{c}." for c in letters) + " "
    return _GLUED_INITIALS.sub(glued, _SPACED_INITIALS.sub(spaced, text))


# A capital set apart from the rest of its word by the PDF's letter spacing
# (2026-10-07): "V enkata", "V angipurapu", "Y eshe", "V ocabulary". Not A or
# I, which are words.
_SPLIT_CAPITAL = re.compile(r"(?<![A-Za-z.'’])([B-HJ-Z]) ([a-z]{2,})\b")


def _join_split_capitals(text: str, is_known_word) -> str:
    """Joined only when what follows the lone capital is no word itself -
    "X will", "B and", "Class V students" and "5° N latitude" stay."""
    return _SPLIT_CAPITAL.sub(lambda m: m.group(0) if is_known_word(m.group(2)) else m.group(1) + m.group(2), text)


def normalize_typeset(text: str, is_name_word=None, is_known_word=None) -> str:
    """is_name_word(word): whether a word can be a name - initials without
    dots are only dotted before one. Without it, before any capitalised word.
    is_known_word(word): whether a lowercase word is in any word or name
    list - see _join_split_capitals; without it nothing is joined."""
    if is_known_word is not None:
        text = _join_split_capitals(text, is_known_word)
    text = _dot_initials(text, is_name_word)
    text = _SPACE_BEFORE_INITIAL_DOT.sub(r"\1.", text)
    text = _TITLE_GLUED_TO_INITIAL.sub(r"\1. ", text)
    text = _INITIAL_GLUED_TO_WORD.sub(r"\1. ", text)
    text = _HYPHENATED_LINE_BREAK.sub(r"\1\2\n", text)
    return _BLANK.sub("___", text)

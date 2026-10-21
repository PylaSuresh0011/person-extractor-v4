"""
src.validation.validators.entity_type_validator
==================================================
Rejects capitalised words and phrases that are another kind of entity - an
ordinary word, a place, a building, an award, a work (film, book) or a group
of people. Added 2026-10-01 from an error analysis of a 274-page school
textbook (Class X English, Telangana SCERT), where these were most of the
false names: "Read" (54 mentions), "Story", "House", "English", "Mysore",
"Krishna Dist", "Philip's Church", "Padma Bhushan", "'Maya Bazaar'",
"the Slaters" (error brief P2 and P3).

Each rule is a cue the document itself gives; none of them names a person:

  lowercase       a single word that the document also uses in lowercase,
                  at least MIN_LOWERCASE_USES times and at least as often as
                  it is capitalised mid-sentence ("Read the passage" /
                  "read it again") - its capital comes from its position
  vocabulary      contains an exercise word (Read, List, Verb, Reader...)
                  and no other word is a known first name
  kinship         a kinship or role word on its own (Grandpa, Amma, Lord)
  contraction     a contraction (I'm, You've)
  section_header  a numbered section heading ("I. Read", "II. Adverbial
                  Clauses") at the start of a line
  place_suffix    ends in a place, building or award word ("Krishna Dist",
                  "Gandhi Nagar", "Nobel Prize")
  place_next      followed on its line by a place or building word
                  ("Ranga Reddy Dist.", "Koko Village")
  year            followed by a year in brackets, as a work is: "(1955)"
  place_context   a single word that follows a place word - in, at, from,
                  to, for, near, towards, via, reached, visited - in at
                  least half of its 2+ mentions ("in Mysore", "reached
                  Dirang", "from Mahabubnagar")
  quoted          quoted as a title, alone or inside a quoted Title-Case
                  phrase ('Maya Bazaar', 'The Dear Departed'), in at least a
                  third of its mentions - not after "called" or "known as",
                  where a quoted word is a nickname, and not in quoted speech
                  ('"Ravi!" she called' has punctuation inside the quotes)
  the             follows "the" in at least half of 3+ mentions ("the West",
                  "the Himalayas", "the Taj Mahal"; "the Wright Brothers"
                  excepted), or a plural after "the" ("the Slaters",
                  "the Pandavas")
  group           a language, nationality, religion or group word on its own
                  (English, French, Muslim, Dalit)

Round 3 (2026-10-01, error brief v2 N1-N4), each by a document-level
majority of the name's mentions where it says so:
  title_only      only titles, honorifics and roles ("Mr. President",
                  "Maharaja", "Sri", "Archduke")
  time_words      only month and season words ("Karthika", "Sharad
                  Aswayuja"), or followed by "calendar" in at least half of
                  its mentions ("the Gregorian calendar")
  org_head        ends in an organisation head-noun ("Shiv Sena", "Janata
                  Dal"), or at least half its mentions are followed by one
                  ("Meira Paibi Movement", "Mandal Commission", "Tata Motors")
  geo             most of its mentions are geographic: a geography word in
                  the next two words ("the Ravi river", "Nirmal district",
                  "Aila super-cyclone"), right before it ("river Ravi") or
                  as "tributaries / banks / valley of (the) X"; part of a
                  known place name ("Jammu and Kashmir"); or in a list that
                  is mostly known rivers and places ("the Jhelum, the
                  Chenab, the Ravi, the Beas")
  region          a compass or coast word and a name ("West Asia",
                  "Coastal Andhra"), or a name and a geography word
                  ("Andhra Region", "Peninsular Rivers")

2026-10-12 (the wrong names of three textbooks and a case folder, each
traced to the evidence that accepted it) - these look at the one mention:
  article         a single word right after "the", "a" or "an", also across
                  the break of a wrapped line ("the Guinness Book", "the
                  Kailash range", "a Rajput woman", "to the / Moon")
  place_context   also: a single word right after "in" or "at" ("In Ajmer,",
                  "in Jordon") - "to", "from" and "for" go before people too
  place_alias     in brackets right after a known place ("Cape Verde (Kabu
                  Verdi)", in a web page's list of countries)
  lowercase       also: a phrase no name list knows a word of, when every
                  word of it passes the lowercase test ("Suicidal Tendency"
                  in a paper that writes both words in lowercase)

2026-10-21 (reference lists in a case folder, rules only) - the mention:
  journal         a word of a journal's abbreviated title: it ends in a
                  period and the title runs on to the volume and pages
                  ("Bull. Am. Meteorol. Soc. 86, 225", "J. Nucl. Mater.
                  283-287, 762"), or medicine's style without periods ("N
                  Engl j med; 341: 1583") - a cited author is followed by a
                  comma or by initials instead
  publisher       after the entry's year in brackets, right after "Town(,
                  State):" or "Town, State," and running to the end of the
                  entry ("Boston: Houghton Mifflin.", "Chichester, UK, John
                  Wiley & Sons.") - the place lists miss most publishers'
                  towns (Englewood Cliffs, Chichester), so the shape decides

The word lists live in assets/entity_cues/. The lowercase, place_context and
the rules let a mention through when a title or a relation word comes just
before it ("Mr. Read", "his wife Rose") or a speech verb just after it
("Will said").
"""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator
from src.validation.document_case import lowercase_uses, mid_sentence_capitalised_uses

MIN_LOWERCASE_USES = 3
MIN_MENTIONS_FOR_THE = 3

_RELATION_BEFORE = re.compile(
    r"\b(?:wife|husband|son|daughter|brother|sister|father|mother|uncle|aunt|cousin|friend|married|dear)"
    r"[ \t]*:?[ \t]*$", re.IGNORECASE)
_SPEECH_AFTER = re.compile(
    r"^[ \t]*,?[ \t]*(?:said|says|asked|asks|replied|answered|told|shouted|cried|whispered|exclaimed)\b")
_ALIAS_BEFORE = re.compile(r"\b(?:called|known as|named|nicknamed|alias)[ \t]*$", re.IGNORECASE)
_THE_BEFORE = re.compile(r"\b[Tt]he[ \t]+$")
# An article right before the name - also across the break of a wrapped line:
# no sentence ends on "the". Not a capital "A", which is also an initial
# ("A Ramesh").
_ARTICLE_BEFORE = re.compile(r"\b(?:[Tt]he|an?)(?:[ \t]+|[ \t]*\n[ \t]*)$")
# "in" and "at" say where; "to", "from" and "for" go before people as well.
_LOCATED_BEFORE = re.compile(r"\b(?:[Ii]n|[Aa]t)(?:[ \t]+|[ \t]*\n[ \t]*)$")
# "Cape Verde (Kabu Verdi)", "Montenegro (Crna Gora)" - a place, then its other name.
_PLACE_NAMED_BEFORE = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z]+(?:[ \t][A-Z][A-Za-z]+){0,2})[ \t]?\($")
_PLACE_BEFORE = re.compile(r"\b(?:in|at|from|to|for|near|towards|via|reached|visited)[ \t]+$", re.IGNORECASE)
_YEAR_AFTER = re.compile(r"^[ \t]*\((?:1[5-9]|20)\d\d\)")
# Volume and pages, then the year in brackets or the end of the reference:
# "86, 225 (2005)", "283-287, \n762 (2000)", "145; 78-86." - not "25, 30
# people".
_VOLUME_PAGES = (r"\d{1,4}(?:[ \t]*[–-][ \t]*\d{1,6})?[ \t]*[,;][ \t]*\n?[ \t]*\d{1,6}"
                 r"(?:[ \t]*[–-][ \t]*\d{1,6})?[ \t]*\n?[ \t]*(?:\((?:1[5-9]|20)\d\d\)|[.;,]|\n|\Z)")
# Matched at the end of the word: its period, up to five more title words,
# the volume and pages ("Bull. Am. \nMeteorol. Soc. 86, 225 (2005)").
_JOURNAL_AFTER = re.compile(r"\.(?:[ \t]*\n?[ \t]*[A-Z][A-Za-z\-]*\.?){0,5}[ \t]*\n?[ \t]*" + _VOLUME_PAGES)
# Medicine's style: title words without periods, then "; volume: pages"
# ("N Engl j med; 341: 1583-1589.", "N Engl J Med. 2001;345:12-19") - not
# after initials ("Rao K. Aging. Lancet. 2001;345:12"), not a time ("10:30").
_JOURNAL_NLM_AFTER = re.compile(
    r"(?![ \t]+[A-Z]{1,3}[.,])(?:[ \t]+[A-Za-z]{1,8}\.?){1,4}[ \t]*[;.][ \t]*(?:(?:1[5-9]|20)\d\d[ \t]*;[ \t]*)?"
    r"(?!\d{1,2}[ \t]*:[ \t]*[0-5]\d(?!\d))\d{1,4}(?:\(\d+\))?[ \t]*:[ \t]*\d{1,6}"
    r"(?:[ \t]*[–-][ \t]*\d{1,6})?[ \t]*(?:[.;,]|\n|\Z)")
_YEAR_IN_BRACKETS = re.compile(r"\((?:1[5-9]|20)\d\d[a-z]?\)")
# A publisher's town right before the name: "Boston: ", "Englewood Cliffs,
# NJ: ", "Chichester, UK, ".
_PUBLISHER_TOWN_BEFORE = re.compile(
    r"(?<![A-Za-z])[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}(?:,[ \t]*[A-Z][A-Za-z.]{0,5})?[ \t]*[:,][ \t]*$")
# The name, at most two more words ("& Sons", a wrapped "Hall"), then the
# end of the entry: a period at the end of the line, or a blank line.
_ENTRY_END_AFTER = re.compile(
    r"(?:[ \t]*(?:&[ \t]*|-|[ \t]*\n[ \t]*)?[A-Z][A-Za-z]*){0,2}"
    r"(?:[ \t]*\.[ \t]*(?:\n|\Z)|[ \t]*\n[ \t]*(?:\n|\Z)|[ \t]*\Z)")
_NEXT_WORD = re.compile(r"^[ \t]+([A-Za-z]+)\.?")
_ROMAN_HEADING = re.compile(r"^[IVX]{1,4}\.$")
_INITIAL = re.compile(r"^[A-Z]\.?$")
_CONTRACTION = re.compile(r"['’](?:m|ve|re|ll)$", re.IGNORECASE)
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)
# A quoted stretch of words on one line, with no punctuation inside - a
# title ('Maya Bazaar', 'The Dear Departed'), not quoted speech.
_QUOTED_WORDS = re.compile(r"(?<![A-Za-z])['‘\"“]([A-Za-z][A-Za-z'’ \-]{0,80}?)['’\"”](?![A-Za-z])")
_TITLE_SMALL_WORDS = frozenset("a an and the of in on at to for by with from or".split())
_CAPITALISED_NEXT = re.compile(r"[ \t]+[A-Z][a-z]")
_PERSON_GROUP = re.compile(r"^(?:brothers|sisters)$", re.IGNORECASE)

ALL_CUES = frozenset({
    "lowercase", "vocabulary", "kinship", "contraction", "section_header", "place_suffix",
    "place_next", "year", "place_context", "quoted", "the", "group",
    "title_only", "time_words", "org_head", "geo", "region", "article", "place_alias",
    "journal", "publisher",
})
_COMPASS = frozenset("north south east west northern southern eastern western central upper lower coastal".split())
# Right before a name ("the river Ravi", "cyclone Aila"), or before "of
# (the)" ("tributaries of the Krishna", "the valley of Kashmir"), a word that
# makes the mention geographic. Wider windows caught people: "the collector
# of a district, Harsh Mander".
_GEO_RIGHT_BEFORE = frozenset("river rivers lake cyclone mount mt peak".split())
_GEO_OF = re.compile(r"\b(?:tributar(?:y|ies)|banks?|basin|source|valley|delta|waters|mouth)\s+of\s+(?:the\s+)?$",
                     re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+")
_TIME_AFTER = frozenset({"calendar", "calendars", "era"})
_PHRASE_BREAK = re.compile(r"[,;:()\[\]{}|–—]|\n[ \t]*\n")


def _same_phrase_words(text: str, start: int, end: int) -> list[str]:
    """The next two words, as far as they continue the phrase the mention is
    in ("Mandal Commission", "Tata Motors"). Punctuation, a blank line, or
    the end of a line the mention has to itself begins something else
    (2026-10-12): a research paper in a case folder named its authors as
    "Stuart Tooley, Queensland University of Technology" and "Fawzi Laswad"
    above a line "School of Accountancy", and both were refused as
    organisations."""
    tail = text[end:end + 40]
    cut = _PHRASE_BREAK.search(tail)
    if cut:
        tail = tail[:cut.start()]
    line_end = tail.find("\n")
    if line_end != -1 and not tail[:line_end].strip() and not text[text.rfind("\n", 0, start) + 1:start].strip():
        tail = tail[:line_end]
    return [w.lower() for w in _WORD.findall(tail)[:2]]


class EntityTypeValidator(BaseValidator):
    name = "entity_type"
    # The rules in use - all of them; experiments switch single ones off.
    cues: frozenset[str] = ALL_CUES

    def __init__(self) -> None:
        self._case_cache: dict[str, tuple[int, int]] = {}
        self._form_cache: dict[str, dict[str, int]] = {}

    def reset_document_cache(self) -> None:
        self._case_cache.clear()
        self._form_cache.clear()

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        if not tokens:
            return self._pass()
        start, end = candidate.candidate.start, candidate.candidate.end
        before = document_text[max(0, start - 40):start]
        after = document_text[end:end + 40]
        single = len(tokens) == 1
        words = [t.rstrip(".").lower() for t in tokens]
        cues = self.cues
        if candidate.candidate.normalized_text.lower() in knowledge_base.confirmed_names:
            return self._pass()  # an owner decision ("Siri") - never another entity type

        if "contraction" in cues and any(_CONTRACTION.search(t) for t in tokens):
            return self._reject("contraction", f"'{candidate.candidate.normalized_text}' is a contraction, not a name")
        if "kinship" in cues and single and words[0] in knowledge_base.kinship_role_words:
            return self._reject("kinship", f"'{tokens[0]}' on its own is a kinship or role word")
        if "group" in cues and single and words[0] in knowledge_base.group_words:
            return self._reject("group", f"'{tokens[0]}' on its own is a language, nationality, religion or group")
        if "vocabulary" in cues and any(w in knowledge_base.textbook_vocabulary for w in words) and not any(
                knowledge_base.is_known_first_name(t) and not knowledge_base.is_ambiguous_first_name(t)
                for t, w in zip(tokens, words) if w not in knowledge_base.textbook_vocabulary):
            return self._reject("vocabulary", "contains an exercise or textbook word and no first name")
        if ("section_header" in cues and len(tokens) > 1 and _ROMAN_HEADING.match(tokens[0])
                and not _INITIAL.match(tokens[1]) and _starts_line(document_text, start)):
            return self._reject("section_header", "a numbered section heading, not a name")
        if "place_suffix" in cues:
            cue_words = knowledge_base.place_building_words | knowledge_base.award_words
            if len(tokens) > 1 and words[-1] in cue_words and not _surname_after_first_name(
                    tokens, len(tokens) - 1, knowledge_base, before):
                return self._reject("place_suffix", f"ends in the place, building or award word '{tokens[-1]}'")
        if "place_next" in cues:
            nxt = _NEXT_WORD.match(after)
            if nxt and nxt.group(1).lower() in knowledge_base.place_building_words and not _surname_after_first_name(
                    tokens + [nxt.group(1)], len(tokens), knowledge_base, before):
                return self._reject("place_next", f"followed by the place or building word '{nxt.group(1)}'")
        if "year" in cues and _YEAR_AFTER.match(after):
            return self._reject("year", "followed by a year in brackets, as a work title is")
        if "journal" in cues and _journal_title_word(tokens, document_text, end):
            return self._reject("journal", f"'{tokens[-1]}' is a word of a journal's abbreviated title, "
                                           "before the volume and pages")
        if "publisher" in cues and _publisher(document_text, start, end):
            return self._reject("publisher", "a publisher - after the town, at the end of a reference")
        if "title_only" in cues and all(
                w in knowledge_base.title_role_words or knowledge_base.is_title(w) or knowledge_base.is_honorific(w)
                or _ROMAN_OR_LETTER.match(t) for t, w in zip(tokens, words)):
            return self._reject("title_only", f"'{candidate.candidate.normalized_text}' is only titles or roles")
        if "time_words" in cues and all(w in knowledge_base.months_and_seasons for w in words):
            return self._reject("time_words", "only month and season words")
        if "org_head" in cues and len(tokens) > 1 and words[-1] in knowledge_base.org_head_nouns:
            return self._reject("org_head", f"ends in the organisation word '{tokens[-1]}'")
        if "region" in cues and len(tokens) > 1 and not any(
                _ROMAN_OR_LETTER.match(t) or _JOINED_INITIALS.match(t) for t in tokens) and (
                words[0] in _COMPASS or words[-1] in knowledge_base.geo_context_words) and not (
                _surname_after_first_name(tokens, len(tokens) - 1, knowledge_base, before)):
            return self._reject("region", "a compass or coast word with a name, or a name with a geography word")
        if {"org_head", "geo", "time_words"} & cues:
            context = self._context_uses(tokens, document_text, knowledge_base)
            if "time_words" in cues and context["time"] and 2 * context["time"] >= context["n"]:
                return self._reject("time_words", f"followed by 'calendar' or 'era' in {context['time']} of its "
                                                  f"{context['n']} mentions")
            if "org_head" in cues and context["org"] and 2 * context["org"] >= context["n"]:
                return self._reject("org_head", f"followed by an organisation word in {context['org']} of its "
                                                f"{context['n']} mentions")
            # Initials or a title before a name make it a person, whatever
            # the rest matches ("K.V. Ranga Reddy" - Ranga Reddy is a district).
            # More than half, not half (measured again 2026-10-12): the river
            # Krishna is geographic in 5 of its 10 mentions in one textbook -
            # and in another, "Krishna" is a district twice and the god twice.
            if ("geo" in cues and not any(knowledge_base.is_title(w) or knowledge_base.is_honorific(w) for w in words)
                    and not any(_ROMAN_OR_LETTER.match(t) or _JOINED_INITIALS.match(t) for t in tokens)
                    and 2 * context["geo"] > context["n"]):
                return self._reject("geo", f"geographic in {context['geo']} of its {context['n']} mentions")

        strong_context = _strong_name_context(before, after, knowledge_base)
        if "lowercase" in cues and single and not strong_context:
            lowercase, mid_sentence = self._case_uses(tokens[0], document_text)
            if lowercase >= MIN_LOWERCASE_USES and lowercase >= mid_sentence:
                return self._reject(
                    "lowercase", f"'{tokens[0]}' is used {lowercase}x in lowercase in this document "
                                 f"(capitalised mid-sentence {mid_sentence}x) - an ordinary word")
        # The same test for a phrase no list knows a word of (2026-10-12):
        # when the document writes every word of it in lowercase too, the
        # capitals are a heading's - a research paper's "Suicidal Tendency"
        # was accepted 12 times in a case; neither word is on the
        # ordinary-word list. A name word in it ("Mark Brown") exempts it.
        if "lowercase" in cues and not single and not strong_context and not any(
                knowledge_base.is_known_first_name(w) or knowledge_base.is_known_last_name(w) for w in words):
            uses = [self._case_uses(t.rstrip("."), document_text) for t in tokens]
            if all(lowercase >= MIN_LOWERCASE_USES and lowercase >= mid_sentence for lowercase, mid_sentence in uses):
                return self._reject("lowercase", "every word of it is also written in lowercase in this document "
                                                 "- ordinary words in a heading's capitals")
        if {"quoted", "place_context", "the"} & cues:
            form = self._form_uses(tokens, document_text)
            if "quoted" in cues and form["quoted"] and 3 * form["quoted"] >= form["n"]:
                return self._reject("quoted", f"quoted as a title in {form['quoted']} of its {form['n']} mentions")
            if ("place_context" in cues and single and not strong_context and form["n"] >= 2
                    and 2 * form["place"] >= form["n"]):
                return self._reject("place_context", f"follows a place word (in, at, from, to...) in {form['place']} "
                                                     f"of its {form['n']} mentions")
            if "the" in cues and not strong_context and not _PERSON_GROUP.match(tokens[-1]):
                if form["n"] >= MIN_MENTIONS_FOR_THE and 2 * form["the"] >= form["n"]:
                    return self._reject("the", f"follows 'the' in {form['the']} of its {form['n']} mentions")
                if single and _THE_BEFORE.search(before) and _is_plural(tokens[0]):
                    return self._reject("the", f"'the {tokens[0]}' - a plural, a family or group")
        # The mention itself (2026-10-12) - see the module docstring.
        if "article" in cues and single and not strong_context and _ARTICLE_BEFORE.search(before):
            return self._reject("article", f"'{tokens[0]}' follows an article here - a thing or a group, not a person")
        if "place_context" in cues and single and not strong_context and _LOCATED_BEFORE.search(before) and (
                after[:2] not in ("'s", "’s")):
            return self._reject("place_context", f"'{tokens[0]}' follows 'in' or 'at' here - where, not who")
        if "place_alias" in cues:
            named = _PLACE_NAMED_BEFORE.search(before)
            if named and after.startswith(")") and knowledge_base.is_location(named.group(1)):
                return self._reject("place_alias", f"another name of '{named.group(1)}', in brackets after it")
        return self._pass()

    def _case_uses(self, word: str, text: str) -> tuple[int, int]:
        """(lowercase uses, capitalised uses mid-sentence) - see src/validation/document_case.py."""
        if word not in self._case_cache:
            self._case_cache[word] = (lowercase_uses(word, text), mid_sentence_capitalised_uses(word, text))
        return self._case_cache[word]

    def _form_uses(self, tokens: list[str], text: str) -> dict[str, int]:
        """How often this exact form occurs, and how often quoted as a title,
        after 'the' and after a place word."""
        form = " ".join(tokens)
        if form not in self._form_cache:
            counts = {"n": 0, "quoted": 0, "the": 0, "place": 0}
            pattern = r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, tokens)) + r"(?![A-Za-z])"
            for m in re.finditer(pattern, text):
                s, e = m.start(), m.end()
                if _CAPITALISED_NEXT.match(text, e):
                    continue  # part of a longer name: "Maya Bazaar" is not a mention of "Maya"
                before = text[max(0, s - 40):s]
                counts["n"] += 1
                if _in_quoted_title(text, s, e):
                    counts["quoted"] += 1
                if _THE_BEFORE.search(before):
                    counts["the"] += 1
                if _PLACE_BEFORE.search(before) and text[e:e + 2] not in ("'s", "’s"):
                    counts["place"] += 1
            self._form_cache[form] = counts
        return self._form_cache[form]

    def _context_uses(self, tokens: list[str], text: str, knowledge_base: KnowledgeBase) -> dict[str, int]:
        """Every occurrence of this form - also inside a longer capitalised
        run ("Meira Paibi Movement", "Godavari River") - and how many have an
        organisation head-noun in the next two words, or a geographic word
        or list around them."""
        key = "ctx " + " ".join(tokens)
        if key not in self._form_cache:
            counts = {"n": 0, "org": 0, "geo": 0, "time": 0}
            pattern = r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, tokens)) + r"(?![A-Za-z])"
            lists = None
            for m in re.finditer(pattern, text):
                s, e = m.start(), m.end()
                after = [w.lower() for w in _WORD.findall(text[e:e + 40])[:2]]
                before_text = text[max(0, s - 40):s]
                before = [w.lower() for w in _WORD.findall(before_text)]
                possessive = text[e:e + 2] in ("'s", "’s")
                counts["n"] += 1
                if any(w in knowledge_base.org_head_nouns for w in _same_phrase_words(text, s, e)):
                    counts["org"] += 1
                elif after[:1] and after[0] in _TIME_AFTER:
                    counts["time"] += 1
                elif (not possessive and any(w in knowledge_base.geo_context_words for w in after)) or (
                        before[-1:] and before[-1] in _GEO_RIGHT_BEFORE) or _GEO_OF.search(before_text) or (
                        _in_known_place(text, s, e, knowledge_base)):
                    counts["geo"] += 1
                else:
                    if lists is None:
                        if "__place_lists__" not in self._form_cache:
                            self._form_cache["__place_lists__"] = _place_lists(text, knowledge_base)
                        lists = self._form_cache["__place_lists__"]
                    if any(a <= s and e <= b for a, b in lists):
                        counts["geo"] += 1
            self._form_cache[key] = counts
        return self._form_cache[key]

    def _reject(self, cue: str, message: str) -> ValidationResult:
        return ValidationResult(self.name, passed=False, severity="hard", message=f"{cue}: {message}")

    def _pass(self) -> ValidationResult:
        return ValidationResult(self.name, passed=True, score=0.0, message="No other entity type indicated")


def _journal_title_word(tokens: list[str], text: str, end: int) -> bool:
    """The last word is a journal title's (see _JOURNAL_AFTER) and any words
    before it are initials ("J. Nucl. Mater. 283"). Not an initial or an
    all-capitals group itself ("Lawrence MG. Bull Am Meteorol Soc.")."""
    last = tokens[-1].rstrip(".")
    if len(last) < 2 or last.isupper() or not all(_INITIAL.match(t) for t in tokens[:-1]):
        return False
    pos = end - 1 if text[end - 1:end] == "." else end
    return bool(_JOURNAL_AFTER.match(text, pos) or _JOURNAL_NLM_AFTER.match(text, end))


def _publisher(text: str, start: int, end: int) -> bool:
    """After the year of a reference entry and a town, running to the end of
    the entry - see _PUBLISHER_TOWN_BEFORE and _ENTRY_END_AFTER."""
    line_start = text.rfind("\n", 0, start) + 1
    before = text[max(line_start, start - 200):start]
    town = _PUBLISHER_TOWN_BEFORE.search(before)
    return bool(town and _YEAR_IN_BRACKETS.search(before, 0, town.start()) and _ENTRY_END_AFTER.match(text, end))


def _in_quoted_title(text: str, start: int, end: int) -> bool:
    """Inside quotes that hold only Title-Case words ('Maya Bazaar', 'Or
    Will the Dreamer Wake') - and not just after "called" or "known as"."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end != -1 else len(text)]
    for q in _QUOTED_WORDS.finditer(line):
        inner_start, inner_end = line_start + q.start(1), line_start + q.end(1)
        if not (inner_start <= start and end <= inner_end):
            continue
        words = q.group(1).split()
        return (bool(words) and words[0][0].isupper()
                and all(w[0].isupper() or w.lower() in _TITLE_SMALL_WORDS for w in words)
                and not _ALIAS_BEFORE.search(text[max(0, inner_start - 41):inner_start - 1]))
    return False


_JOINED_INITIALS = re.compile(r"^(?:[A-Z]\.){2,4}$")
_ROMAN_OR_LETTER = re.compile(r"^(?:[IVX]{1,4}|[A-Z])\.?$")


def _in_known_place(text: str, start: int, end: int, knowledge_base: KnowledgeBase) -> bool:
    """Part of a known place name with the words around it: "Jammu and
    [Kashmir]", "West [Bengal]", "[Andhra] Pradesh"."""
    before = re.findall(r"[A-Za-z]+", text[max(0, start - 30):start])[-3:]
    after = re.findall(r"[A-Za-z]+", text[end:end + 30])[:2]
    name = text[start:end]
    for i in range(len(before) + 1):
        for j in range(len(after) + 1):
            if (i or j) and knowledge_base.is_location(" ".join(before[len(before) - i:] + [name] + after[:j])):
                return True
    return False


def _place_lists(text: str, knowledge_base: KnowledgeBase) -> list[tuple[int, int]]:
    """Spans of coordinated lists most of whose items are known rivers,
    regions or places ("the Jhelum, the Chenab, the Ravi, the Beas and the
    Sutlej") - every item of such a list is a place."""
    from src.candidate.list_propagation import find_lists  # avoids an import cycle
    spans = []
    for items in find_lists(re.sub(r"\b[Tt]he ", "    ", text)):
        known = sum(i.text.lower() in knowledge_base.rivers_and_regions or knowledge_base.is_location(i.text)
                    for i in items)
        if known >= 2 and 2 * known >= len(items):
            spans.append((items[0].start, items[-1].end))
    return spans


def _starts_line(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    return not text[line_start:start].strip()


def _strong_name_context(before: str, after: str, knowledge_base: KnowledgeBase) -> bool:
    """A title or relation word just before ("Mr. Read", "his wife Rose",
    "Dear Will") or a speech verb just after ("Will said")."""
    prev = re.search(r"([A-Za-z]+)\.?[ \t]*$", before)
    if prev and (knowledge_base.is_title(prev.group(1)) or knowledge_base.is_honorific(prev.group(1))):
        return True
    return bool(_RELATION_BEFORE.search(before) or _SPEECH_AFTER.match(after))


def _surname_after_first_name(tokens: list[str], cue_index: int, knowledge_base: KnowledgeBase, before: str) -> bool:
    """A cue word that is also a surname, after a known first name
    ("Charlotte Church", "David Villa") - a person, unless the phrase is
    possessive ("Philip's Church") or follows "the" ("the Taj Mahal")."""
    if cue_index == 0 or not knowledge_base.is_known_last_name(tokens[cue_index].rstrip(".")):
        return False
    prev = tokens[cue_index - 1]
    if _POSSESSIVE.search(prev) or _THE_BEFORE.search(before):
        return False
    return knowledge_base.is_known_first_name(prev)


def _is_plural(word: str) -> bool:
    return len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is"))

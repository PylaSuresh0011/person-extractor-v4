"""The rules fixed on 2026-10-07, after a 194-page school textbook (Class IX
English, Telangana SCERT) where the extractor found 106 of 123 people. Each
missed or cut-off name was traced to the rule that dropped it; these tests
hold the rule changes, not the names - none of them was added to a list.

  repetition    "V. V. Subrahmanyam" rejected: the initial repeats
  split         "Mohammed Quli Qutub Shah", "F. W. de Klerk", "Martin Luther
                King Jr." cut into two halves
  structure     "F. W. de Klerk": a lowercase particle inside a name
  span choice   "Pydimarri Venkata Subba Rao" -> "Venkata", "Kurra Suresh
                Babu" -> "Suresh Babu": only the listed part of a name kept
  trim          "Chief Seattle" -> "Chief": nothing of the name left
  corroboration "William Shakespeare", "Leo Tolstoy": a first name that is
                also an ordinary word, a surname no list holds
  location      "Virginia Woolf": the first name is also a place
  typeset       names cut by a line wrap or by a spaced-out capital
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.boundary_refiner import split_double_name_candidates, trim_trailing_designations
from src.candidate.candidate_factory import _select_canonical_span
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.line_indexer import LineIndex
from src.preprocessing.typeset import join_wrapped_names, normalize_typeset
from src.validation.validators.corroboration_validator import CorroborationValidator
from src.validation.validators.location_validator import LocationValidator
from src.validation.validators.repetition_validator import RepetitionValidator
from src.validation.validators.structure_validator import StructureValidator

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _candidate(doc: str, name: str, patterns: tuple[str, ...] = ("bare",), dictionary_word: str | None = None) -> CandidateResult:
    """A candidate for `name` where it stands in `doc`, with one detection
    per pattern ("bare", "spacy_ner", "dictionary_single_token" - the last
    on dictionary_word, the name's first word by default)."""
    start = doc.index(name)
    end = start + len(name)
    detections = []
    for pattern in patterns:
        if pattern == "dictionary_single_token":
            word = dictionary_word or name.split()[0]
            w_start = start + name.index(word)
            detections.append(Detection(word, w_start, w_start + len(word), 0, DetectorName.DICTIONARY, 0.4,
                                        {"pattern": pattern}))
        else:
            detector = DetectorName.SPACY if pattern == "spacy_ner" else DetectorName.REGEX
            detections.append(Detection(name, start, end, 0, detector, 0.55, {"pattern": pattern}))
    return CandidateResult(candidate=Candidate.new(name, name, start, end, 0, tuple(detections)))


# --- repetition -------------------------------------------------------------

def test_repeated_initials_are_not_a_repeated_word():
    doc = "V.V.S. Laxman tells V. V. Subrahmanyam."
    assert RepetitionValidator().validate(_candidate(doc, "V. V. Subrahmanyam"), KB, doc).passed is True


# --- split of a four-word span ----------------------------------------------

def _split(doc: str, name: str) -> list[str]:
    parts = split_double_name_candidates([_candidate(doc, name)], KB, LineIndex.build(doc), doc)
    return [p.candidate.text for p in parts]


def test_a_long_name_in_a_sentence_stays_one_name():
    doc = "That is how the city was named Hyderabad. Mohammed Quli Qutub Shah of the Shahi Dynasty named it."
    assert _split(doc, "Mohammed Quli Qutub Shah") == ["Mohammed Quli Qutub Shah"]
    doc = "He and F. W. de Klerk shared the Nobel Peace Prize."
    assert _split(doc, "F. W. de Klerk") == ["F. W. de Klerk"]
    doc = "Let us read the speech by Martin Luther King Jr. once again."
    assert _split(doc, "Martin Luther King Jr") == ["Martin Luther King Jr"]


def test_one_word_on_both_name_lists_does_not_make_a_pair():
    # "Martin" is a first name and a surname; "Luther" is on neither list.
    doc = "- Martin Luther King Jr"
    assert _split(doc, "Martin Luther King Jr") == ["Martin Luther King Jr"]


def test_two_names_glued_on_a_line_of_fields_still_split():
    doc = "14045551193 Suresh Kumar Felix Amaro 14045551190"
    assert _split(doc, "Suresh Kumar Felix Amaro") == ["Suresh Kumar", "Felix Amaro"]


def test_two_confirmed_names_split_inside_a_sentence_too():
    doc = "The letter was signed by Suresh Kumar Ramesh Reddy yesterday."
    assert _split(doc, "Suresh Kumar Ramesh Reddy") == ["Suresh Kumar", "Ramesh Reddy"]


# --- a lowercase particle inside a name -------------------------------------

def test_lowercase_particle_inside_a_name_is_a_valid_shape():
    def passed(name: str) -> bool:
        return StructureValidator().validate(_candidate(name, name), KB, name).passed
    assert passed("F. W. de Klerk") and passed("Ludwig van Beethoven")
    assert not passed("de Klerk")      # a particle first: not a complete name
    assert not passed("Ludwig van")    # a particle last
    assert not passed("Ludwig the Beethoven")   # any other lowercase word


# --- the listed part of a name must not become the whole name ----------------

def _cluster(text: str, hit: str, pattern: str) -> list[Detection]:
    start = text.index(hit)
    return [Detection(hit, start, start + len(hit), 0, DetectorName.DICTIONARY, 0.6, {"pattern": pattern}),
            Detection(text, 0, len(text), 0, DetectorName.REGEX, 0.55, {"pattern": "bare"})]


def test_a_dictionary_hit_widens_to_the_whole_capitalised_name():
    chosen = lambda text, hit, pattern: _select_canonical_span(_cluster(text, hit, pattern), KB).text  # noqa: E731
    assert chosen("Pydimarri Venkata Subba Rao", "Venkata", "dictionary_single_token") == "Pydimarri Venkata Subba Rao"
    assert chosen("Kurra Suresh Babu", "Suresh Babu", "dictionary") == "Kurra Suresh Babu"
    assert chosen("Vaikom Muhammad Basheer", "Muhammad Basheer", "dictionary") == "Vaikom Muhammad Basheer"
    assert chosen("P.V. Narsimha Rao", "Rao", "dictionary_single_token") == "P.V. Narsimha Rao"


def test_no_widening_onto_sentence_words_or_from_a_calendar_word():
    chosen = lambda text, hit, pattern: _select_canonical_span(_cluster(text, hit, pattern), KB).text  # noqa: E731
    assert chosen("Marcus Webb I'll", "Marcus Webb", "dictionary") == "Marcus Webb"      # a contraction
    assert chosen("Suresh Babu Report", "Suresh Babu", "dictionary") == "Suresh Babu"   # an ordinary word
    assert chosen("Bharatiya Jan Sangh", "Jan", "dictionary_single_token") == "Jan"     # "Jan" alone is refused later


def test_byline_gives_the_whole_name(pipeline):
    pages = ["Jaya jaya jaya, jaya he!\nPLEDGE\n- Pydimarri Venkata Subba Rao\nIndia is my country. All Indians are "
             "my brothers and sisters.\n" + FOOTER, FILLER + FOOTER, FILLER + FOOTER]
    assert "Pydimarri Venkata Subba Rao" in _names(pipeline.run_pages(pages))


# --- trimming a place off the end must leave a name --------------------------

def _trimmed(doc: str, name: str) -> str:
    out = trim_trailing_designations([_candidate(doc, name)], doc, LineIndex.build(doc), KB)
    return out[0].candidate.text


def test_trailing_place_is_trimmed_only_when_a_name_word_is_left():
    assert _trimmed("Chief Seattle became world famous for a moving speech.", "Chief Seattle") == "Chief Seattle"
    assert _trimmed("Reviewed by Rao Yerdoor Hyderabad last year.", "Rao Yerdoor Hyderabad") == "Rao Yerdoor"


# --- a first name that is also an ordinary word ------------------------------

def _corroborated(doc: str, name: str, patterns: tuple[str, ...]) -> bool:
    return CorroborationValidator().validate(_candidate(doc, name, patterns), KB, doc).passed


def test_surname_the_document_only_ever_capitalises_is_no_ordinary_word():
    """The frequency list of ordinary words holds "shakespeare"; a document
    that writes it twice, never in lowercase, uses it as a name."""
    doc = "- William Shakespeare\nAbout the poet\nWilliam Shakespeare (1564 - 1616) was an English poet and playwright."
    assert _corroborated(doc, "William Shakespeare", ("bare", "dictionary_single_token", "spacy_ner"))


def test_surname_the_document_also_writes_in_lowercase_stays_an_ordinary_word():
    doc = "Stored in Ruby Chamber on Monday. The chamber was locked. Ruby Chamber is cold."
    assert not _corroborated(doc, "Ruby Chamber", ("bare", "dictionary_single_token", "spacy_ner"))


def test_first_name_and_unlisted_surname_inside_a_sentence_need_no_spacy_tag():
    doc = "We see the cover of the book: The Kingdom of God is Within You, by Leo Tolstoy."
    assert _corroborated(doc, "Leo Tolstoy", ("bare", "dictionary_single_token"))


def test_the_same_pair_as_a_record_field_or_after_a_nationality_is_refused():
    doc = "Ref: Leo Tolstoy | Code: 4319 | Status: archived"
    assert not _corroborated(doc, "Leo Tolstoy", ("bare", "dictionary_single_token"))
    doc = "This led to the rise of German Fascism in the years after the war."
    assert not _corroborated(doc, "German Fascism", ("bare", "dictionary_single_token"))


# --- a first name that is also a place ---------------------------------------

def _not_a_location(doc: str, name: str, patterns: tuple[str, ...]) -> bool:
    return LocationValidator().validate(_candidate(doc, name, patterns), KB, doc).passed


def test_place_word_as_a_first_name_before_an_unlisted_surname():
    doc = "There is no gate, no lock, no bolt that you can set upon the freedom of my mind. - Virginia Woolf, 1929"
    assert _not_a_location(doc, "Virginia Woolf", ("bare", "dictionary_single_token", "spacy_ner"))
    assert not _not_a_location(doc, "Virginia Woolf", ("bare", "dictionary_single_token"))   # no spaCy tag
    doc = "They drove to Virginia Beach for the weekend."
    assert not _not_a_location(doc, "Virginia Beach", ("bare", "dictionary_single_token", "spacy_ner"))


# --- typeset text -------------------------------------------------------------

PROSE = ("V.V.S. Laxman: It has been wonderful. I was lucky that the\n"
         "Hyderabad Cricket Association promoted me and Zaki (Syed\n"
         "Moinuddin Zaki) from the under-16 level itself. They made\n"
         "us play in the under-19 grade the same year. It was a huge\n")
VERSE = ("He was my friend, faithful and just to me:\n"
         "So let it be with Caesar. The noble Brutus\n"
         "Hath told you Caesar was ambitious:\n"
         "If it were so, it was a grievous fault,\n")


def test_name_cut_by_the_wrap_of_a_paragraph_line_is_joined():
    joined = join_wrapped_names(PROSE, KB.is_common_word, KB.is_location)
    assert "(Syed Moinuddin Zaki) from" in joined and len(joined) == len(PROSE)
    text = ("lies about ten miles away from Golconda in Hyderabad. Mohammad\n"
            "Quli Qutub Shah, the fifth Sultan of the dynasty, fell in love with\nher and built a city.\n")
    assert "Mohammad Quli Qutub Shah, the" in join_wrapped_names(text, KB.is_common_word, KB.is_location)


def test_no_join_in_verse_after_a_possessive_or_before_a_place():
    assert join_wrapped_names(VERSE, KB.is_common_word, KB.is_location) == VERSE   # every line starts with a capital
    text = ("the classic of Telugu life thereafter. Sasirekha's, nay Ghatothkacha's\n"
            "Manadi Sodara Prema became immortal in the minds of the people who\nsaw the film in those years.\n")
    assert join_wrapped_names(text, KB.is_common_word, KB.is_location) == text
    text = ("members of the group were a practitioner from Bhopal and Dr. Homen Thangjam\n"
            "Manipur University, who read the chapters and gave many comments on\nthe draft of the book.\n")
    assert "Thangjam\nManipur" in join_wrapped_names(text, KB.is_common_word, KB.is_location)


def test_capital_spaced_off_its_word_is_joined_unless_a_word_follows():
    known = lambda w: KB.is_common_word(w) or KB.is_stopword(w) or KB.is_known_first_name(w) or KB.is_known_last_name(w)  # noqa: E731
    fixed = normalize_typeset("V angipurapu V enkata Sai Laxman announced his retirement.", is_known_word=known)
    assert fixed == "Vangipurapu Venkata Sai Laxman announced his retirement."
    for text in ("Students of Class X will read it.", "Options A and B are correct.", "It lies at 17 N latitude."):
        assert normalize_typeset(text, is_known_word=known) == text
    assert normalize_typeset("V enkata came.") == "V enkata came."   # no word list given: nothing is joined


# --- whole pipeline -----------------------------------------------------------

FOOTER = "English Class IX Free distribution by Government of Telangana 2024-25"
FILLER = "The students read the lesson carefully and wrote answers in their notebooks.\n"


@pytest.fixture(scope="module")
def pipeline():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["enabled"] = False
    return Pipeline(config=cfg, base_dir=BASE_DIR)


def _names(result) -> set[str]:
    return {p.display_text for p in result.persons} | {p.display_text for p in result.review_persons}


def test_long_name_in_a_sentence_is_one_person(pipeline):
    pages = [FILLER + "After they got married the city was named Hyderabad. Mohammed Quli Qutub Shah of the Shahi "
             "Dynasty named it Hyderabad.\n" + FOOTER, FILLER + FOOTER, FILLER + FOOTER]
    names = _names(pipeline.run_pages(pages))
    assert "Mohammed Quli Qutub Shah" in names and "Qutub Shah" not in names and "Mohammed Quli" not in names

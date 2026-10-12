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

2026-10-12: the case folder scanned again with these rules, against the 365
names read from it by hand. Names it had found on 2026-10-01 and now lost:

  span choice   "LtCol RK Verma": widened onto a word the structure check
                refuses, and refused whole
  corroboration "William H. Gates III": a suffix hid the middle initial
  entity type   "Stuart Tooley, Queensland University": an organisation
                after a comma counted as part of the name
  location      "Chanda Nagar, India" accepted: a town before its country
  score         "Watson", "Wright": listed surnames scored as ordinary words;
                "Siri", a name by an owner decision, dropped by its score
  citations     "Abdolmohammadi, M; Harris, J and Smith, K (2002)", "Chi et
                al.": the authors a research paper cites - most of them never
                a candidate
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.boundary_refiner import split_double_name_candidates, trim_trailing_designations
from src.candidate.candidate_factory import _select_canonical_span, is_middle_initial_name
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.decision.feature_score import person_features
from src.detection.regex_detector import CITATION_CUE, RegexDetector
from src.knowledge.knowledge_base import KnowledgeBase
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.line_indexer import LineIndex
from src.preprocessing.typeset import join_wrapped_names, normalize_typeset
from src.validation.validators.corroboration_validator import CorroborationValidator
from src.validation.validators.entity_type_validator import EntityTypeValidator
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


def test_no_widening_onto_a_word_the_structure_check_refuses():
    """ "I am LtCol RK Verma from the lab": the widened span was refused
    whole for "LtCol", and the person with it."""
    chosen = lambda text, hit, pattern: _select_canonical_span(_cluster(text, hit, pattern), KB).text  # noqa: E731
    assert chosen("LtCol RK Verma", "Verma", "dictionary_single_token") == "Verma"
    assert chosen("NDTV Suresh Babu", "Suresh Babu", "dictionary") == "Suresh Babu"     # an acronym


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


def test_generational_suffix_after_a_middle_initial_name():
    assert is_middle_initial_name("William H. Gates III") and is_middle_initial_name("John F. Kennedy Jr.")
    assert not is_middle_initial_name("William Gates III")   # no initial
    doc = "Signature |  | Title\n\n/s/ William H. Gates III |\nWilliam H. Gates III |  | Chairman"
    assert _corroborated(doc, "William H. Gates III", ("bare", "dictionary_single_token", "spacy_ner"))


# --- an organisation after the name -------------------------------------------

def _entity_type(doc: str, name: str):
    return EntityTypeValidator().validate(_candidate(doc, name, ("bare", "spacy_ner")), KB, doc)


def test_organisation_after_a_comma_or_on_the_next_line_is_not_part_of_the_name():
    doc = ("Stuart Tooley, Queensland University of Technology, Brisbane, Australia\n\n"
           "Corresponding author:\n\nFawzi Laswad\nUniversity of Otago\nPrivate Bag 11 222\n")
    assert _entity_type(doc, "Stuart Tooley").passed
    assert _entity_type(doc, "Fawzi Laswad").passed


def test_organisation_word_in_the_same_phrase_still_makes_an_organisation():
    doc = ("They waited for the Mandal\nCommission to finish its work. A year later the Mandal Commission "
           "gave its report to the government.")
    result = _entity_type(doc, "Mandal")
    assert not result.passed and "org_head" in result.message


# --- a first name that is also a place ---------------------------------------

def _not_a_location(doc: str, name: str, patterns: tuple[str, ...]) -> bool:
    return LocationValidator().validate(_candidate(doc, name, patterns), KB, doc).passed


def test_place_word_as_a_first_name_before_an_unlisted_surname():
    doc = "There is no gate, no lock, no bolt that you can set upon the freedom of my mind. - Virginia Woolf, 1929"
    assert _not_a_location(doc, "Virginia Woolf", ("bare", "dictionary_single_token", "spacy_ner"))
    assert not _not_a_location(doc, "Virginia Woolf", ("bare", "dictionary_single_token"))   # no spaCy tag
    doc = "They drove to Virginia Beach for the weekend."
    assert not _not_a_location(doc, "Virginia Beach", ("bare", "dictionary_single_token", "spacy_ner"))


def test_place_word_first_name_stays_a_place_before_its_country():
    doc = "The parcel was sent to the branch office at Chanda Nagar, India last week."
    assert not _not_a_location(doc, "Chanda Nagar", ("bare", "dictionary_single_token", "spacy_ner"))


# --- the final score: a listed surname is not an ordinary word ----------------

def _scored_as_ordinary_word(doc: str, name: str) -> bool:
    return person_features([_candidate(doc, name, ("bare", "dictionary_single_token"))], doc, KB)["ordinary_word"]


def test_listed_surname_capitalised_inside_a_sentence_is_no_ordinary_word():
    """The frequency list of ordinary words holds "watson" and "wright"."""
    assert not _scored_as_ordinary_word("Similar results were reported by Hodge and Watson in the same year.", "Watson")
    assert _scored_as_ordinary_word("Watson\nBlock\nRead\n", "Watson")                 # labels: no sentence
    assert _scored_as_ordinary_word("It was signed by Watson. The watson unit was renamed.", "Watson")   # lowercase too
    assert _scored_as_ordinary_word("The reply came from the Parliament in the same year.", "Parliament")   # no surname


def test_owner_confirmed_name_takes_no_negative_feature():
    """ "Siri" is a name by an owner decision (confirmed_names.txt); the
    voice-command file that holds it also writes "siri call"."""
    doc = "Siri call $CONTACTS\nsiri call $CONTACTS\nsiri text $CONTACTS\nsiri open $APP"
    features = person_features([_candidate(doc, "Siri", ("dictionary_single_token",))], doc, KB)
    assert not features["lowercase_use"] and not features["ordinary_word"]


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


# --- cited authors --------------------------------------------------------------

REFERENCES = (
    "Abdolmohammadi, M; Harris, J and Smith, K (2002), Government financial reporting on the internet.\n\n"
    "Ashbaugh, H; Johnstone, K.M and Warfield, T.D (1999), Corporate reporting on the internet.\n\n"
    "Brown, D; Beach, L.R and Stone, D.N (1991), Human judgment accuracy.\n\n"
    "Fergusson, D.M., Woodward, L.J. & Horwood, L.J. (1997). Early onset cannabis use.\n\n"
    "Lipe, M.G and Salterio, S.E (1 999), The balance scorecard.\n\n"                   # the PDF spaced a digit off
    "Fazel, S., Cartwright, J., Norman-Nott, A.& Hawton, K. Suicide in prisoners: a systematic review.\n\n"
    "Innamorati, M., Pompili, V., Ferrari et al., (2008 ) Cannabis use and the risk behaviour syndrome.\n\n"
    "Rohrman, B. 1986. Evaluating the usefulness of decision aids.\n\n"
    "Vera-Munoz, S.C; Kinney Jr, W.R and Bonner, S.E (2001), The effects of domain experience.\n\n"
    "Vessey, I (1991), Cognitive fit. Englewood Cliffs, N.J.: Prentice-Hall\n"
)


def _cited(doc: str) -> list[str]:
    return [d.text for d in RegexDetector().detect(doc, 0) if d.metadata.get("cue") == CITATION_CUE]


def test_authors_of_a_reference_list_are_cited_people():
    cited = _cited(REFERENCES)
    for name in ("Abdolmohammadi", "Harris", "Smith", "Warfield", "Beach", "Horwood", "Salterio", "Norman-Nott",
                 "Hawton", "Pompili", "Ferrari", "Rohrman", "Vera-Munoz", "Kinney Jr", "Vessey"):
        assert name in cited, name
    assert "Cliffs" not in cited      # the publisher's town: no year closes it
    # An organisation cited by its acronym is no author with initials.
    assert "Programme" not in _cited(REFERENCES + "as reported (Arrestee Drug Abuse Monitoring Programme, ADAM, 2000).")


def test_name_before_et_al_is_a_cited_person():
    assert _cited("as others found (Ashbaugh et al., 1999), and Chi et al. (1981) agreed.") == ["Ashbaugh", "Chi"]


def test_no_reference_list_no_cited_authors():
    assert _cited("Sorry, I 2000 pay chesta\nOk, I 2024 lo vasta\nSare, U 2000 pampu\nHyderabad, T.S. 500081\n") == []
    assert _cited("Smith, K (2002) wrote it, and Jones, B (2004) answered.") == []   # two entries are no list


def test_citation_gives_the_extent_of_the_name():
    """spaCy tagged "Chi et al" whole, and the structure check refused it."""
    cluster = [Detection("Chi et al", 0, 9, 0, DetectorName.SPACY, 0.8, {"pattern": "spacy_ner"}),
               Detection("Chi", 0, 3, 0, DetectorName.REGEX, 0.85, {"pattern": "person_cue", "cue": CITATION_CUE})]
    assert _select_canonical_span(cluster, KB).text == "Chi"


def test_cited_author_is_not_scored_as_an_ordinary_word():
    start = REFERENCES.index("Beach")
    cue = Detection("Beach", start, start + 5, 0, DetectorName.REGEX, 0.85, {"pattern": "person_cue", "cue": CITATION_CUE})
    cited = CandidateResult(candidate=Candidate.new("Beach", "Beach", start, start + 5, 0, (cue,)))
    assert not person_features([cited], REFERENCES, KB)["ordinary_word"]


def test_cited_authors_come_out_of_a_paper(pipeline):
    paper = ("This study examines how the format of a report affects the decisions of its users. Earlier work "
             "found that users group similar items together (Ashbaugh et al., 1999).\n\nReferences\n\n" + REFERENCES)
    accepted = {p.display_text for p in pipeline.run_text(paper).persons}
    assert {"Abdolmohammadi", "Warfield", "Beach", "Salterio", "Hawton"} <= accepted
    # The initials after "Hawton," and the title's first word are no person in the list of authors.
    assert "K. Suicide" not in accepted


# --- whole pipeline -----------------------------------------------------------

FOOTER ="English Class IX Free distribution by Government of Telangana 2024-25"
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

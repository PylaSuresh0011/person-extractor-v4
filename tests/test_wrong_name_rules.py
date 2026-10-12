"""The rules of 2026-10-12 against wrong names. The accepted rows that are no
person - in three school textbooks and in a case folder - were each traced
to the evidence that had accepted them; these tests hold the rules that came
out of it. No word was added to a list.

  article        "the Guinness Book", "a Rajput woman": a word after an article
  place context  "In Ajmer,", "in Jordon": a word after "in" or "at"
  place alias    "Cape Verde (Kabu Verdi)": a place's other name, in brackets
  lowercase      "Suicidal Tendency": a heading of words the document also
                 writes in lowercase
  organisation   "Zomato Customer Service": an organisation and ordinary words
  person cues    "Dear Zomato Customer Service,", "Population Pyramid of India"

Tried and taken out again, measured: a single word followed by a capitalised
ordinary word ("Manohar Publishers") as the name of a thing - a chat export
writes "Devraj Bhatt We should coordinate"; no credit for "never written in
lowercase" to a word seen once - it moved nine real people out of the
accepted names for seven wrong ones; and a place at exactly half of its
mentions geographic - one book's river Krishna, another book's god Krishna.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.detection.regex_detector import RegexDetector
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.validators.entity_type_validator import EntityTypeValidator
from src.validation.validators.organization_validator import OrganizationValidator

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _candidate(doc: str, name: str, occurrence: int = 0) -> CandidateResult:
    start = -1
    for _ in range(occurrence + 1):
        start = doc.index(name, start + 1)
    detection = Detection(name, start, start + len(name), 0, DetectorName.DICTIONARY, 0.4,
                          {"pattern": "dictionary_single_token"})
    return CandidateResult(candidate=Candidate.new(name, name, start, start + len(name), 0, (detection,)))


def _entity_type(doc: str, name: str, occurrence: int = 0):
    return EntityTypeValidator().validate(_candidate(doc, name, occurrence), KB, doc)


def _rejected_as(doc: str, name: str, cue: str, occurrence: int = 0) -> bool:
    result = _entity_type(doc, name, occurrence)
    return not result.passed and result.message.startswith(cue + ":")


# --- a word after an article ---------------------------------------------------

def test_single_word_after_an_article_is_no_person():
    assert _rejected_as("His name found place in the Guinness Book of World Records.", "Guinness", "article")
    assert _rejected_as("In the town a Rajput woman arrived at the polls in a veiled chariot.", "Rajput", "article")
    # A wrapped line: no sentence ends on "the".
    assert _rejected_as("The US sent Neil Armstrong and others to the\nMoon in 1969.", "Moon", "article")


def test_article_rule_leaves_people_alone():
    assert _entity_type("Then Guinness wrote a long letter to his brother.", "Guinness").passed
    assert _entity_type("A Ramesh and K Suresh signed the register.", "Ramesh").passed      # "A" is an initial
    assert _entity_type("He is the son of the Mr. Rao who taught us.", "Rao").passed         # a title before it


# --- a word after "in" or "at" ---------------------------------------------------

def test_single_word_after_in_or_at_is_a_place():
    assert _rejected_as("The organisation was established in 1964 in Jordon and grew quickly.", "Jordon", "place_context")
    assert _rejected_as("In Ajmer, voting booths staffed by women were provided.", "Ajmer", "place_context")


def test_people_after_to_from_and_a_possessive_stay():
    assert _entity_type("She gave the letter to Jordon before leaving.", "Jordon").passed
    assert _entity_type("The meeting was held in Jordon's house last week.", "Jordon").passed


# --- another name of a place, in brackets ------------------------------------------

def test_name_in_brackets_after_a_known_place_is_that_place():
    doc = '["Montenegro (Crna Gora)","me","382"],["Montserrat","ms","1664"]'
    assert _rejected_as(doc, "Crna Gora", "place_alias")
    assert _entity_type("The award went to our teacher (Crna Gora) this year.", "Crna Gora").passed


# --- a heading of ordinary words ----------------------------------------------------

def test_phrase_the_document_also_writes_in_lowercase_is_a_heading():
    doc = ("Relationship between Abuse and Suicidal Tendency among Prisoners\n"
           "A suicidal act is rare. The suicidal tendency was measured twice. Any suicidal thought was noted, "
           "and the tendency to hide it. This tendency grows with age; the tendency is strong.")
    assert _rejected_as(doc, "Suicidal Tendency", "lowercase")


def test_a_name_word_keeps_the_phrase():
    doc = ("Mark Brown signed. We mark the brown paper, mark it again and mark it once more; "
           "the brown box, the brown bag and a brown coat.")
    assert _entity_type(doc, "Mark Brown").passed


# --- an organisation and ordinary words -------------------------------------------------

def _organization(name: str):
    return OrganizationValidator().validate(_candidate(name, name), KB, name)


def test_organisation_with_only_ordinary_words_is_no_person():
    result = _organization("Zomato Customer Service")
    assert not result.passed and "department or product" in result.message
    assert _organization("Ratan Tata").passed          # a name word beside it


# --- person cues ---------------------------------------------------------------------------

def _cues(doc: str) -> list[tuple[str, str]]:
    detector = RegexDetector(
        is_ordinary_word=lambda w: KB.is_common_word(w) and (not KB.is_known_first_name(w) or KB.is_ambiguous_first_name(w)),
        is_place=KB.is_location, is_organization=KB.is_organization)
    return [(d.metadata["cue"], d.text) for d in detector.detect(doc, 0) if d.metadata.get("cue")]


def test_salutation_to_an_organisation_is_no_person_cue():
    assert _cues("Dear Zomato Customer Service,\nmy order was not delivered.") == []
    assert ("salutation", "Ravi") in _cues("Dear Ravi,\nmy order was not delivered.")


def test_of_a_country_marks_a_name_only():
    assert ("of_country", "Nasser") in _cues("He met Nasser of Egypt in Cairo.")
    assert _cues("Graph 1 : Population Pyramid of India, 2011") == []
    assert _cues("It is run by the Ministry of Communication, Govt. of India every year.") == []
    assert _cues("The movement began in the Garhwal Himalayas of Uttarakhand in the 1970s.") == []

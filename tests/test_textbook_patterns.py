"""One test per pattern added for the Class X English textbook error brief
(2026-10-01), using the brief's own examples (its section 4) where it gives
one. Steps: a) page furniture, b) typeset normalisation, c) name
boundaries, d) other entity types. The end-to-end checks run the whole
pipeline; the rest test the one function or validator that holds the rule."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.boundary_refiner import (
    extend_left_over_initials,
    extend_particles_and_suffixes,
    trim_leading_non_name_words,
)
from src.candidate.alias_resolution import resolve_aliases
from src.candidate.explicit_aliases import find_explicit_aliases
from src.candidate.list_propagation import find_lists
from src.detection.regex_detector import RegexDetector
from src.core.models import Candidate, CandidateResult, Decision, Detection, DetectorName
from src.decision.feature_score import entity_reason, score_from_features
from src.knowledge.knowledge_base import KnowledgeBase
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.line_indexer import LineIndex
from src.preprocessing.page_furniture import blank_page_furniture
from src.preprocessing.typeset import join_wrapped_names, normalize_typeset
from src.validation.validators.corroboration_validator import CorroborationValidator
from src.validation.validators.entity_type_validator import EntityTypeValidator
from src.validation.validators.organization_validator import OrganizationValidator
from src.validation.validators.repetition_validator import RepetitionValidator
from src.validation.validators.structure_validator import StructureValidator

KB = KnowledgeBase.load(BASE_DIR / "assets")
FOOTER = "English Class X Free distribution by T.S. Government 2019-20"
FILLER = "The students read the lesson carefully and wrote answers in their notebooks.\n"


@pytest.fixture(scope="module")
def pipeline():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["enabled"] = False
    return Pipeline(config=cfg, base_dir=BASE_DIR)


def _names(result) -> set[str]:
    return {p.display_text for p in result.persons} | {p.display_text for p in result.review_persons}


def _candidate(document: str, name: str, occurrence: int = 0) -> CandidateResult:
    start = -1
    for _ in range(occurrence + 1):
        start = document.index(name, start + 1)
    end = start + len(name)
    detection = Detection(text=name, start=start, end=end, page_index=0,
                          detector=DetectorName.REGEX, confidence=0.9, metadata={"pattern": "bare"})
    return CandidateResult(candidate=Candidate.new(name, name, start, end, 0, (detection,)))


def _entity_type(document: str, name: str, occurrence: int = 0):
    return EntityTypeValidator().validate(_candidate(document, name, occurrence), KB, document)


# --- a) page furniture (brief P1) ---------------------------------------------

def test_brief_footer_gives_no_names(pipeline):
    # Brief: "English Class X Free distribution by T.S. Government" -> no names
    assert _names(pipeline.run_pages([FILLER + FOOTER] * 3)) == set()


def test_footer_glued_in_either_word_order_is_one_footer():
    pages = ["Text one.\n31Free distribution by T.S. Government 2019-20English Class X 31",
             "Text two.\n32 English Class XFree distribution by T.S. Government 2019-20",
             "Text three.\n33English Class X 33Free distribution by T.S. Government 2019-20"]
    assert [p.strip() for p in blank_page_furniture(pages)] == ["Text one.", "Text two.", "Text three."]


def test_watermark_letter_lines_are_blanked():
    pages = ["SC\nRavi went home.", "ER\nSita stayed.", "T\nThey met."]
    assert [p.strip() for p in blank_page_furniture(pages)] == ["Ravi went home.", "Sita stayed.", "They met."]


# --- b) typeset normalisation (brief P4, P7) ----------------------------------

def test_initial_glued_to_name_gets_a_space():
    # Brief: "Sri K.Venkata Rama Rayudu, OiC" - the tokenizer split the name at "K.Venkata"
    assert normalize_typeset("Sri K.Venkata Rama Rayudu, OiC") == "Sri K. Venkata Rama Rayudu, OiC"


def test_space_before_initial_dot_is_removed():
    assert normalize_typeset("S.V . Ranga Rao acted") == "S.V. Ranga Rao acted"


def test_word_hyphenated_across_line_break_is_joined():
    assert normalize_typeset("Abel Merry-\nweather came") == "Abel Merryweather\n came"


def test_fill_in_blank_becomes_one_placeholder():
    assert normalize_typeset("Srinu ________ to school.") == "Srinu ___ to school."


def test_name_wrapped_to_next_line_is_joined_without_changing_length():
    text = "Dr. Vikram\nSarabhai of the Department"
    joined = join_wrapped_names(text)
    assert joined.startswith("Dr. Vikram Sarabhai ") and len(joined) == len(text)


def test_wrapped_name_is_not_joined_to_a_designation():
    text = "Dr. G. Gopal Reddy\nDirector, SCERT"
    assert join_wrapped_names(text, KB.is_common_word) == text


def test_brief_name_wrapped_across_page_footer(pipeline):
    # Brief: "Dr. Brahm\n\n254 Free distribution...\nPrakash, father of" -> Brahm Prakash
    pages = [FILLER + "The work of Dr. Brahm\n\n254 " + FOOTER,
             "Prakash, father of the program, was honoured.\n" + FILLER + FOOTER,
             FILLER + FOOTER]
    assert any("Brahm Prakash" in n for n in _names(pipeline.run_pages(pages)))


# --- c) name boundaries (brief P4, P5) ----------------------------------------

def _refined(function, text: str, name: str) -> str:
    """The span the refiner leaves, as it stands in the text."""
    out = function([_candidate(text, name)], text, LineIndex.build(text))
    return out[0].candidate.text


def test_brief_regnal_suffix_is_part_of_the_name(pipeline):
    # Brief: "Lee Iacocca was fired by Henry Ford II" -> Lee Iacocca and Henry Ford II
    names = _names(pipeline.run_text("Lee Iacocca was fired by Henry Ford II in 1978."))
    assert {"Lee Iacocca", "Henry Ford II"} <= names


def test_name_ending_in_a_particle_is_extended():
    assert _refined(extend_particles_and_suffixes, "He admired Vincent Van Gogh deeply.", "Vincent Van") == "Vincent Van Gogh"


def test_generational_suffix_is_added():
    assert _refined(extend_particles_and_suffixes, "Tom Watson Sr. founded it.", "Tom Watson") == "Tom Watson Sr."


def test_leading_sentence_word_is_dropped():
    assert _refined(trim_leading_non_name_words, "As Bayaji walked home.", "As Bayaji") == "Bayaji"


def test_span_is_cut_at_a_dash_between_words():
    assert _refined(trim_leading_non_name_words, "Ray- Ganashatru was his film.", "Ray- Ganashatru") == "Ray"


def test_title_with_only_a_bare_initial_is_rejected():
    for text in ("Dr. G", "Sri P.", "Mr. K"):
        result = StructureValidator().validate(_candidate(text, text), KB, text)
        assert result.passed is False and result.severity == "hard", text


def test_joined_initials_before_a_name_are_valid():
    text = "M.L. Vasantha Kumari"
    assert StructureValidator().validate(_candidate(text, text), KB, text).passed is True


# --- d) other entity types (brief P2, P3) -------------------------------------

def test_brief_exercise_word_is_no_name(pipeline):
    # Brief: "Read the following passage" -> no names
    assert _names(pipeline.run_text("Read the following passage and answer the questions.")) == set()


def test_brief_quoted_film_title_is_no_name(pipeline):
    # Brief: "'Maya Bazaar' has been voted..." -> no names
    assert _names(pipeline.run_text("'Maya Bazaar' has been voted the greatest Indian film of all time.")) == set()


def test_word_also_used_in_lowercase_is_rejected():
    doc = "Potter made pots. The potter sold them. A potter works hard. Every potter knows clay."
    assert _entity_type(doc, "Potter").message.startswith("lowercase")


def test_lowercase_count_ignores_email_addresses():
    doc = "Marcus wrote. marcus.d@gmail.com, marcus.d@gmail.com, marcus.d@gmail.com sent it."
    assert _entity_type(doc, "Marcus").passed is True


def test_lowercase_word_after_a_speech_verb_cue_is_kept():
    doc = "Will said hello. I will go. You will see. We will win."
    assert _entity_type(doc, "Will").passed is True


def test_exercise_vocabulary_is_rejected_but_not_after_a_first_name():
    assert _entity_type("Supplementary Reader for Class X", "Supplementary Reader").message.startswith("vocabulary")
    assert _entity_type("Herbert Read wrote on art.", "Herbert Read").passed is True


def test_kinship_word_on_its_own_is_rejected():
    assert _entity_type("Grandpa smiled at us.", "Grandpa").message.startswith("kinship")


def test_contraction_is_rejected():
    assert _entity_type("I'm going home.", "I'm").message.startswith("contraction")


def test_numbered_section_heading_is_rejected():
    assert _entity_type("Unit 4\nIII. Letter Writing\nWrite a letter.", "III. Letter Writing").message.startswith(
        "section_header")


def test_place_building_or_award_word_at_the_end_is_rejected():
    for doc, name in (("He lived in Krishna Dist for years.", "Krishna Dist"),
                      ("They met at Philip's Church on Sunday.", "Philip's Church"),
                      ("She won the Nobel Prize in 1911.", "Nobel Prize")):
        assert _entity_type(doc, name).message.startswith("place_suffix"), name


def test_surname_that_is_also_a_building_word_is_kept_after_a_first_name():
    assert _entity_type("Charlotte Church sang well.", "Charlotte Church").passed is True


def test_place_word_after_the_name_is_rejected():
    assert _entity_type("Mr. Nana lives in Koko Village, Nigeria.", "Koko").message.startswith("place_next")


def test_year_in_brackets_after_the_name_is_rejected():
    assert _entity_type("Her first film was Varasdar (1956).", "Varasdar").message.startswith("year")


def test_word_mostly_after_place_words_is_rejected():
    doc = "I left for Dirang at ten. Finally, we reached Dirang. Dirang was cold."
    assert _entity_type(doc, "Dirang", 2).message.startswith("place_context")


def test_title_in_quotes_is_rejected_but_not_a_nickname_or_speech():
    assert _entity_type("We saw the play 'The Dear Departed' today.", "Departed").message.startswith("quoted")
    assert _entity_type("He was affectionately called 'Manikda' by all.", "Manikda").passed is True
    assert _entity_type('"Ravi!" she called from the door.', "Ravi").passed is True


def test_part_of_a_longer_quoted_name_does_not_count_for_the_shorter_one():
    doc = "'Maya Bazaar' ran long. 'Maya Bazaar' won. My name is Maya. Maya was born here."
    assert _entity_type(doc, "Maya", 2).passed is True


def test_word_mostly_after_the_is_rejected():
    doc = "The West was far. They went to the West. In the West it rained. West winds blew."
    assert _entity_type(doc, "West", 3).message.startswith("the")


def test_plural_after_the_is_a_family_or_group():
    assert _entity_type("We visited the Slaters on Sunday.", "Slaters").message.startswith("the")


def test_language_or_group_word_on_its_own_is_rejected():
    assert _entity_type("She teaches English at school.", "English").message.startswith("group")


# --- e) new detection patterns (brief P7, P8) ---------------------------------

def _cues(text: str) -> list[tuple[str, str]]:
    detector = RegexDetector(is_ordinary_word=lambda w: KB.is_common_word(w) and (
        not KB.is_known_first_name(w) or KB.is_ambiguous_first_name(w)))
    return [(d.text, d.metadata["cue"]) for d in detector.detect(text, 0) if d.metadata["pattern"] == "person_cue"]


def _regex_spans(text: str) -> list[str]:
    return [d.text for d in RegexDetector().detect(text, 0) if d.metadata["pattern"] in ("titled", "bare")]


def test_brief_title_initial_name_comma(pipeline):
    # Brief: "Sri K.Venkata Rama Rayudu, OiC" -> K. Venkata Rama Rayudu
    pages = [FILLER + "Sri K.Venkata Rama Rayudu, OiC, Textbook Press, Hyderabad.\n" + FOOTER,
             FILLER + FOOTER, FILLER + FOOTER]
    assert any("K. Venkata Rama Rayudu" in n for n in _names(pipeline.run_pages(pages)))


def test_brief_list_of_names(pipeline):
    # Brief: "Ghantasala (music), M.L.Vasantha Kumari, Leela, Suseela and Madhavapeddi (playback)" -> 5 names
    line = "Ghantasala (music), M.L.Vasantha Kumari, Leela, Suseela and Madhavapeddi (playback) worked on it.\n"
    names = _names(pipeline.run_pages([line + FOOTER, FILLER + FOOTER, FILLER + FOOTER]))
    assert {"Ghantasala", "M.L. Vasantha Kumari", "Leela", "Suseela", "Madhavapeddi"} <= names


def test_brief_kinship_cue(pipeline):
    # Brief: "His wife Sumanthamma is in a state of shock" -> Sumanthamma
    assert "Sumanthamma" in _names(pipeline.run_text("His wife Sumanthamma is in a state of shock."))


def test_indian_honorific_is_a_title():
    assert _regex_spans("Sri Navuluri Peraiah, SA, ZPHS") == ["Sri Navuluri Peraiah"]


def test_joined_initials_are_one_name_token():
    assert _regex_spans("Dr. K.N. Anandan, Linguist") == ["Dr. K.N. Anandan"]
    assert _regex_spans("from S.V. Ranga Rao, Savitri") == ["S.V. Ranga Rao"]


def test_kinship_words_mark_the_names_after_them():
    assert _cues("survived by wife and two daughters Rashmitha (3) and Ashwitha.") == [
        ("Rashmitha", "kinship"), ("Ashwitha", "kinship")]
    assert _cues("My brother -in-law Jallaluddin would tell") == [("Jallaluddin", "kinship")]
    assert _cues("Father: Frank Hawking, a medical researcher") == [("Frank Hawking", "kinship")]
    assert _cues("Mother Teresa served the poor.") == []


def test_sign_off_and_salutation_mark_a_name():
    assert _cues("Thanking you,\nYours faithfully,\nRamu\nGovt. High School") == [("Ramu", "sign_off")]
    assert _cues("Dear Ravi,\nHow are you?") == [("Ravi", "salutation")]
    assert _cues("The Dear Departed (Part-I)") == []
    assert _cues("Dear Student,\nAs you all know") == []


def test_speaker_label_used_twice_marks_a_name():
    play = "Sarath : Oh! What happened?\nRishi : I saw it.\nSarath : It was great.\nRishi : Yes."
    assert sorted(set(_cues(play))) == [("Rishi", "speaker"), ("Sarath", "speaker")]
    assert _cues("Sarath : Oh! What happened?") == []                  # used once
    assert _cues("Student : Yes.\nStudent : No.") == []               # an ordinary word
    assert _cues("Ref: Grey Ridge | Code: 1\nRef: Grey Ridge | Code: 2") == []  # record fields


def test_rare_surname_after_a_first_name_like_word_is_trusted():
    doc = "He read about Stephen Hawking yesterday."
    cand = _candidate(doc, "Stephen Hawking")
    spacy = Detection(text="Stephen Hawking", start=cand.candidate.start, end=cand.candidate.end, page_index=0,
                      detector=DetectorName.SPACY, confidence=0.8, metadata={"pattern": "spacy_ner"})
    cand.candidate = Candidate.new("Stephen Hawking", "Stephen Hawking", cand.candidate.start, cand.candidate.end,
                                   0, (spacy,))
    assert CorroborationValidator().validate(cand, KB, doc).passed is True


def test_list_needs_and_or_three_items():
    assert [[i.text for i in l] for l in find_lists("students Plato and Xenophon wrote")] == [["Plato", "Xenophon"]]
    assert [[i.text for i in l] for l in find_lists("producers Nagireddy and\nChakrapani, would")] == [
        ["Nagireddy", "Chakrapani"]]
    assert find_lists("Finally, Ravi went home.") == []


def test_name_is_extended_over_initials_before_it():
    assert _refined(extend_left_over_initials, "from S.V. Ranga Rao, Savitri", "Ranga Rao") == "S.V. Ranga Rao"
    assert _refined(extend_left_over_initials, "rich in vitamin A. Ravi said so", "Ravi") == "Ravi"
    assert _refined(extend_left_over_initials, "printed on 200 G.S.M. White Art Card", "White") == "White"


def test_word_repeated_with_another_between_is_allowed():
    assert RepetitionValidator().validate(_candidate("Rao Gopal Rao", "Rao Gopal Rao"), KB, "Rao Gopal Rao").passed
    assert not RepetitionValidator().validate(_candidate("John John", "John John"), KB, "John John").passed


def test_government_is_an_organisation_word():
    assert not OrganizationValidator().validate(_candidate("T.S. Government", "T.S. Government"), KB,
                                                "T.S. Government").passed


# --- f) aliases (brief P6) -----------------------------------------------------

def _row(display: str, occurrences: int = 1, page: int = 1) -> dict:
    return {"display": display, "occurrences": occurrences, "decision": "accepted", "files": ["book.pdf"],
            "places": {("book.pdf", page)}}


def _aliases(rows: list[dict], explicit=(), merge: bool = False) -> dict[str, list[str]]:
    return {e["display"]: e["aliases"] for e in resolve_aliases(rows, explicit, merge=merge)}


def test_forms_of_one_person_are_aliases():
    # Brief: Kalam, Abdul Kalam, Dr. Kalam / Mrs. Slater, Amelia, Mrs. Amelia Slater
    rows = [_row("Kalam", 9), _row("Abdul Kalam", 3), _row("Dr. Kalam", 2), _row("Mrs. Slater", 5), _row("Amelia", 4),
            _row("Mrs. Amelia Slater", 2)]
    merged = _aliases(rows, merge=True)
    assert sorted(merged["Abdul Kalam"]) == ["Dr. Kalam", "Kalam"]
    assert sorted(merged["Mrs. Amelia Slater"]) == ["Amelia", "Mrs. Slater"]
    assert sorted(_aliases(rows)["Kalam"]) == ["Abdul Kalam", "Dr. Kalam"]  # default: rows kept, aliases listed


def test_ambiguous_surname_is_not_merged():
    # Brief: Murthy -> Sudha / Narayana Murthy - do not merge
    rows = [_row("Murthy", 6), _row("Sudha Murthy", 2), _row("Narayana Murthy", 2)]
    assert _aliases(rows)["Murthy"] == []


def test_spelling_variants_and_initials_are_one_person():
    # Brief: Amebedkar/Ambedkar, Shastry/Sastry, Hendricks/Hendriks
    rows = [_row("Ambedkar"), _row("Babasaheb Amebedkar"), _row("Shastry"), _row("Sastry"),
            _row("R.K. Agnihotri"), _row("Rama Kant Agnihotri"), _row("Sarada"), _row("Sarala")]
    aliases = _aliases(rows)
    assert aliases["Ambedkar"] == ["Babasaheb Amebedkar"] and aliases["Sastry"] == ["Shastry"]
    assert aliases["R.K. Agnihotri"] == ["Rama Kant Agnihotri"]
    assert aliases["Sarada"] == []  # a different letter is a different name


def test_different_people_are_not_merged():
    rows = [_row("Mr. Slater"), _row("Mrs. Slater"), _row("Henry Ford"), _row("Henry Ford II"),
            _row("Albert", page=140), _row("Albert Einstein", page=200)]
    aliases = _aliases(rows)
    assert aliases["Mr. Slater"] == [] and aliases["Henry Ford"] == [] and aliases["Albert"] == []


def test_alias_stated_in_the_text():
    # Brief: "Nandamuri Taraka Ramarao (NTR)", "Manikda [as Ray was affectionately called"
    text = "youth, Nandamuri Taraka Ramarao (NTR) to pretend. Manikda [as Ray was\naffectionately called] was shy."
    accepted = []
    for name in ("Nandamuri Taraka Ramarao", "Ray"):
        c = _candidate(text, name)
        c.state.decision = Decision.ACCEPTED
        accepted.append(c)
    assert find_explicit_aliases(text, accepted) == [("NTR", "Nandamuri Taraka Ramarao"), ("Manikda", "Ray")]
    assert _aliases([_row("Nandamuri Taraka Ramarao"), _row("NTR")], [("NTR", "Nandamuri Taraka Ramarao")])["NTR"] == [
        "Nandamuri Taraka Ramarao"]


def test_acronym_alias_mentions_are_names(pipeline):
    text = ("In the film, Savitri meets Nandamuri Taraka Ramarao (NTR) at the gate. Later NTR and Savitri sing. "
            "Nandamuri Taraka Ramarao was a star.")
    assert "NTR" in _names(pipeline.run_text(text))


# --- g) feature score (brief P8) -----------------------------------------------

def test_person_cues_score_higher_than_ordinary_words(pipeline):
    result = pipeline.run_text("His wife Sumanthamma is in a state of shock. Dr. Rama Kant Agnihotri visited her.")
    scores = {p.display_text: p.score for p in list(result.persons) + list(result.review_persons)}
    assert scores["Dr. Rama Kant Agnihotri"] >= 0.7 and scores["Sumanthamma"] >= 0.4
    features = {"lowercase_use": True, "ordinary_word": True}
    assert score_from_features(features) < 0.4


def test_entity_reason_names_the_evidence(pipeline):
    result = pipeline.run_text("His wife Sumanthamma is in a state of shock.")
    person = next(p for p in result.persons + result.review_persons if p.display_text == "Sumanthamma")
    assert "kinship word" in entity_reason(person.mentions)


# --- found on the real case scan after steps a-g --------------------------------

def test_owner_confirmed_name_is_never_another_entity_type():
    # "Siri" (assets/owner_decisions/confirmed_names.txt) also appears as "siri call $CONTACTS"
    doc = "Siri call $CONTACTS\nsiri call $CONTACTS\nsiri text $CONTACTS\nsiri open $APP"
    assert _entity_type(doc, "Siri").passed is True


def test_dump_lines_are_no_speaker_labels():
    dump = ("GestureMode: MULTI_TOUCH DeviceType: TOUCH_SCREEN\nGestureMode: MULTI_TOUCH DeviceType: TOUCH_SCREEN\n"
            'Ssp: "PatternMatcher{LITERAL: com.sec.android.app}"\nSsp: "PatternMatcher{LITERAL: com.sec.android.b}"')
    assert _cues(dump) == []


def test_data_lines_get_no_rare_surname_exemption():
    doc = '["Fiji","fj","679"],["French Guiana (Guyane)","gf","594"],["French Polynesia","pf","689"]'
    cand = _candidate(doc, "French Guiana")
    spacy = Detection(text="French Guiana", start=cand.candidate.start, end=cand.candidate.end, page_index=0,
                      detector=DetectorName.SPACY, confidence=0.8, metadata={"pattern": "spacy_ner"})
    cand.candidate = Candidate.new("French Guiana", "French Guiana", cand.candidate.start, cand.candidate.end, 0, (spacy,))
    assert CorroborationValidator().validate(cand, KB, doc).passed is False

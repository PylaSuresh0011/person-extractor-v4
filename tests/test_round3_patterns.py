"""The unit tests of the round-2 error brief (docs/PERSON_EXTRACTOR_improvement_
brief_v2.md, section 4) - the Social Studies textbook's patterns, round 3
(2026-10-01). Each runs the whole pipeline on a PDF-like set of pages (Fast
mode: no LLM reviewer), except the alias-cluster test, which tests the
combined list's resolution directly."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.alias_resolution import resolve_aliases
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.typeset import normalize_typeset

FOOTER = "Social Studies Class X Free distribution by Government of Telangana 2020-21"
FILLER = "The students read the lesson carefully and wrote answers in their notebooks.\n"


@pytest.fixture(scope="module")
def pipeline():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["enabled"] = False
    return Pipeline(config=cfg, base_dir=BASE_DIR)


def _names(pipeline, line: str) -> set[str]:
    result = pipeline.run_pages([FILLER + line + "\n" + FOOTER, FILLER + FOOTER, FILLER + FOOTER])
    return {p.display_text for p in result.persons} | {p.display_text for p in result.review_persons}


def test_rivers_in_a_list_are_no_persons(pipeline):
    # Brief: "the Jhelum, the Chenab, the Ravi, the Beas and the Sutlej" -> no persons
    assert _names(pipeline, "The Indus and its tributaries, the Jhelum, the Chenab, the Ravi, the Beas and the "
                            "Sutlej flowing from the Himalayas.") == set()


def test_months_and_seasons_are_no_persons(pipeline):
    # Brief: "Vasantha Chaitra, Varsha Sraavana, Sharad Aswayuja" -> no persons
    assert _names(pipeline, "The seasons are Vasantha Chaitra, Varsha Sraavana, Sharad Aswayuja and others.") == set()


def test_a_movement_is_no_person(pipeline):
    # Brief: "Meira Paibi Movement" -> no persons
    assert _names(pipeline, "The Meira Paibi Movement protested against the army in Manipur.") == set()


def test_dotless_initials_glued(pipeline):
    # Brief: "National Front Government with Deve Gowda and IK Gujral as PMs" -> Deve Gowda, I.K. Gujral
    assert {"Deve Gowda", "I.K. Gujral"} <= _names(
        pipeline, "The National Front Government with Deve Gowda and IK Gujral as PMs lasted two years.")


def test_dotless_initials_after_led_by(pipeline):
    # Brief: "NDA government led by AB Vajpayee" -> A.B. Vajpayee
    assert "A.B. Vajpayee" in _names(pipeline, "The NDA government led by AB Vajpayee came to power in 1998.")


def test_dotless_initials_are_dotted_but_acronyms_are_not():
    assert normalize_typeset("L K Advani met the US President and the NDA leaders") == \
        "L.K. Advani met the US President and the NDA leaders"


def test_name_of_country(pipeline):
    # Brief: "Nasser of Egypt, Sukarno of Indonesia, Tito of Yugoslavia" -> 3 persons
    assert {"Nasser", "Sukarno", "Tito"} <= _names(
        pipeline, "Leaders like Nasser of Egypt, Sukarno of Indonesia, Tito of Yugoslavia met at Bandung.")


def test_hyphenated_name_with_life_span(pipeline):
    # Brief: "a republic was established under Sun-Yat-Sen (1866-1925)" -> Sun Yat-sen
    assert "Sun-Yat-Sen" in _names(pipeline, "In 1911 a republic was established under Sun-Yat-Sen (1866-1925).")


def test_short_word_asian_name_before_who(pipeline):
    # Brief: "In February 1930, Ho Chi Minh (who spent time in France" -> Ho Chi Minh
    assert "Ho Chi Minh" in _names(pipeline, "In February 1930, Ho Chi Minh (who spent time in France) formed it.")


def test_initials_beat_a_place_match(pipeline):
    # Brief: "... J.V. Narsinga Rao and K.V. Ranga Reddy from Telangana" -> 4 persons incl. K.V. Ranga Reddy
    names = _names(pipeline, "Burgula Ramakrishna Rao, Marri Chenna Reddy, J.V. Narsinga Rao and K.V. Ranga Reddy "
                             "from Telangana attended.")
    assert {"Burgula Ramakrishna Rao", "Marri Chenna Reddy", "J.V. Narsinga Rao", "K.V. Ranga Reddy"} <= names


def test_district_is_no_person(pipeline):
    # Brief: "hailstorm in Chevella region of Ranga Reddy district" -> no persons
    assert _names(pipeline, "There was a hailstorm in Chevella region of Ranga Reddy district last week.") == set()


def test_title_on_its_own_is_no_person(pipeline):
    # Brief: "Mr. President, Sir, I introduce the Draft" -> no persons
    assert _names(pipeline, "Mr. President, Sir, I introduce the Draft Constitution today.") == set()


def test_religious_title(pipeline):
    # Brief: "Pastor Niemoeller, a resistance fighter" -> Niemoeller
    assert any("Niemoeller" in n for n in _names(pipeline, "Pastor Niemoeller, a resistance fighter, wrote it."))


def test_parents_were_x_and_y(pipeline):
    # Brief: "His parents were Mahalakshimi and Lakshmikanth Rao" -> 2 persons
    assert {"Mahalakshimi", "Lakshmikanth Rao"} <= _names(
        pipeline, "He was born in a village. His parents were Mahalakshimi and Lakshmikanth Rao.")


def test_one_row_per_alias_cluster():
    # Brief: exactly 1 row for {Nehru, Jawaharlal Nehru, J. Nehru, Pandit Jawaharlal Nehru}
    rows = [{"display": d, "occurrences": n, "decision": "accepted", "files": ["s.pdf"], "places": {("s.pdf", 1)}}
            for d, n in (("Nehru", 10), ("Jawaharlal Nehru", 11), ("J. Nehru", 1), ("Pandit Jawaharlal Nehru", 1),
                         ("King Mahendra", 1), ("King", 5))]
    merged = resolve_aliases(rows, merge=True)
    nehru = [r for r in merged if "Nehru" in r["display"]]
    assert len(nehru) == 1 and len(nehru[0]["aliases"]) == 3
    # "King Mahendra" stays apart from a bare "King"
    assert {r["display"] for r in merged} >= {"King Mahendra", "King"}

"""The rules of 2026-10-21 for reference lists. In a case folder scanned
without the LLM reviewer, the wrong names left in three research papers were
words of journal titles ("Bull. Am. Meteorol. Soc.", "J. Nucl. Mater.", "N
Engl j med") and publishers ("Boston: Houghton Mifflin.", "John Wiley &
Sons", "Prentice Hall"). The examples here are made up in the same shapes.

  journal    a word with a period, the rest of the title, then volume and
             pages - or medicine's style, "; volume: pages", without periods
  publisher  after the year of the entry and "Town(, State):", to its end
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.validators.entity_type_validator import EntityTypeValidator

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


# --- journal titles --------------------------------------------------------------

def test_word_of_an_abbreviated_journal_title_is_no_person():
    assert _rejected_as("4K. Mehta and R. Iyer, Solid State Technol. 41, 210 (1998); R. Gupta, Bull. Am. \n"
                        "Meteorol. Soc. 80, 115 (2004).", "Bull", "journal")
    assert _rejected_as("5A. Sen, P. Das, and L. Roy, J. Nucl. Mater. 290–295, \n640 (2001).", "Mater", "journal")
    assert _rejected_as("vi. Nair, S. (1990). Patients in Prison. Br.J. Psych. 151; 40-48.", "J. Psych", "journal")
    assert _rejected_as("2T. Rao and A. Bose, J. Non-Cryst. Solids210, 1520 \n(2006).", "J. Non-Cryst", "journal")


def test_journal_title_in_medicine_style_is_no_person():
    assert _rejected_as("xl. Wright, M. (1998) Deaths among buyers of handguns. N Engl j med; 312: 1200- 1205.",
                        "Engl", "journal")
    assert _rejected_as("Shah P. Deaths among buyers. N Engl J Med. 2001;345:12-19.", "Engl", "journal")


def test_cited_authors_and_chat_lines_stay():
    # An author is followed by a comma or initials, not by the volume.
    assert _entity_type("4K. Mehta and R. Iyer, Solid State Technol. 41, 210 (1998).", "Iyer").passed
    assert _entity_type("Shah P, Rao K. Aging. Lancet. 2001;345:12-19.", "Rao").passed
    assert _entity_type("Lawrence MG. Bull Am Meteorol Soc. 2005;86:225-233.", "Lawrence").passed
    # A time or a count after a name.
    assert _entity_type("ok see you Ravi. 10:30 at the gate.", "Ravi").passed
    assert _entity_type("Thanks Ravi see you. 10:30 tomorrow then", "Ravi").passed
    assert _entity_type("We met Ravi. 25, 30 people came to the hall.", "Ravi").passed
    assert _entity_type("He met Ravi. Mr. Sen came later with 5, 6 friends.", "Ravi").passed


# --- publishers ------------------------------------------------------------------

def test_publisher_at_the_end_of_a_reference_is_no_person():
    assert _rejected_as("xxx. Mehta, P.R. (1975). A Study of Work. Boston: Houghton Mifflin. \n",
                        "Houghton Mifflin", "publisher")
    assert _rejected_as("Das, R (1984), Decisions and Choice, Chichester, UK, John Wiley & Sons. \n",
                        "John Wiley", "publisher")
    assert _rejected_as("Iyer, S. J. (1991), Judgment at work, Englewood Cliffs, NJ: Prentice Hall. \n",
                        "Prentice", "publisher")
    # Wrapped onto the next line, the entry ending at a blank line.
    assert _rejected_as("Rao, A and Sen, H.A (1977), Problem solving, Englewood Cliffs, N.J., Prentice \nHall \n \n"
                        "Next, B (2006), Another title", "Prentice", "publisher")


def test_people_after_a_town_or_in_an_author_list_stay():
    # A news dateline: no year before it, and the sentence runs on.
    assert _entity_type("Hyderabad: Ravi Kumar, the collector, said on Monday that the work is done.",
                        "Ravi Kumar").passed
    assert _entity_type("(2019) Hyderabad: Ravi Kumar said the work is done.", "Ravi Kumar").passed
    # Authors come before the year.
    assert _entity_type("Kessler, R.C., Borges, G., & Walters, E.E. (1999). Prevalence of attempts.", "Walters").passed
    assert _entity_type("Rao, K. (2001). In: Ravi Kumar (Ed.), Studies. Delhi: Sage.", "Ravi Kumar").passed

"""Tests for src.preprocessing.language_filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.language_filter import MIN_WORDS_FOR_LANGUAGE_CHECK, check_english, is_short_non_prose

KB = KnowledgeBase.load(BASE_DIR / "assets")


def test_short_text_is_always_treated_as_english():
    assert check_english("0g ~~ | Suresh 8 ~ ## 4)", KB).is_english


def test_short_non_prose_catches_ocr_scraps_and_terse_records():
    # (2026-09-28, exp08) Made-up examples of the shapes found in a real case:
    # a garbled OCR scrap of an image, a lone word, a terse key/value record.
    assert is_short_non_prose("0g ~~ | Suresh 8 ~ ఇ అ ## 4)", KB)
    assert is_short_non_prose("Suresh", KB)
    assert is_short_non_prose("caseNo: 100, station: Central, officerName: Ravi Kumar", KB)


def test_short_genuine_notes_are_prose():
    for note in ("Call Rajesh tomorrow about the payment.",
                 "Suresh met Priya at the station and gave her the keys.",
                 "Thanks Nadia, see you soon"):
        assert not is_short_non_prose(note, KB), note


def test_long_text_is_never_short_non_prose():
    text = " ".join(["Suresh"] * MIN_WORDS_FOR_LANGUAGE_CHECK)
    assert not is_short_non_prose(text, KB)


SCREEN = ("Search Datasets\nFilters  Datasets  Computer Science  Education\nTrending Datasets\n"
          "Airbnb Reviews: Travel Stays\nNadia Kovalenko - Updated 2 days ago\nUsability 10.0 - 1 File - 7 kB\n"
          "Admissions Dataset - 200,000 Students\nTrevor Higgins - Updated 14 days ago\nUsability 10.0 - 1 File - 5 MB\n"
          "Movies Shows Series\nMarcus Delaney - Updated 3 hours ago\nUsability 9.4 - 2 Files\n")


def test_an_app_screen_is_english_without_being_prose():
    """(2026-10-12) A screenshot's text: labels, titles and names, few
    function words - below the prose threshold, and English all the same."""
    check = check_english(SCREEN, KB)
    assert 0.10 <= check.english_word_ratio < 0.15, check
    assert check.is_english


def test_the_grey_zone_still_skips_other_languages_and_scripts():
    french = ("Contrat de licence utilisateur final pour le logiciel. Vous devez lire attentivement ce contrat "
              "avant toute utilisation du produit, car il contient des informations importantes sur vos droits. "
              "Merci the and of to ")
    check = check_english(french, KB)
    assert check.english_word_ratio >= 0.10 and not check.is_english, check
    # A document in another script keeps a few Latin words - a brand name, a URL.
    other_script = "నిబంధనలు మరియు షరతులు " * 40 + SCREEN
    assert not check_english(other_script, KB).is_english


def test_the_grey_zone_is_for_a_screen_of_text_not_a_log():
    """The same words thirty times over are a machine log's size: skipped as before."""
    log = check_english(SCREEN * 30, KB)
    assert log.word_count > 1000 and 0.10 <= log.english_word_ratio < 0.15 and not log.is_english

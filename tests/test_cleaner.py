"""Tests for src.preprocessing.cleaner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.cleaner import clean_text

UNDECODED = "�"


def test_a_long_run_of_undecoded_characters_is_padding():
    # (2026-10-13) A file padded with NUL bytes, and a copy of it in which an
    # earlier tool had turned every NUL into U+FFFD: both clean to the same text.
    with_nuls = "app_process_name=com.example\n# " + "\x00" * 5000 + "\nversion=2\n"
    decoded = with_nuls.replace("\x00", UNDECODED)
    assert clean_text(decoded) == clean_text(with_nuls) == "app_process_name=com.example\n#\nversion=2\n"


def test_a_wrongly_decoded_character_is_left_alone():
    # One character read in the wrong encoding gives one to four of them.
    for text in (f"Jos{UNDECODED} Ramos called.", f"Caf{UNDECODED * 3} Luna", f"x{UNDECODED * 7}y"):
        assert clean_text(text) == text

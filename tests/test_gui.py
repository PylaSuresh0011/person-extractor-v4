"""Tests for gui.py's non-window parts (path parsing, combining, saving)."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gui


def test_parse_paths_handles_quotes_blanks_comments_folders_and_problems(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "folder"
    sub.mkdir()
    (sub / "b.txt").write_text("x", encoding="utf-8")
    (sub / "skip.exe").write_text("x", encoding="utf-8")
    text = f'"{tmp_path / "a.txt"}"\n\n# comment\n{sub}\n{tmp_path / "missing.txt"}\n{tmp_path / "a.txt"}\n'
    files, problems = gui.parse_paths(text)
    assert files == [tmp_path / "a.txt", sub / "b.txt"]  # folder expanded, duplicate dropped
    assert problems == [f"Not found: {tmp_path / 'missing.txt'}"]


def _person(name, count):
    return SimpleNamespace(normalized_text=name, display_text=name, occurrence_count=count)


def test_combine_sums_and_accepted_wins(tmp_path):
    r1 = SimpleNamespace(persons=[_person("Ravi Kumar", 2)], review_persons=[_person("Nadia", 1)])
    r2 = SimpleNamespace(persons=[_person("Nadia", 3)], review_persons=[])
    rows = gui.combine([("one.txt", r1), ("two.txt", r2)])
    assert [(e["display"], e["occurrences"], e["decision"], e["files"]) for e in rows] == [
        ("Nadia", 4, "accepted", ["one.txt", "two.txt"]),
        ("Ravi Kumar", 2, "accepted", ["one.txt"]),
    ]
    csv_path = gui.save(rows, [Path("one.txt"), Path("two.txt")], [], out_dir=tmp_path)
    assert csv_path.exists() and csv_path.with_suffix(".txt").exists()
    assert "Nadia" in csv_path.with_suffix(".txt").read_text(encoding="utf-8")

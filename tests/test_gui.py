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


def test_every_run_is_stored_with_its_times_and_added_to_the_history(tmp_path):
    import csv
    from datetime import datetime
    rows = [{"display": "Nadia", "occurrences": 4, "decision": "accepted", "files": ["one.txt"]}]
    file_rows = [{"path": "one.txt", "status": "ok", "accepted": 1, "review": 0, "seconds": 2.5},
                 {"path": "bad.pdf", "status": "failed: unreadable", "accepted": 0, "review": 0, "seconds": 0.1}]
    start, end = datetime(2026, 10, 1, 9, 58, 0), datetime(2026, 10, 1, 10, 1, 7)
    first = gui.save_run(rows, file_rows, start, end, "full", "on (test)", ["C:/case"], out_dir=tmp_path)
    second = gui.save_run(rows, file_rows, start, end, "fast", "off", ["C:/case"], out_dir=tmp_path)
    assert first.name == "2026-10-01_09-58-00" and second.name == "2026-10-01_09-58-00_2"  # never overwritten
    for name in ("results.csv", "results.txt", "files.csv"):
        assert (first / name).exists()
    report = (first / "results.txt").read_text(encoding="utf-8")
    for expected in ("Date        : 2026-10-01", "Started     : 09:58:00", "Finished    : 10:01:07",
                     "Time taken  : 3m 07s", "Files       : 2 (1 failed)", "Nadia"):
        assert expected in report, expected
    # Run data and extraction data together, per run and across all runs.
    per_run = list(csv.DictReader(open(first / "results.csv", encoding="utf-8")))
    assert per_run[0]["name"] == "Nadia" and per_run[0]["start_time"] == "09:58:00" and per_run[0]["time_taken"] == "3m 07s"
    all_runs = list(csv.DictReader(open(tmp_path / gui.ALL_RUNS_FILE, encoding="utf-8")))
    assert [(r["run_folder"], r["name"], r["mode"]) for r in all_runs] == [
        ("2026-10-01_09-58-00", "Nadia", "full"), ("2026-10-01_09-58-00_2", "Nadia", "fast")]
    empty = gui.save_run([], file_rows, start, end, "full", "on", ["C:/x"], out_dir=tmp_path)
    last = list(csv.DictReader(open(tmp_path / gui.ALL_RUNS_FILE, encoding="utf-8")))[-1]
    assert last["run_folder"] == empty.name and last["name"] == ""  # a run with no names still appears
    history = list(csv.DictReader(open(tmp_path / "run_history.csv", encoding="utf-8")))
    assert [h["mode"] for h in history] == ["full", "fast", "full"]
    assert history[0]["start_time"] == "09:58:00" and history[0]["end_time"] == "10:01:07"
    assert history[0]["time_taken_seconds"] == "187.0" and history[0]["files_failed"] == "1"


def test_format_duration():
    assert [gui.format_duration(s) for s in (12.44, 187, 3725)] == ["12.4s", "3m 07s", "1h 02m 05s"]

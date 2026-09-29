"""
src.export.phase1_exporter
=============================
phase_1.csv - Phase 1's output: the names that pass into the validation
firewall. The regex, dictionary and spaCy detections are merged into
candidates (CandidateFactory) and boundary-refined; nothing has been
validated, scored or decided yet. One row per distinct name, like the
normal names CSV but without a decision:

  name, occurrences, num_source_files, detectors, patterns, source_files

"detectors" lists every detector that found the name anywhere
("dictionary+regex+spacy"), "patterns" their patterns. Written by cli.py
and scripts/scan_directory.py with --phase1-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.core.models import ExtractionResult

COLUMNS = ["name", "occurrences", "num_source_files", "detectors", "patterns", "source_files"]
FILE_NAME = "phase_1"


def phase1_path(output_dir: Path) -> Path:
    """output_dir/phase_1.csv - or phase_1_2.csv, phase_1_3.csv, ... when
    that name is taken, so a new run never overwrites an earlier one."""
    path, n = output_dir / f"{FILE_NAME}.csv", 2
    while path.exists():
        path, n = output_dir / f"{FILE_NAME}_{n}.csv", n + 1
    return path


def add_phase1_names(names: dict[str, dict], result: ExtractionResult, source_label: str) -> int:
    """Adds one file's firewall input to `names` (name -> totals); returns
    how many candidates the file had."""
    for c in result.firewall_input:
        entry = names.setdefault(c.candidate.normalized_text,
                                 {"occurrences": 0, "files": {}, "detectors": set(), "patterns": set()})
        entry["occurrences"] += 1
        entry["files"][source_label] = None
        for d in c.candidate.source_detections:
            entry["detectors"].add(d.detector.value)
            entry["patterns"].add(d.metadata.get("pattern", ""))
    return len(result.firewall_input)


def write_phase1_csv(path: Path, names: dict[str, dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for name, e in sorted(names.items(), key=lambda kv: (-kv[1]["occurrences"], kv[0].lower())):
            writer.writerow([name, e["occurrences"], len(e["files"]), "+".join(sorted(e["detectors"])),
                             "+".join(sorted(e["patterns"])), "; ".join(e["files"])])
    return path

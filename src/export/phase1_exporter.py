"""
src.export.phase1_exporter
=============================
CSV of Phase 1's output (ingestion and detection) - one row per raw
detection from the regex, dictionary and spaCy detectors, before the
validation firewall, the ML classifier or any decision. Written by
cli.py and scripts/scan_directory.py with --phase1-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.core.models import ExtractionResult

COLUMNS = ["source_file", "detector", "pattern", "text", "page", "line", "column",
           "char_start", "char_end", "confidence"]


def phase1_rows(result: ExtractionResult, source_label: str) -> list[list]:
    return [
        [source_label, d.detector.value, d.metadata.get("pattern", ""), d.text,
         loc.page_number if loc.page_number is not None else "", loc.line_number, loc.column_number,
         d.start, d.end, f"{d.confidence:.2f}"]
        for d, loc in result.detections
    ]


def write_phase1_csv(path: Path, rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path

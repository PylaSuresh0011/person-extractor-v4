"""Create a deterministic, file-separated evaluation manifest.

The manifest stores paths and hashes only; it never copies case material.
Files listed here are automatically excluded by retrain_from_feedback.py,
so they remain an honest holdout set.

By default it reserves PDF and TXT files, as it always has. With
`--formats all` (added 2026-10-01) it draws from every format the tool reads
(docx, html, eml, json, csv, xml...), spread evenly over the formats found -
each gets count // formats files, the remainder going to random formats -
so the evaluation shows how each kind of file does, not just the most
common one. A file is only reserved when it has at least --min-words words
of text (an image-only PDF can't be labelled) and at most --max-words (one
sitting's labeling), only once per content (the same file saved twice
counts once), and never when it already
appears in the feedback or training data (datasets/feedback/*.jsonl): it
has to be unseen.

An existing manifest is never replaced without --overwrite: replacing it
silently invalidates every label made for it.

Usage:
    python scripts/create_pdf_txt_holdout.py --input-dir "C:\\case_files" --count 25
    python scripts/create_pdf_txt_holdout.py --input-dir "C:\\case_files" --count 40 --formats all
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PDF_TXT = frozenset({".pdf", ".txt"})
DEFAULT_MANIFEST = BASE_DIR / "datasets" / "evaluation_holdout" / "holdout_manifest.jsonl"
FEEDBACK_DIR = BASE_DIR / "datasets" / "feedback"
DEFAULT_MIN_WORDS = 30
# A file a person can label in one sitting; a 48,000-word document is not.
DEFAULT_MAX_WORDS = 5000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lower()


def _candidate_files(root: Path, suffixes: frozenset[str]) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: str(path).lower(),
    )


def _seen_in_feedback() -> frozenset[str]:
    """Source files of every feedback / training record - not unseen."""
    seen: set[str] = set()
    for jsonl in FEEDBACK_DIR.glob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                source = json.loads(line).get("source_file", "") if line.strip() else ""
            except json.JSONDecodeError:
                continue
            if source:
                seen.add(_norm(source))
    return frozenset(seen)


def _word_count(path: Path) -> int:
    """Words of text the tool reads from the file (0 if it can't read it)."""
    from src.core.document_factory import DocumentFactory
    result = DocumentFactory().create(path)
    if not result.success or result.document is None:
        return 0
    return len(re.findall(r"[A-Za-z]{2,}", result.document.full_text or ""))


def select_stratified(files_by_format: dict[str, list[Path]], count: int, rng: random.Random,
                      usable) -> list[Path]:
    """count files spread evenly over the formats, each shuffled with rng and
    taken in that order while usable(path) holds. A format with too few
    usable files leaves its share to the others."""
    formats = sorted(files_by_format)
    queues = {f: rng.sample(files_by_format[f], len(files_by_format[f])) for f in formats}
    quota = {f: count // len(formats) for f in formats}
    for f in rng.sample(formats, count % len(formats)):
        quota[f] += 1
    selected: list[Path] = []
    while True:
        progressed = False
        for f in formats:
            while quota[f] > 0 and queues[f]:
                path = queues[f].pop(0)
                if usable(path):
                    selected.append(path)
                    quota[f] -= 1
                    progressed = True
                    break
        short = sum(q for f, q in quota.items() if not queues[f] and q > 0)
        if short:
            # Hand an exhausted format's remaining share to formats with files left.
            for f in formats:
                if not queues[f]:
                    quota[f] = 0
            open_formats = [f for f in formats if queues[f]]
            for i in range(short):
                if open_formats:
                    quota[open_formats[i % len(open_formats)]] += 1
            progressed = progressed or bool(open_formats)
        if not progressed or sum(quota.values()) == 0:
            return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a file-separated holdout manifest")
    parser.add_argument("--input-dir", required=True, help="Folder to draw the files from")
    parser.add_argument("--count", type=int, default=25, help="How many files to reserve (default: 25)")
    parser.add_argument("--formats", default="pdf,txt",
                        help="Comma-separated extensions, or 'all' for every format the tool reads (default: pdf,txt)")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                        help=f"Skip files with fewer words of text (default: {DEFAULT_MIN_WORDS})")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                        help=f"Skip files with more words than a person can label in a sitting (default: {DEFAULT_MAX_WORDS})")
    parser.add_argument("--seed", type=int, default=20260925, help="Sampling seed recorded in the manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST, help="Manifest JSONL path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Skip files whose relative path or name matches (fnmatch), e.g. report.xml; repeatable")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    root = Path(args.input_dir).resolve()
    if not root.is_dir():
        parser.error(f"--input-dir is not a directory: {root}")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists - replacing it invalidates its labels; use --overwrite to do it anyway")

    if args.formats.strip().lower() == "all":
        from src.io.dispatcher import ReaderDispatcher
        suffixes = frozenset(ReaderDispatcher().supported_extensions())
    else:
        suffixes = frozenset("." + f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip())
    available = _candidate_files(root, suffixes)
    if not available:
        parser.error(f"No {', '.join(sorted(suffixes))} files found under --input-dir")

    seen = _seen_in_feedback()
    by_format: dict[str, list[Path]] = defaultdict(list)
    for path in available:
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat) for pat in args.exclude):
            continue
        if _norm(str(path)) not in seen:
            by_format[path.suffix.lower()].append(path)
    words: dict[Path, int] = {}
    contents: set[str] = set()

    def usable(path: Path) -> bool:
        words[path] = _word_count(path)
        if not args.min_words <= words[path] <= args.max_words:
            return False
        digest = _sha256(path)
        if digest in contents:
            return False  # the same content, saved under another path
        contents.add(digest)
        return True

    selected = select_stratified(by_format, args.count, random.Random(args.seed), usable)
    selected.sort(key=lambda path: str(path).lower())

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in selected:
            stat = path.stat()
            row = {
                "source_file": str(path),
                "relative_path": str(path.relative_to(root)),
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": stat.st_size,
                "words": words.get(path),
                "sha256": _sha256(path),
                "selection_seed": args.seed,
                "selected_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_format = defaultdict(int)
    for path in selected:
        per_format[path.suffix.lower()] += 1
    print(f"Reserved {len(selected)} of {len(available)} files for evaluation "
          f"({', '.join(f'{k} {v}' for k, v in sorted(per_format.items()))}).")
    if seen:
        print(f"Skipped files already in feedback/training data: {sum(_norm(str(p)) in seen for p in available)}")
    print(f"Manifest: {output}")
    print("Do not label these files for retraining. Label their complete person mentions separately for evaluation.")


if __name__ == "__main__":
    main()

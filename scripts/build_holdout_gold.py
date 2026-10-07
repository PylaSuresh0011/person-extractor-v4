"""Turn checked holdout markup into gold labels.

For every `<name>.marked.md` in the snapshots folder whose header says
STATUS: DONE (scripts/prepare_holdout_snapshots.py --premark writes them with
STATUS: TODO), strips the labeling note and the **markers** and writes the
`<name>_gold.json` that `cli.py --evaluate` and scripts/evaluate_holdout.py
read. A file still at TODO is skipped - an unchecked pre-mark is the
extractor's own output, and scoring it would only measure the extractor
against itself.

The text left after stripping must be the snapshot's text exactly; if a
marker was misplaced or a character edited, the file is reported and not
used, so gold offsets never point at the wrong words.

Usage:
    python scripts/build_holdout_gold.py
    python scripts/build_holdout_gold.py --snapshots-dir datasets/evaluation_holdout/snapshots
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from build_benchmark_from_markup import strip_markup  # noqa: E402

DEFAULT_SNAPSHOTS = BASE_DIR / "datasets" / "evaluation_holdout" / "snapshots"
MARKED_SUFFIX = ".marked.md"
_NOTE = re.compile(r"\A\s*<!--(.*?)-->\n?", re.DOTALL)


def read_marked(path: Path) -> tuple[str, str]:
    """(status, marked text without the note) - status 'DONE', 'TODO' or ''."""
    raw = path.read_text(encoding="utf-8")
    note = _NOTE.match(raw)
    status = ""
    if note:
        m = re.search(r"STATUS:\s*(\w+)", note.group(1))
        status = m.group(1).upper() if m else ""
        raw = raw[note.end():]
    return status, raw


def build(path: Path) -> tuple[str, int]:
    """('built' | 'todo' | 'mismatch' | 'unbalanced', gold mentions)."""
    status, marked = read_marked(path)
    if status != "DONE":
        return "todo", 0
    if marked.count("**") % 2:
        return "unbalanced", 0
    text, spans = strip_markup(marked)
    snapshot = path.with_name(path.name[: -len(MARKED_SUFFIX)] + ".txt")
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
        return "mismatch", 0
    if not snapshot.exists():
        snapshot.write_text(text, encoding="utf-8")
    gold = snapshot.with_name(snapshot.stem + "_gold.json")
    gold.write_text(json.dumps({"source_markup_file": path.name, "mention_count": len(spans),
                                "mentions": spans}, indent=2, ensure_ascii=False), encoding="utf-8")
    return "built", len(spans)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build gold labels from checked *.marked.md files")
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    args = parser.parse_args()
    counts = {"built": 0, "todo": 0, "mismatch": 0, "unbalanced": 0}
    mentions = 0
    for path in sorted(args.snapshots_dir.glob(f"*{MARKED_SUFFIX}")):
        outcome, n = build(path)
        counts[outcome] += 1
        mentions += n
        if outcome in ("mismatch", "unbalanced"):
            print(f"NOT USED {path.name}: " + ("an odd number of ** markers" if outcome == "unbalanced" else
                                               "the text differs from its snapshot - only add or remove **"))
    print(f"Gold built for {counts['built']} file(s), {mentions} mention(s); still TODO: {counts['todo']}; "
          f"not used: {counts['mismatch'] + counts['unbalanced']}.")
    return 1 if counts["mismatch"] + counts["unbalanced"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

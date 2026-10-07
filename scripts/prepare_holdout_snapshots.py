"""Create labelable TXT snapshots from a holdout manifest.

Snapshots hold exactly the text the extractor reads - for a PDF its pages
after the footer, typesetting and line-wrap fixes (Pipeline's own
prepare_pdf_text, 2026-10-01), for other formats the cleaned text - so gold
offsets generated from them match the spans scored by `cli.py --evaluate`.
They can contain sensitive source text and are ignored by git.

With --premark each snapshot also gets a `<name>.marked.md`: the same text
with every name the extractor accepts or sends to review already wrapped in
**double asterisks**, and a STATUS: TODO header. Correcting marks is much
faster than marking from scratch - but read the whole text: the marks show
only what the extractor found, and the point is to find what it missed.
scripts/build_holdout_gold.py turns a file into gold only once its header
says STATUS: DONE, so an unchecked pre-mark can never be scored as truth.

Usage:
    python scripts/prepare_holdout_snapshots.py --premark
    python scripts/prepare_holdout_snapshots.py --manifest path\\to\\holdout_manifest.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.document_factory import DocumentFactory  # noqa: E402
from src.core.models import SourceFormat  # noqa: E402
from src.knowledge.knowledge_base import KnowledgeBase  # noqa: E402
from src.pipeline.orchestrator import prepare_pdf_text  # noqa: E402
from src.preprocessing.cleaner import clean_text  # noqa: E402

DEFAULT_HOLDOUT_DIR = BASE_DIR / "datasets" / "evaluation_holdout"
DEFAULT_MANIFEST = DEFAULT_HOLDOUT_DIR / "holdout_manifest.jsonl"
MARKED_SUFFIX = ".marked.md"
STATUS_TODO = "STATUS: TODO"
LABELING_GUIDE = """<!-- STATUS: TODO    <- change TODO to DONE when this file is fully checked
How to label (person names only):
- Every mention of a person is wrapped in **double asterisks**: **Ravi Kumar**, **Dr. G. Gopal Reddy**.
- The extractor pre-marked what it found. Remove the ** around anything that is NOT a person
  (a place, a river, an organisation, a month, a title on its own), fix wrong edges
  (**Dr. G** -> **Dr. G. Gopal Reddy**), and wrap every person it MISSED.
- Read the whole text, not just the marks - missed names are what this measures.
- A first name or surname on its own is a mention too (**Ravi**), as are nicknames and initials.
- Do not change any other text. Everything above the closing line of this note is ignored.
-->
"""


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "document"


def pipeline_ready_text(document, knowledge_base: KnowledgeBase | None = None) -> str:
    if document.source_format == SourceFormat.PDF:
        text, _ = prepare_pdf_text([page.text for page in document.pages], knowledge_base)
    else:
        text = clean_text(document.full_text)
    # A literal "**" would read as a label marker ("***" needs more than one pass).
    while "**" in text:
        text = text.replace("**", "* *")
    return text


def premark(text: str, spans: list[tuple[int, int]]) -> str:
    """The text with every span wrapped in **, behind the labeling guide."""
    out, pos = [], 0
    for start, end in sorted(set(spans)):
        if start < pos or text[start - 1:start] == "*" or text[end:end + 1] == "*":
            continue  # overlapping span - keep the first; next to a "*" it would merge with it
        out += [text[pos:start], "**", text[start:end], "**"]
        pos = end
    out.append(text[pos:])
    return LABELING_GUIDE + "".join(out)


def _extractor_spans(pipeline, snapshot: Path) -> list[tuple[int, int]]:
    result = pipeline.run(str(snapshot))
    return [(m.candidate.start, m.candidate.end)
            for person in list(result.persons) + list(result.review_persons) for m in person.mentions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare holdout snapshots for manual gold labeling")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Holdout JSONL manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HOLDOUT_DIR / "snapshots",
                        help="Directory for TXT snapshots and gold labels")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing snapshots")
    parser.add_argument("--premark", action="store_true",
                        help="Also write <name>.marked.md with the extractor's names pre-marked (Fast mode)")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.exists():
        parser.error(f"Holdout manifest not found: {manifest}")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").split("\n") if line.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = DocumentFactory()
    knowledge_base = KnowledgeBase.load(BASE_DIR / "assets")
    pipeline = None
    if args.premark:
        from config import DEFAULT_CONFIG
        from src.pipeline.orchestrator import Pipeline
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["feedback"]["enabled"] = False  # holdout files never go into feedback
        cfg["llm_reviewer"]["enabled"] = False
        pipeline = Pipeline(config=cfg, base_dir=BASE_DIR)
    prepared = skipped = marked = 0

    for index, row in enumerate(rows, start=1):
        source = Path(row["source_file"])
        name = f"{index:03d}_{row.get('format', 'file')}_{_safe_stem(source.stem)[:40]}_{row['sha256'][:10]}"
        target = output_dir / f"{name}.txt"
        marked_path = output_dir / f"{name}{MARKED_SUFFIX}"
        if target.exists() and not args.overwrite:
            print(f"SKIPPED {target.name}: already exists (use --overwrite to replace)")
            skipped += 1
        else:
            result = factory.create(source)
            if not result.success or result.document is None:
                print(f"SKIPPED {source}: {result.error or 'unable to read'}")
                skipped += 1
                continue
            target.write_text(pipeline_ready_text(result.document, knowledge_base), encoding="utf-8")
            prepared += 1
            print(f"Wrote {target.name}")
        if pipeline is not None and target.exists():
            if marked_path.exists() and STATUS_TODO not in marked_path.read_text(encoding="utf-8")[:200]:
                print(f"KEPT {marked_path.name}: already checked (not STATUS: TODO)")
                continue
            text = target.read_text(encoding="utf-8")
            marked_path.write_text(premark(text, _extractor_spans(pipeline, target)), encoding="utf-8")
            marked += 1

    print(f"Prepared {prepared} snapshot(s); skipped {skipped}; pre-marked {marked}.")
    if marked:
        print(f"Next: open each *{MARKED_SUFFIX}, correct the marks, set STATUS: DONE, then run:")
        print("  python scripts/build_holdout_gold.py")
        print("  python scripts/evaluate_holdout.py")
    else:
        print("Copy one snapshot to a .md file, wrap every person mention in **double asterisks**, then run:")
        print("  python scripts/build_benchmark_from_markup.py marked.md file_name --output-dir datasets/evaluation_holdout/snapshots")


if __name__ == "__main__":
    main()

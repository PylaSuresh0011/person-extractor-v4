"""The independent-evaluation workflow (2026-10-01): stratified holdout
sampling, pre-marked labeling with a TODO/DONE gate, and the scored report."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_holdout_gold  # noqa: E402
import create_pdf_txt_holdout  # noqa: E402
import evaluate_holdout  # noqa: E402
from prepare_holdout_snapshots import STATUS_TODO, premark  # noqa: E402


def _files(fmt: str, n: int) -> list[Path]:
    return [Path(f"case/{fmt}_{i}.{fmt}") for i in range(n)]


def test_holdout_is_spread_over_formats_and_reproducible():
    by_format = {".pdf": _files("pdf", 50), ".txt": _files("txt", 50), ".docx": _files("docx", 50)}
    pick = lambda seed: create_pdf_txt_holdout.select_stratified(by_format, 9, random.Random(seed), lambda p: True)  # noqa: E731
    first = pick(7)
    assert sorted(p.suffix for p in first) == [".docx"] * 3 + [".pdf"] * 3 + [".txt"] * 3
    assert first == pick(7) and first != pick(8)


def test_a_short_format_hands_its_share_to_the_others_and_unusable_files_are_skipped():
    by_format = {".pdf": _files("pdf", 50), ".eml": _files("eml", 1)}
    unusable = {Path("case/pdf_0.pdf"), Path("case/pdf_1.pdf")}
    picked = create_pdf_txt_holdout.select_stratified(by_format, 6, random.Random(1), lambda p: p not in unusable)
    assert len(picked) == 6 and sum(p.suffix == ".eml" for p in picked) == 1
    assert not unusable & set(picked)


def _snapshot(tmp_path: Path, text: str) -> Path:
    snapshot = tmp_path / "001_txt_note_abcdef1234.txt"
    snapshot.write_text(text, encoding="utf-8")
    return snapshot


def test_premarked_file_becomes_gold_only_when_marked_done(tmp_path):
    text = "Ravi met Sita near the Godavari river."
    snapshot = _snapshot(tmp_path, text)
    marked = snapshot.with_name(snapshot.stem + build_holdout_gold.MARKED_SUFFIX)
    marked.write_text(premark(text, [(0, 4), (23, 31)]), encoding="utf-8")  # extractor: Ravi, Godavari
    assert STATUS_TODO in marked.read_text(encoding="utf-8")
    assert build_holdout_gold.build(marked) == ("todo", 0)
    # The labeler removes the river, adds the missed Sita, marks it DONE.
    checked = marked.read_text(encoding="utf-8").replace("STATUS: TODO", "STATUS: DONE")
    checked = checked.replace("**Godavari**", "Godavari").replace("met Sita", "met **Sita**")
    marked.write_text(checked, encoding="utf-8")
    assert build_holdout_gold.build(marked) == ("built", 2)
    gold = (tmp_path / "001_txt_note_abcdef1234_gold.json").read_text(encoding="utf-8")
    assert '"text": "Ravi"' in gold and '"text": "Sita"' in gold and "Godavari" not in gold


def test_a_label_that_edits_the_text_is_not_used(tmp_path):
    snapshot = _snapshot(tmp_path, "Ravi met Sita.")
    marked = snapshot.with_name(snapshot.stem + build_holdout_gold.MARKED_SUFFIX)
    marked.write_text(premark("Ravi met Sita.", []).replace("STATUS: TODO", "STATUS: DONE")
                      .replace("Ravi met Sita.", "**Ravi** met **Sitha**."), encoding="utf-8")
    assert build_holdout_gold.build(marked)[0] == "mismatch"
    marked.write_text(premark("Ravi met Sita.", []).replace("STATUS: TODO", "STATUS: DONE")
                      .replace("Ravi met Sita.", "**Ravi met Sita."), encoding="utf-8")
    assert build_holdout_gold.build(marked)[0] == "unbalanced"


def test_bootstrap_interval_is_reproducible_and_holds_the_estimate():
    counts = [(8, 1, 2), (5, 0, 1), (9, 3, 0), (2, 2, 4), (7, 1, 1)]
    ci = evaluate_holdout.bootstrap_ci(counts, samples=500)
    assert ci == evaluate_holdout.bootstrap_ci(counts, samples=500)
    p, r, f1 = evaluate_holdout.prf(sum(c[0] for c in counts), sum(c[1] for c in counts), sum(c[2] for c in counts))
    assert ci["p"][0] <= p <= ci["p"][1] and ci["f1"][0] <= f1 <= ci["f1"][1]


def test_snapshot_names_carry_their_format():
    assert evaluate_holdout.file_format("007_pdf_report_ab12cd34ef.txt") == "pdf"
    assert evaluate_holdout.file_format("benchmark_001.txt") == "unknown"

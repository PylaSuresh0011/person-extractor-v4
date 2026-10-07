"""Score the extractor on the labeled holdout - the independent test set.

Runs the full pipeline on every holdout snapshot that has gold labels
(scripts/build_holdout_gold.py) and reports, for ACCEPTED only and for
ACCEPTED + REVIEW:
  - mention precision, recall and F1, with 95% confidence intervals from a
    bootstrap over files (a file is resampled, not a mention: mentions in
    one file are not independent);
  - the same per file format (pdf, txt, docx, html, ...);
  - recall per name shape (single vs multi-token, a name the lists know vs
    an unseen one) - as `cli.py --evaluate` measures it.

Every run is recorded with what produced it (README "reproducible scores"):
a hash of the config, of the model file, of the rules and word lists
(assets/), of the code (src/), of the manifest and of the gold labels - so a
later score can be compared fairly, or found not comparable. Numbers only go
to experiments/holdout_results/<time>_<label>.json and a line of
experiments/holdout_ledger.csv; the names (false positives, misses per file)
go to output/holdout/, which is never committed.

Holdout files are never logged as feedback (feedback is off here) and are
excluded from retraining by the manifest (scripts/retrain_from_feedback.py).

Usage:
    python scripts/evaluate_holdout.py --label baseline
    python scripts/evaluate_holdout.py --label llm --llm
    python scripts/evaluate_holdout.py --label after-fix --compare experiments/holdout_results/<earlier>.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG  # noqa: E402

DEFAULT_SNAPSHOTS = BASE_DIR / "datasets" / "evaluation_holdout" / "snapshots"
DEFAULT_MANIFEST = BASE_DIR / "datasets" / "evaluation_holdout" / "holdout_manifest.jsonl"
RESULTS_DIR = BASE_DIR / "experiments" / "holdout_results"
LEDGER = BASE_DIR / "experiments" / "holdout_ledger.csv"
DETAILS_DIR = BASE_DIR / "output" / "holdout"
BOOTSTRAP_SAMPLES = 2000
LEDGER_COLUMNS = ["time_utc", "label", "mode", "files", "gold_mentions", "accepted_p", "accepted_r", "accepted_f1",
                  "accepted_f1_ci", "all_p", "all_r", "all_f1", "config", "model", "assets", "code", "manifest", "gold"]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def bootstrap_ci(counts: list[tuple[int, int, int]], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> dict:
    """95% intervals of micro P/R/F1 when files are resampled with replacement."""
    if not counts:
        return {}
    rng = random.Random(seed)
    draws = {"p": [], "r": [], "f1": []}
    for _ in range(samples):
        pick = [counts[rng.randrange(len(counts))] for _ in counts]
        p, r, f = prf(sum(c[0] for c in pick), sum(c[1] for c in pick), sum(c[2] for c in pick))
        draws["p"].append(p), draws["r"].append(r), draws["f1"].append(f)
    out = {}
    for k, values in draws.items():
        values.sort()
        out[k] = [values[int(0.025 * (samples - 1))], values[int(0.975 * (samples - 1))]]
    return out


def file_format(snapshot_name: str) -> str:
    """'007_pdf_report_ab12cd34ef.txt' -> 'pdf' (scripts/prepare_holdout_snapshots.py names)."""
    parts = snapshot_name.split("_")
    return parts[1] if len(parts) > 2 and parts[0].isdigit() else "unknown"


def _sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(BASE_DIR) if path.is_relative_to(BASE_DIR) else path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def fingerprint(config: dict, snapshots: Path, manifest: Path, provenance: dict) -> dict:
    """What produced a score - hashes, so two runs can be compared fairly."""
    return {
        "config": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "model": provenance.get("model_sha256_short") or "n/a",
        "assets": _sha([p for p in (BASE_DIR / "assets").rglob("*.txt") if not p.name.endswith(".local.txt")]),
        "code": _sha(list((BASE_DIR / "src").rglob("*.py"))),
        "manifest": _sha([manifest]) if manifest.exists() else "n/a",
        "gold": _sha(list(snapshots.glob("*_gold.json"))),
    }


def summarise(report) -> dict:
    """Overall, per format and per shape, with bootstrap intervals."""
    out: dict = {"files": len(report.file_results)}
    for scope, attr in (("accepted", "accepted_metrics"), ("all", "candidate_metrics")):
        counts = [(getattr(f, attr).true_positives, getattr(f, attr).false_positives,
                   getattr(f, attr).false_negatives) for f in report.file_results]
        tp, fp, fn = (sum(c[i] for c in counts) for i in range(3))
        p, r, f1 = prf(tp, fp, fn)
        by_format: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for f, c in zip(report.file_results, counts):
            row = by_format[file_format(f.name)]
            row[0] += c[0]; row[1] += c[1]; row[2] += c[2]; row[3] += 1
        out[scope] = {
            "p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "ci95": bootstrap_ci(counts),
            "by_format": {fmt: dict(zip(("p", "r", "f1"), prf(*v[:3])), files=v[3], gold=v[0] + v[2])
                          for fmt, v in sorted(by_format.items())},
        }
    out["gold_mentions"] = out["all"]["tp"] + out["all"]["fn"]
    out["shape_recall_accepted"] = {k: {"recall": round(v.hits / v.total, 4) if v.total else 0.0, "of": v.total}
                                    for k, v in report.shape_recall_accepted.items()}
    out["shape_recall_all"] = {k: {"recall": round(v.hits / v.total, 4) if v.total else 0.0, "of": v.total}
                               for k, v in report.shape_recall_candidate.items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Name of this run, e.g. baseline or after-fix-3")
    parser.add_argument("--llm", action="store_true", help="With the LLM reviewer (default: Fast mode)")
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compare", type=Path, default=None, help="An earlier results JSON to show the change against")
    args = parser.parse_args()

    from src.evaluation.evaluator import run_benchmark
    from src.pipeline.orchestrator import Pipeline
    from src.utils.logger import get_logger
    get_logger("evaluate_holdout")  # logger set-up
    import logging
    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)

    labelled = sorted(args.snapshots_dir.glob("*_gold.json"))
    total = len(list(args.snapshots_dir.glob("*.txt")))
    if not labelled:
        print(f"No gold labels in {args.snapshots_dir} yet ({total} snapshot(s)) - label some first "
              "(scripts/prepare_holdout_snapshots.py --premark, then scripts/build_holdout_gold.py).")
        return 1
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    if not args.llm:
        cfg["llm_reviewer"]["enabled"] = False
    pipeline = Pipeline(config=cfg, base_dir=BASE_DIR)
    report = run_benchmark(pipeline, args.snapshots_dir)
    summary = summarise(report)
    now = datetime.now(timezone.utc)
    result = {"label": args.label, "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "mode": "LLM reviewer" if args.llm else "Fast", "labelled_files": len(labelled), "snapshots": total,
              "fingerprint": fingerprint(cfg, args.snapshots_dir, args.manifest, pipeline.extractor.provenance),
              **summary}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{now:%Y%m%dT%H%M%SZ}_{args.label}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    details = DETAILS_DIR / f"{now:%Y%m%dT%H%M%SZ}_{args.label}_details.csv"
    with details.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "kind", "text"])
        for f in report.file_results:
            w.writerows([[f.name, "false positive (accepted)", t] for t in f.false_positive_texts])
            w.writerows([[f.name, "missed (not even in review)", t] for t in f.missed_mentions])
    new_ledger = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_ledger:
            w.writerow(LEDGER_COLUMNS)
        a, c, fp = result["accepted"], result["all"], result["fingerprint"]
        w.writerow([result["time_utc"], args.label, result["mode"], len(labelled), result["gold_mentions"],
                    a["p"], a["r"], a["f1"], "-".join(map(str, a["ci95"].get("f1", []))), c["p"], c["r"], c["f1"],
                    fp["config"], fp["model"], fp["assets"], fp["code"], fp["manifest"], fp["gold"]])

    print(f"{result['mode']}: {len(labelled)} labelled of {total} holdout files, {result['gold_mentions']} gold mentions")
    for scope in ("accepted", "all"):
        s = result[scope]
        ci = s["ci95"]
        print(f"  {scope:8} P {s['p']:.3f} [{ci['p'][0]:.3f}-{ci['p'][1]:.3f}]  R {s['r']:.3f} "
              f"[{ci['r'][0]:.3f}-{ci['r'][1]:.3f}]  F1 {s['f1']:.3f} [{ci['f1'][0]:.3f}-{ci['f1'][1]:.3f}]")
        for fmt, m in s["by_format"].items():
            print(f"      {fmt:6} {m['files']:3} file(s) {m['gold']:5} gold  P {m['p']:.3f}  R {m['r']:.3f}  F1 {m['f1']:.3f}")
    print("  recall by name shape (accepted): " + ", ".join(
        f"{k} {v['recall']:.3f} of {v['of']}" for k, v in result["shape_recall_accepted"].items()))
    if args.compare and args.compare.exists():
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        same = {k: before.get("fingerprint", {}).get(k) == v for k, v in result["fingerprint"].items()}
        print(f"  vs {before.get('label')}: accepted F1 {before['accepted']['f1']:.3f} -> {result['accepted']['f1']:.3f}, "
              f"all F1 {before['all']['f1']:.3f} -> {result['all']['f1']:.3f}")
        if not same["gold"] or not same["manifest"]:
            print("  NOTE: the gold labels or the manifest changed since then - not a like-for-like comparison")
    print(f"Saved {out}\nNames (local only): {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

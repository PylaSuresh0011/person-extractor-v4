"""
experiments/regression_check.py
==================================
The existing accuracy checks in one quick run (Fast mode, no LLM reviewer),
so a fix for one document can't quietly make the others worse:
benchmark (7 files), the held-out unseen-names file, and the people-free
system dump (benchmark_structured_dump_005, where every accepted name is a
false positive).

    python experiments/regression_check.py [--label NAME]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, DEFAULT_CONFIG  # noqa: E402
from src.evaluation.evaluator import run_benchmark  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402


def check() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["enabled"] = False
    pipeline = Pipeline(config=cfg, base_dir=BASE_DIR)
    out = {}
    for name in ("benchmark", "benchmark_unseen"):
        report = run_benchmark(pipeline, BASE_DIR / "datasets" / name)
        a, c = report.overall_accepted, report.overall_candidate
        out[name] = {"accepted_P": round(a.precision, 4), "accepted_R": round(a.recall, 4),
                     "accepted_FP": a.false_positives, "all_R": round(c.recall, 4), "all_FP": c.false_positives}
        if name == "benchmark":
            dump = [f for f in report.file_results if "dump_005" in f.name]
            out["dump005_accepted_FP"] = dump[0].accepted_metrics.false_positives if dump else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)
    result = check()
    for name in ("benchmark", "benchmark_unseen"):
        r = result[name]
        print(f"{name:17} accepted P {r['accepted_P']:.4f} R {r['accepted_R']:.4f} FP {r['accepted_FP']:>3} | "
              f"accepted+review R {r['all_R']:.4f} FP {r['all_FP']}")
    print(f"system dump: accepted FP {result['dump005_accepted_FP']}")
    if args.label:
        out = BASE_DIR / "experiments" / "probes" / f"regression_{args.label}.json"
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

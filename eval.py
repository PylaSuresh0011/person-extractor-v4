"""
eval.py
=======
Scores the extractor on a labelled PDF - by default the Class X English
textbook against tests/fixtures/10th_eng_gold.csv and its labelled key,
datasets/textbook_eval/10th_eng_comparison.csv. The work is done by
experiments/textbook_eval.py; this is the short entry point.

    python eval.py                      # Fast mode (no LLM reviewer)
    python eval.py --llm                # with the LLM reviewer
    python eval.py --details            # also list non-names, truncated and missed names
    python eval.py --pdf "C:\\path\\to\\other.pdf"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from textbook_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

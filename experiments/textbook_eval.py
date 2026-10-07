"""
experiments/textbook_eval.py
===============================
Scores the pipeline on three school textbooks (Telangana SCERT) against
hand-labelled keys - per book and macro-averaged:

  english   10th eng.pdf           (274 pages, 279 people)  round 1, 2026-10-01
  social    10th social em.pdf     (346 pages, 234 people)  round 2, 2026-10-01
  english9  9 english-2024-25.pdf  (194 pages, 123 people)  2026-10-06

Each book has a key - every entry of one labelled run, marked correct /
partial-or-truncated / not a person, plus the people that run missed
(datasets/textbook_eval/<class>_<book>_comparison.csv) - and a gold list of
the people in it (tests/fixtures/<class>_<book>_gold.csv: canonical_name,
aliases; the older reference_name column is read too). Key rows with the
decision "other_run" label entries of later runs; --key-only ignores them.
The PDFs are not in the project (the owner's Downloads copies).

Metrics, as the error briefs define them (docs/PERSON_EXTRACTOR_improvement_
brief*.md), for ACCEPTED only and for ACCEPTED + REVIEW:
  precision     - share of entries that are a correct, complete person name
  recall        - share of the gold people found by at least one correct,
                  complete entry (a truncated one doesn't count)
  F1, truncated entries, non-names, duplicates (entries beyond the first
  for the same person), review junk (share of REVIEW entries that are not a
  correct name)

How an entry is labelled: its exact text in the key if present; otherwise it
is matched to a gold person - equal after normalising (titles, case,
punctuation, accents) to the name or an alias, within 2 edits, or initials
for the given names = correct; its words a strict part of one person's
name, or that name plus extra words = partial; anything else is
"unlabelled", counted as a non-name and listed so a person can label it.

    python eval.py                       # both books, Fast mode
    python eval.py --book social --llm   # one book, with the LLM reviewer
    python eval.py --key-only            # score the keys' own runs (the briefs' numbers)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, DEFAULT_CONFIG  # noqa: E402

DOWNLOADS = Path(r"C:\Users\Suresh\Downloads")
BOOKS = {
    "english": {"pdf": DOWNLOADS / "10th eng.pdf",
                "key": BASE_DIR / "datasets" / "textbook_eval" / "10th_eng_comparison.csv",
                "gold": BASE_DIR / "tests" / "fixtures" / "10th_eng_gold.csv"},
    "social": {"pdf": DOWNLOADS / "10th social em.pdf",
               "key": BASE_DIR / "datasets" / "textbook_eval" / "10th_social_comparison.csv",
               "gold": BASE_DIR / "tests" / "fixtures" / "10th_social_gold.csv"},
    "english9": {"pdf": DOWNLOADS / "9 english-2024-25.pdf",
                 "key": BASE_DIR / "datasets" / "textbook_eval" / "9th_eng_comparison.csv",
                 "gold": BASE_DIR / "tests" / "fixtures" / "9th_eng_gold.csv"},
}
# Kept for older callers: the English book.
KEY = BOOKS["english"]["key"]
DEFAULT_PDF = BOOKS["english"]["pdf"]
HONORIFICS = {"mr", "mrs", "ms", "dr", "sri", "smt", "prof", "professor", "fr", "sir", "colonel", "lord"}
SCOPES = ("accepted", "accepted_plus_review")


def norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()  # Lemière = Lemiere
    words = re.sub(r"[.,'’\-()]", " ", name.lower()).split()
    return " ".join(w for w in words if w not in HONORIFICS)


def _split_reference(reference: str) -> tuple[str, list[str]]:
    """'Jawaharlal Nehru (Nehru; J. Nehru)' -> ('Jawaharlal Nehru', ['Nehru',
    'J. Nehru']); a lowercase note in brackets ('Ranga Rao (translator)') is
    no alias."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", reference.strip())
    if not m:
        return reference.strip(), []
    aliases = [a.strip() for a in re.split(r"[;|]", m.group(2)) if a.strip()[:1].isupper()]
    return m.group(1).strip(), aliases


def _forms(reference: str) -> list[str]:
    canonical, aliases = _split_reference(reference)
    return [norm(canonical)] + [norm(a) for a in aliases]


def _same_surname(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) > 4 and _edits(a, b) <= 2)


def _edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Key:
    def __init__(self, path: Path = KEY, gold: Path | None = None) -> None:
        rows = list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []
        # The people: from the gold list when there is one, else the key's references.
        self.forms: dict[str, list[str]] = {}
        gold_rows = list(csv.DictReader(open(gold, encoding="utf-8"))) if gold and gold.exists() else []
        for g in gold_rows:
            if g.get("canonical_name"):
                aliases = [a.strip() for a in re.split(r"[;|]", g.get("aliases") or "") if a.strip()]
                self.forms[g["canonical_name"].strip()] = [norm(g["canonical_name"])] + [norm(a) for a in aliases]
            elif g.get("reference_name"):
                canonical, _ = _split_reference(g["reference_name"])
                self.forms[canonical] = _forms(g["reference_name"])
        if not gold_rows:
            for r in rows:
                ref = (r.get("matches_reference_name") or "").strip()
                if ref:
                    self.forms.setdefault(_split_reference(ref)[0], _forms(ref))
        self.references = set(self.forms)
        self.labels: dict[str, tuple[str, str]] = {}
        self.key_entries: list[dict] = []
        for r in rows:
            if r["verdict"].startswith("MISSED"):
                continue
            verdict = {"correct name": "correct", "partial/truncated name": "partial",
                       "partial/truncated/ambiguous name": "partial"}.get(r["verdict"], "not_name")
            ref = (r.get("matches_reference_name") or "").strip()
            self.labels[r["extractor_name"].strip()] = (verdict, self._reference_of(ref) if ref else "")
            self.key_entries.append({"display": r["extractor_name"], "occurrences": int(r["count"]),
                                     "decision": r["extractor_decision"]})

    def _reference_of(self, ref: str) -> str:
        """The gold person a key's reference string names."""
        canonical, _ = _split_reference(ref)
        if canonical in self.forms:
            return canonical
        wanted = set(_forms(ref))
        for person, forms in self.forms.items():
            if wanted & set(forms):
                return person
        return canonical

    def label(self, name: str) -> tuple[str, str]:
        """(verdict, reference) - verdict: correct / partial / not_name / unlabelled."""
        name = name.strip()
        if name in self.labels:
            return self.labels[name]
        n = norm(name)
        if not n:
            return ("unlabelled", "")
        words = set(n.split())
        exact = [ref for ref, fs in self.forms.items() if n in fs]
        if exact:
            return ("correct", exact[0])
        close = [ref for ref, fs in self.forms.items() if any(_edits(n, f) <= 2 for f in fs if len(f) > 4)]
        if len(close) == 1:
            return ("correct", close[0])
        # Initials for given names: "R.K. Agnihotri" = "Rama Kant Agnihotri",
        # "A.L. Hendricks" = "Arthur Lemière Hendriks" (surname within 2 edits).
        toks = n.split()
        initialled = [ref for ref, fs in self.forms.items() for f in fs
                      if len(f.split()) == len(toks) > 1 and _same_surname(f.split()[-1], toks[-1])
                      and all(t == g or (len(t) == 1 and g.startswith(t)) for t, g in zip(toks[:-1], f.split()[:-1]))]
        if len(set(initialled)) == 1:
            return ("correct", initialled[0])
        part = [ref for ref, fs in self.forms.items()
                if any(words < set(f.split()) or set(f.split()) < words for f in fs)]
        if len(set(part)) == 1:
            return ("partial", part[0])
        return ("unlabelled", "")


def score(entries: list[dict], key: Key) -> dict:
    out = {}
    for scope, keep in (("accepted", {"accepted"}), ("accepted_plus_review", {"accepted", "review"})):
        rows = [dict(e, **dict(zip(("verdict", "reference"), key.label(e["display"]))))
                for e in entries if e["decision"] in keep]
        verdicts = Counter(r["verdict"] for r in rows)
        # As the briefs count them: a person is found only by a correct,
        # complete entry, and duplicates are extra correct entries for one person.
        correct_refs = [r["reference"] for r in rows if r["verdict"] == "correct" and r["reference"] in key.forms]
        found = set(correct_refs)
        per_person = Counter(correct_refs)
        p = verdicts["correct"] / len(rows) if rows else 0.0
        rec = len(found) / len(key.references) if key.references else 0.0
        review = [r for r in rows if r["decision"] == "review"]
        out[scope] = {
            "entries": len(rows), "correct": verdicts["correct"], "partial": verdicts["partial"],
            "not_name": verdicts["not_name"], "unlabelled": verdicts["unlabelled"],
            "precision": round(p, 4), "recall": round(rec, 4),
            "f1": round(2 * p * rec / (p + rec), 4) if p + rec else 0.0,
            "people_found": len(found), "people_total": len(key.references),
            "duplicates": sum(n - 1 for n in per_person.values()),
            "review_entries": len(review),
            "review_junk": round(sum(r["verdict"] != "correct" for r in review) / len(review), 4) if review else 0.0,
            "top_non_names": [f"{r['display']} ({r['occurrences']})" for r in
                              sorted((r for r in rows if r["verdict"] in ("not_name", "unlabelled")),
                                     key=lambda r: -r["occurrences"])[:20]],
            "unlabelled_names": sorted(r["display"] for r in rows if r["verdict"] == "unlabelled"),
            "truncated": sorted(r["display"] for r in rows if r["verdict"] == "partial"),
            "missed_people": sorted(key.references - found),
        }
    return out


def macro(books: dict[str, dict]) -> dict:
    """Precision, recall, F1 and review junk averaged over the books; counts summed."""
    out = {}
    for scope in SCOPES:
        per = [b[scope] for b in books.values()]
        avg = {k: round(sum(s[k] for s in per) / len(per), 4) for k in ("precision", "recall", "f1", "review_junk")}
        tot = {k: sum(s[k] for s in per) for k in ("partial", "not_name", "unlabelled", "duplicates", "entries")}
        out[scope] = {**avg, **tot}
    return out


def run_pipeline(pdf: Path, llm: bool) -> tuple[list[dict], float]:
    import gui  # the window's own combining, so the numbers match what it shows
    from src.pipeline.orchestrator import Pipeline
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    if not llm:
        cfg["llm_reviewer"]["enabled"] = False
    started = time.perf_counter()
    result = Pipeline(config=cfg, base_dir=BASE_DIR).run(str(pdf))
    rows = gui.combine([(pdf.name, result)])
    return ([{"display": e["display"], "occurrences": e["occurrences"], "decision": e["decision"]} for e in rows],
            time.perf_counter() - started)


def _print_row(name: str, s: dict) -> None:
    print(f"  {name:22}{s['precision']:>8.1%}{s['recall']:>8.1%}{s['f1']:>8.1%}{s['partial']:>7}"
          f"{s['not_name'] + s['unlabelled']:>9}{s['duplicates']:>7}{s['review_junk']:>8.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", choices=[*BOOKS, "all"], default="all")
    parser.add_argument("--llm", action="store_true", help="With the LLM reviewer (default: Fast mode)")
    parser.add_argument("--key-only", action="store_true", help="Score the keys' own runs instead of new ones")
    parser.add_argument("--label", default=None, help="Save the scores as experiments/probes/textbook_eval_<label>.json")
    parser.add_argument("--details", action="store_true", help="Print the non-names, truncated and missed lists")
    args = parser.parse_args()
    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)

    mode = "key's own run" if args.key_only else ("LLM reviewer" if args.llm else "Fast")
    results: dict[str, dict] = {}
    for book in (BOOKS if args.book == "all" else [args.book]):
        paths = dict(BOOKS[book])
        for kind in ("key", "gold"):
            stand_in = paths[kind].with_suffix(".PROVISIONAL.csv")
            if not paths[kind].exists() and stand_in.exists():
                # Built from a partial paste of the labelled run (2026-10-01) until
                # the real file is copied in - recall is approximate.
                print(f"{book}: PROVISIONAL {kind} - {stand_in.name}")
                paths[kind] = stand_in
        if not paths["key"].exists() or not paths["gold"].exists():
            print(f"{book}: skipped - missing {paths['key'] if not paths['key'].exists() else paths['gold']}")
            continue
        key = Key(paths["key"], paths["gold"])
        if args.key_only:
            entries, seconds = key.key_entries, 0.0
        else:
            entries, seconds = run_pipeline(paths["pdf"], args.llm)
        results[book] = score(entries, key)
        results[book]["seconds"] = round(seconds, 1)
        print(f"{book}: {len(entries)} entries, {len(key.references)} gold people"
              + (f", {seconds:.0f}s" if seconds else ""))

    if not results:
        return 1
    print(f"\n{mode}{'':16}{'P':>8}{'R':>8}{'F1':>8}{'trunc':>7}{'nonname':>9}{'dupes':>7}{'revjunk':>8}")
    for scope in SCOPES:
        for book, r in results.items():
            _print_row(f"{book} {scope.replace('accepted_plus_review', 'all rows')}", r[scope])
        if len(results) > 1:
            _print_row(f"MACRO {scope.replace('accepted_plus_review', 'all rows')}", macro(results)[scope])
    if args.details:
        for book, r in results.items():
            for scope in SCOPES:
                print(f"\n[{book} {scope}] top non-names: {r[scope]['top_non_names']}")
                print(f"[{book} {scope}] unlabelled ({len(r[scope]['unlabelled_names'])}): {r[scope]['unlabelled_names']}")
            print(f"[{book}] missed ({len(r['accepted_plus_review']['missed_people'])}): "
                  f"{r['accepted_plus_review']['missed_people']}")
    if args.label:
        saved = {"mode": mode, "books": results, "macro": macro(results) if len(results) > 1 else None}
        out = BASE_DIR / "experiments" / "probes" / f"textbook_eval_{args.label}.json"
        out.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

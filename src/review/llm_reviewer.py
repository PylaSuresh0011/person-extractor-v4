"""
src.review.llm_reviewer
=========================
Optional final check of each FILE's names by a local LLM, through Ollama
(http://127.0.0.1:11434 - nothing leaves the machine). The tool never
installs or downloads anything: scripts/setup_llm_reviewer.ps1 is the
one-time setup, run by hand with internet, and only on a machine with a GPU.

Measured before it was built (experiments/llm_reviewer_probe.py, 2026-09-28,
Llama 3.1 8B): of the ACCEPTED names in a real case scan, 13 of the 14 the
model gave P(person) < 0.3 were not people (companies, places, interface
labels, OCR scraps); on the benchmark the same move removed one false
positive and nothing real. With GPU+CPU a name takes 0.41 s, on CPU only
4.25 s.

Two ways to use the model's answer:

  With the score gate (config decision.feature_score_gate, the default since
  round 3, 2026-10-01): the model reads the sentences around up to three
  mentions and says what the entity is - PERSON, PLACE, ORG, WORK, EVENT,
  TIME or OTHER - with few-shot examples of the textbooks' traps (a river
  named Ravi, the Meira Paibi movement, the month Karthika, "Mr. President").
  Its P(PERSON) moves the name's final score by score_weight x (2p - 1) -
  score_weight_down when p < 0.5, so a confident "not a person" can drop a
  name and not just send it to review (measured on both textbooks: review
  went from 21 and 43 rows, 62% and 95% junk, to 10 and 8);
  the decision is then read off the score like any other
  (src/decision/feature_score.py). Every decision it changes is an
  override: logged, put in the evidence trail and in the run's
  model_info["llm_overrides"], with the class, P(PERSON) and old -> new score.

  Without it (the rules decide): ACCEPTED -> REVIEW below `demote_below`
  (it never rejects anything), and REVIEW -> ACCEPTED at or above
  `promote_at_or_above` only when that is switched on (off by default - an
  owner decision).

Names in assets/owner_decisions/confirmed_names.txt are never lowered. Every
verdict goes into the evidence trail with the model, its digest and the
prompt version.

Active only when all of these hold - otherwise the pipeline runs exactly as
without it and the reason is recorded in the run's provenance:
  - config llm_reviewer.enabled is "auto" or True (and the environment
    variable PERSON_EXTRACTOR_LLM_REVIEWER isn't "off" - the test suite
    sets it),
  - Ollama answers (if it's installed but not running, it is started),
  - the pinned model digest is installed,
  - the model runs on a GPU (Ollama's /api/ps reports GPU memory) - on a
    machine without one it is skipped, as it would be 10x slower.
Live text (Pipeline.run_text, realtime sessions) is never reviewed - see
PersonExtractor.extract.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.core.models import CandidateResult, Decision
from src.decision.feature_score import REVIEW_AT, decision_for
from src.detection.regex_detector import CITATION_CUE
from src.utils.logger import get_logger

logger = get_logger("review.llm_reviewer")

PROMPT_VERSION = "v2"
SYSTEM_PROMPT = (
    "You check the output of a person-name extractor that runs on forensic evidence: chats, emails, "
    "documents, web pages, app data and OCR text. For the candidate, decide whether the excerpts use it "
    "as the name of a specific human being - a first name, a surname, a full name, or a name with a "
    "title or initials. Answer only 'yes' or 'no'. Answer 'no' for organizations, companies, brands, "
    "products, apps, software and code identifiers; places, streets, localities, countries, languages "
    "and nationalities; job titles; days and months; journal and publication titles; ordinary words; "
    "interface labels; and garbled text."
)
CLASSES = ("PERSON", "PLACE", "ORG", "WORK", "EVENT", "TIME", "OTHER")
CLASS_SYSTEM_PROMPT = (
    "You check the output of a person-name extractor. Read the sentences and say what the candidate "
    "[[in double brackets]] refers to there. Answer with exactly one word:\n"
    "PERSON - a specific human being, real or fictional, also a god or a character named like a person\n"
    "PLACE - a river, mountain, lake, region, district, town, village, country or building\n"
    "ORG - an organisation, party, movement, programme, company, institution or group of people\n"
    "WORK - a book, film, poem, song, newspaper or other title\n"
    "EVENT - a war, revolt, festival, agreement or other event\n"
    "TIME - a month, season, calendar, day or era\n"
    "OTHER - a title or role on its own, an ordinary word, a language or anything else"
)
# Few-shot examples - the traps of the round-2 error brief (N11). None is a
# person from the evaluation's gold lists.
FEW_SHOT = (
    ("Ravi", ["Five rivers flow through the plains: the Jhelum, the Chenab, the [[Ravi]], the Beas and the "
              "Sutlej."], "PLACE"),
    ("Meira Paibi", ["The women of the [[Meira Paibi]] Movement held torches and guarded the streets at night."],
     "ORG"),
    ("Karthika", ["Sharad Aswayuja and [[Karthika]] are the months of the autumn season in the Telugu calendar."],
     "TIME"),
    ("Mr. President", ["[[Mr. President]], Sir, I introduce the Draft Constitution as settled by the committee."],
     "OTHER"),
    ("Rangaiah", ["[[Rangaiah]], a farmer from Warangal, said that the rains came late this year.",
                  "When the loan was refused, [[Rangaiah]] sold his two bullocks."], "PERSON"),
    ("Godan", ["Premchand's novel [[Godan]] describes the life of a poor peasant."], "WORK"),
)
ENV_SWITCH = "PERSON_EXTRACTOR_LLM_REVIEWER"
MAX_EXCERPTS = 3
EXCERPT_WIDTH = 120
SENTENCE_MAX_CHARS = 220
DEFAULT_SCORE_WEIGHT = 0.3
DEFAULT_SCORE_WEIGHT_DOWN = 0.8
# Set by _cap_lone_single_tokens; such names aren't promoted - the model
# sees no more context there than the rules did.
_LONE_WORD_REASON_PREFIX = "Single word in"
_SENTENCE_END = re.compile(r"[.!?][\"'”’)]?\s+(?=[\"'“‘(]?[A-Z0-9])|\n\s*\n")


def excerpt(text: str, start: int, end: int, width: int = EXCERPT_WIDTH) -> str:
    """The mention with up to `width` characters either side, within its line,
    marked [[like this]]."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    a, b = max(line_start, start - width), min(line_end, end + width)
    return f"{text[a:start]}[[{text[start:end]}]]{text[end:b]}".strip()


def sentence_context(text: str, start: int, end: int, max_chars: int = SENTENCE_MAX_CHARS) -> str:
    """The sentence holding the mention, line breaks joined, at most
    `max_chars` characters either side, the mention marked [[like this]]."""
    window_start = max(0, start - max_chars)
    a = window_start
    for m in _SENTENCE_END.finditer(text, window_start, start):
        a = m.end()
    m = _SENTENCE_END.search(text, end, min(len(text), end + max_chars))
    b = m.start() + 1 if m else min(len(text), end + max_chars)
    flat = lambda s: " ".join(s.split())  # noqa: E731
    return f"{flat(text[a:start])} [[{text[start:end]}]] {flat(text[end:b])}".strip()


def _request(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _start_ollama(url: str) -> bool:
    """Starts an installed-but-stopped local Ollama server. Never installs."""
    exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
    if not Path(exe).exists():
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    for _ in range(30):
        time.sleep(1)
        try:
            _request(f"{url}/api/version", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            continue
    return False


@dataclass(frozen=True)
class ReviewerStatus:
    active: bool
    reason: str


def resolve_status(settings: dict) -> ReviewerStatus:
    """Whether the reviewer can run here - see the module docstring."""
    if os.environ.get(ENV_SWITCH, "").lower() == "off":
        return ReviewerStatus(False, f"off ({ENV_SWITCH}=off)")
    enabled = settings.get("enabled", "auto")
    if enabled is False:
        return ReviewerStatus(False, "off (disabled in config)")
    url, model, digest = settings["url"], settings["model"], settings["model_digest"]
    try:
        _request(f"{url}/api/version", timeout=2)
    except (urllib.error.URLError, OSError):
        if not _start_ollama(url):
            return ReviewerStatus(False, "off (Ollama is not installed or would not start)")
    try:
        installed = {m["name"]: m["digest"] for m in _request(f"{url}/api/tags")["models"]}
        if model not in installed:
            return ReviewerStatus(False, f"off (model {model} is not installed)")
        if installed[model] != digest:
            return ReviewerStatus(False, f"off (model {model} is {installed[model][:12]}, pinned {digest[:12]})")
        for attempt in range(2):
            _request(f"{url}/api/generate", {"model": model, "prompt": "", "keep_alive": "30m"}, timeout=600)
            loaded = [m for m in _request(f"{url}/api/ps")["models"] if m["name"] == model]
            if loaded and loaded[0].get("size_vram", 0) > 0:
                return ReviewerStatus(True, f"on ({model} {digest[:12]}, prompt {PROMPT_VERSION}, GPU)")
            # Loaded CPU-only (no usable GPU, or an earlier CPU-only request) -
            # unload once and let Ollama place it again.
            _request(f"{url}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        return ReviewerStatus(False, "off (no usable GPU - the model would run on the CPU only)")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        return ReviewerStatus(False, f"off (Ollama error: {exc})")


def _cited(group: list[CandidateResult]) -> bool:
    """A cited author (the regex detector's citation cue) is never lowered
    (2026-10-12). The layout of a reference list names its people exactly;
    the model, given "Mueser, K., [[Yarnold]] , P. & Bellack, A.", answered
    ORG 0.84 - on a case scan it dropped 24 hand-read names this way, nearly
    all of them cited authors."""
    return any(d.metadata.get("cue") == CITATION_CUE for c in group for d in c.candidate.source_detections)


def _load_names(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def _class_messages(name: str, sentences: list[str]) -> list[dict]:
    def ask(n: str, ss: list[str]) -> str:
        return f"Candidate: {n}\nSentences:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(ss, 1))
    messages = [{"role": "system", "content": CLASS_SYSTEM_PROMPT}]
    for example_name, example_sentences, label in FEW_SHOT:
        messages += [{"role": "user", "content": ask(example_name, example_sentences)},
                     {"role": "assistant", "content": label}]
    return messages + [{"role": "user", "content": ask(name, sentences)}]


class LlmReviewer:
    def __init__(self, settings: dict, assets_dir: Path) -> None:
        self.url = settings["url"]
        self.model = settings["model"]
        self.digest = settings["model_digest"]
        self.demote_below = settings.get("demote_below")
        self.promote_at_or_above = settings.get("promote_at_or_above")
        self.score_weight = settings.get("score_weight", DEFAULT_SCORE_WEIGHT)
        self.score_weight_down = settings.get("score_weight_down", DEFAULT_SCORE_WEIGHT_DOWN)
        self.confirmed_names = _load_names(assets_dir / "owner_decisions" / "confirmed_names.txt")
        self.failed: str | None = None
        self.overrides: list[dict] = []

    def p_person(self, name: str, excerpts: list[str]) -> float:
        user = (f"Candidate: {name}\nExcerpts:\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(excerpts, 1))
                + f"\nIs [[{name}]] used as a person's name here?")
        reply = _request(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "logprobs": True, "top_logprobs": 10, "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_predict": 1},
        }, timeout=600)
        yes = no = 0.0
        for cand in reply["logprobs"][0]["top_logprobs"]:
            token = cand["token"].strip().lower()
            if token in ("yes", "y"):
                yes += math.exp(cand["logprob"])
            elif token in ("no", "n"):
                no += math.exp(cand["logprob"])
        return yes / (yes + no) if yes + no else 0.5

    def classify(self, name: str, sentences: list[str]) -> dict[str, float]:
        """{class: probability} over CLASSES, from the first answer token."""
        reply = _request(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "logprobs": True, "top_logprobs": 20, "keep_alive": "30m",
            "messages": _class_messages(name, sentences),
            "options": {"temperature": 0, "seed": 0, "num_predict": 1},
        }, timeout=600)
        probs = dict.fromkeys(CLASSES, 0.0)
        for cand in reply["logprobs"][0]["top_logprobs"]:
            token = cand["token"].strip().upper()
            match = [c for c in CLASSES if len(token) >= 2 and c.startswith(token)]
            if len(match) == 1:
                probs[match[0]] += math.exp(cand["logprob"])
        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()} if total else {}

    def review(self, candidates: list[CandidateResult], text: str, score_gate: bool = False) -> None:
        """See the module docstring: the score path with the gate, the
        demote/promote path without it. On an Ollama failure it stops
        reviewing for the rest of this process and leaves decisions as they are."""
        self.overrides = []
        if self.failed:
            return
        if score_gate:
            self._review_scores(candidates, text)
        else:
            self._review_decisions(candidates, text)

    def _review_scores(self, candidates: list[CandidateResult], text: str) -> None:
        groups: dict[str, list[CandidateResult]] = {}
        for c in candidates:
            if c.state.person_score is not None and c.state.person_score >= REVIEW_AT - self.score_weight:
                groups.setdefault(c.candidate.normalized_text, []).append(c)
        tag = f"{self.model} {self.digest[:12]}, prompt {PROMPT_VERSION}"
        for name, group in groups.items():
            group.sort(key=lambda c: c.candidate.start)
            picks = sorted({0, len(group) // 2, len(group) - 1})[:MAX_EXCERPTS]
            sentences = [sentence_context(text, group[k].candidate.start, group[k].candidate.end) for k in picks]
            try:
                probs = self.classify(name, sentences)
            except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError) as exc:
                self.failed = str(exc)
                logger.warning("LLM reviewer stopped for this process (Ollama error: %s) - "
                               "remaining names keep their scores", exc)
                return
            if not probs:
                continue
            p = probs["PERSON"]
            top = max(probs, key=probs.get)
            old = group[0].state.person_score
            delta = (self.score_weight if p >= 0.5 else self.score_weight_down) * (2 * p - 1)
            if delta < 0 and (name.lower() in self.confirmed_names or _cited(group)):
                delta = 0.0  # an owner decision ("Siri"), or a cited author - see _cited
            new = round(min(1.0, max(0.0, old + delta)), 2)
            old_decision, new_decision = decision_for(old), decision_for(new)
            label = f"LLM reviewer: {top} {probs[top]:.2f}, P(person) {p:.2f}; score {old:.2f} -> {new:.2f} ({tag})"
            for c in group:
                c.state.person_score = new
                c.state.decision = new_decision
                c.state.add_evidence("llm_reviewer", label, 0.0)
                if new_decision == Decision.REJECTED:
                    c.state.rejection_reason = f"{label} - below {REVIEW_AT}"
                elif new_decision != old_decision:
                    c.state.rejection_reason = None
            if new_decision != old_decision:
                override = {"name": name, "from": old_decision.value, "to": new_decision.value, "class": top,
                            "class_p": round(probs[top], 2), "p_person": round(p, 2), "score_before": old,
                            "score_after": new, "sentence": sentences[0]}
                self.overrides.append(override)
                logger.info("LLM override: %s %s -> %s (%s %.2f, P(person) %.2f, score %.2f -> %.2f)",
                            name, old_decision.value, new_decision.value, top, probs[top], p, old, new)

    def _review_decisions(self, candidates: list[CandidateResult], text: str) -> None:
        """Judges each distinct ACCEPTED name (and REVIEW name, if promotion is
        on) once per file, from up to three of its mentions, and applies the
        thresholds to all its mentions."""
        groups: dict[tuple[Decision, str], list[CandidateResult]] = {}
        for c in candidates:
            if c.state.decision == Decision.ACCEPTED or (
                    c.state.decision == Decision.REVIEW and self.promote_at_or_above is not None
                    and not (c.state.rejection_reason or "").startswith(_LONE_WORD_REASON_PREFIX)):
                groups.setdefault((c.state.decision, c.candidate.normalized_text), []).append(c)
        tag = f"{self.model} {self.digest[:12]}, prompt {PROMPT_VERSION}"
        for (decision, name), group in groups.items():
            group.sort(key=lambda c: c.candidate.start)
            picks = sorted({0, len(group) // 2, len(group) - 1})[:MAX_EXCERPTS]
            try:
                p = self.p_person(name, [excerpt(text, group[k].candidate.start, group[k].candidate.end) for k in picks])
            except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError) as exc:
                self.failed = str(exc)
                logger.warning("LLM reviewer stopped for this process (Ollama error: %s) - "
                               "remaining files keep their rule-based decisions", exc)
                return
            if decision == Decision.ACCEPTED and self.demote_below is not None and p < self.demote_below \
                    and name.lower() not in self.confirmed_names and not _cited(group):
                for c in group:
                    c.state.decision = Decision.REVIEW
                    c.state.rejection_reason = (f"LLM reviewer: P(person) {p:.2f} is below {self.demote_below} "
                                                f"({tag}) - held for review")
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f} - held for review", 0.0)
            elif decision == Decision.REVIEW and p >= self.promote_at_or_above:
                for c in group:
                    c.state.decision = Decision.ACCEPTED
                    c.state.rejection_reason = None
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f} - promoted", 0.0)
            else:
                for c in group:
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f}", 0.0)

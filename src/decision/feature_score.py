"""
src.decision.feature_score
=============================
A person score in [0, 1] from the features the textbook error brief lists
(P8, 2026-10-01), and a short plain-words reason for each name.

The final score is computed per surface form across all its mentions in the
document, for every name the validators let through, and goes into the
output's `score` column. Round 3 (2026-10-01): it decides - ACCEPT_AT or more
is accepted, REVIEW_AT or more goes to review, lower is dropped
(config["decision"]["feature_score_gate"], on by default).

The decision engine's verdict (after the structured-data caps) is one input:
ENGINE_BASE gives an accepted name 0.65 and a review name 0.30; the features
below then add POSITIVE_SCALE x their weights and NEGATIVE_SCALE x the
negative ones. So a name the engine accepted keeps its place with any
supporting feature, a review name needs real evidence to stay, and one with
none is dropped. Weights picked on the English textbook alone did as well
on the Social one. The round-2 feature score alone, without the engine's
verdict, could not tell a bare first name from noise (accepted F1 82.5 ->
66.6).

Positive features: a title before the name; a kinship word, sign-off,
salutation or speaker label (person cues); a list of names; an alias or
the first name of an accepted full name (document_name); a speech verb
after it ("said"); two or more name-like words; never written in
lowercase in the document; a first-name list hit; a spaCy PERSON tag;
three or more mentions. Negative: an ordinary word by the document's own
use (the lowercase test of entity_type_validator.py); a single English
word the first-name list doesn't know; quoted as a title. (Place, building and award cues,
headers and footers already remove a name before it gets here -
entity_type_validator.py, page_furniture.py.)
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.candidate.name_propagation import DOCUMENT_NAME_PATTERN
from src.core.models import CandidateResult, Decision, DetectorName
from src.detection.regex_detector import PERSON_CUE_PATTERN
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.document_case import lowercase_uses, mid_sentence_capitalised_uses
from src.validation.validators.entity_type_validator import MIN_LOWERCASE_USES, _in_quoted_title

ACCEPT_AT = 0.7
REVIEW_AT = 0.45
ENGINE_BASE = {Decision.ACCEPTED: 0.65, Decision.REVIEW: 0.30}
POSITIVE_SCALE = 0.6
NEGATIVE_SCALE = 2.0

_WEIGHTS = {
    "title": 0.35, "initials": 0.35, "person_cue": 0.30, "list": 0.25, "vouched": 0.20, "speech_verb_after": 0.15,
    "two_name_words": 0.20, "never_lowercase": 0.10, "first_name_list": 0.15, "spacy": 0.10, "repeated": 0.05,
    "lowercase_use": -0.30, "ordinary_word": -0.15, "quoted_title": -0.25,
}
_SPEECH_AFTER = re.compile(
    r"[ \t]*,?[ \t]*(?:said|says|asked|asks|replied|answered|told|shouted|cried|whispered|exclaimed|wrote)\b")
_INITIALS = re.compile(r"^(?:[A-Z]\.){1,4}$")
_REASON_TEXT = {
    "title": "title", "kinship": "kinship word", "speaker": "speaker label", "sign_off": "letter sign-off",
    "salutation": "salutation", "list": "list of names", "two_name_words": "first+last name",
    "first_name_list": "first-name list", "spacy": "spaCy", "speech_verb_after": "speech verb after",
    "never_lowercase": "never lowercase", "repeated": "repeated",
}


def person_features(group: list[CandidateResult], document_text: str, knowledge_base: KnowledgeBase) -> dict[str, bool]:
    """The features of one surface form, over all its mentions."""
    detections = [d for c in group for d in c.candidate.source_detections]
    patterns = {d.metadata.get("pattern") for d in detections}
    cues = {d.metadata.get("cue") for d in detections if d.metadata.get("pattern") == PERSON_CUE_PATTERN}
    tokens = group[0].candidate.normalized_text.split()
    body = [t for t in tokens if not (knowledge_base.is_title(t.rstrip(".")) or knowledge_base.is_honorific(t.rstrip(".")))]
    words = [t.rstrip(".") for t in body if len(t.rstrip(".")) > 1]
    lowercase = sum(lowercase_uses(w, document_text) for w in words) if words else 0
    single = len(body) == 1
    # An ordinary word by the validator's own test: used in lowercase at
    # least MIN_LOWERCASE_USES times, and at least as often as capitalised
    # mid-sentence - one stray lowercase "savitri" is no evidence.
    ordinary_by_case = single and bool(words) and lowercase >= MIN_LOWERCASE_USES and (
        lowercase >= mid_sentence_capitalised_uses(words[0], document_text))
    return {
        "title": "titled" in patterns or len(body) < len(tokens),
        # Initials before a name ("A.B. Vajpayee", "K.V. Ranga Reddy") - round 3.
        "initials": len(body) >= 2 and any(_INITIALS.match(t) for t in body[:-1]) and not _INITIALS.match(body[-1]),
        "person_cue": bool(cues - {"list"}),
        "list": "list" in cues,
        "vouched": DOCUMENT_NAME_PATTERN in patterns,
        "speech_verb_after": any(_SPEECH_AFTER.match(document_text, c.candidate.end) for c in group),
        "two_name_words": len(body) >= 2 and any(
            knowledge_base.is_known_first_name(w) or knowledge_base.is_known_last_name(w) for w in words),
        "never_lowercase": bool(words) and lowercase == 0,
        "first_name_list": bool(words) and knowledge_base.is_known_first_name(words[0]),
        "spacy": any(d.detector == DetectorName.SPACY for d in detections),
        "repeated": len(group) >= 3,
        "lowercase_use": ordinary_by_case,
        # common_english_words.txt lists many first names ("henry", "ray") -
        # an ordinary word here is one the first-name list doesn't know.
        "ordinary_word": single and bool(words) and knowledge_base.is_common_word(words[0])
                         and not knowledge_base.is_known_first_name(words[0]),
        # As the entity-type validator counts it: quoted as a title in at
        # least a third of its mentions - one quoted contact field in a chat
        # export ('"Priya"') says nothing (benchmark_real_660, 2026-10-01).
        "quoted_title": 3 * sum(_in_quoted_title(document_text, c.candidate.start, c.candidate.end)
                                for c in group) >= len(group),
        "_cues": sorted(c for c in cues if c),
    }


def score_from_features(features: dict, engine_decision: Decision | None = None) -> float:
    """The final score: the engine's verdict plus the scaled features."""
    total = ENGINE_BASE.get(engine_decision, 0.0) + sum(
        w * (POSITIVE_SCALE if w > 0 else NEGATIVE_SCALE) for name, w in _WEIGHTS.items() if features.get(name))
    return round(min(1.0, max(0.0, total)), 2)


def _engine_decision(group: list[CandidateResult]) -> Decision:
    decisions = {c.state.decision for c in group}
    return Decision.ACCEPTED if Decision.ACCEPTED in decisions else (
        Decision.REVIEW if Decision.REVIEW in decisions else Decision.REJECTED)


def score_persons(candidates: list[CandidateResult], document_text: str, knowledge_base: KnowledgeBase) -> None:
    """Sets state.person_score on every candidate the validators let through
    (accepted, review, or rejected by the engine for a low score) - the same
    score for every mention of one surface form."""
    groups: dict[str, list[CandidateResult]] = defaultdict(list)
    for c in candidates:
        if c.state.decision in (Decision.ACCEPTED, Decision.REVIEW) or (
                c.state.decision == Decision.REJECTED and not c.state.is_hard_rejected):
            groups[c.candidate.normalized_text].append(c)
    for group in groups.values():
        score = score_from_features(person_features(group, document_text, knowledge_base), _engine_decision(group))
        for c in group:
            c.state.person_score = score


def decision_for(score: float) -> Decision:
    return Decision.ACCEPTED if score >= ACCEPT_AT else Decision.REVIEW if score >= REVIEW_AT else Decision.REJECTED


def apply_score_gate(candidates: list[CandidateResult]) -> None:
    """Accept / review / drop by the final score alone."""
    for c in candidates:
        score = c.state.person_score
        if score is None:
            continue
        decision = decision_for(score)
        if decision != c.state.decision:
            c.state.add_evidence("feature_score", f"Final score {score:.2f} -> {decision.value}", 0.0)
        c.state.decision = decision
        if decision == Decision.REJECTED:
            c.state.rejection_reason = f"Final score {score:.2f} below {REVIEW_AT}"
        elif c.state.rejection_reason and c.state.rejection_reason.startswith(("Final confidence", "Final score")):
            c.state.rejection_reason = None


def entity_reason(mentions: list[CandidateResult]) -> str:
    """Why this is a person, in a few words: "title; first+last name; spaCy"."""
    detections = [d for c in mentions for d in c.candidate.source_detections]
    reasons: list[str] = []
    patterns = {d.metadata.get("pattern") for d in detections}
    if "titled" in patterns:
        reasons.append("title")
    for d in detections:
        cue = d.metadata.get("cue") if d.metadata.get("pattern") == PERSON_CUE_PATTERN else None
        if cue and _REASON_TEXT.get(cue, cue) not in reasons:
            reasons.append(_REASON_TEXT.get(cue, cue))
    vouched = sorted({d.metadata.get("vouched_by") for d in detections
                      if d.metadata.get("pattern") == DOCUMENT_NAME_PATTERN and d.metadata.get("vouched_by")})
    if vouched:
        reasons.append(f"also named '{vouched[0]}'")
    if "dictionary" in patterns:
        reasons.append("first+last name")
    elif "dictionary_single_token" in patterns:
        reasons.append("name list")
    if any(d.detector == DetectorName.SPACY for d in detections):
        reasons.append("spaCy")
    if len(mentions) >= 3:
        reasons.append(f"{len(mentions)} mentions")
    return "; ".join(reasons) or "capitalised name shape"

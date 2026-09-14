"""Turning a topic into a shootable script.

Three things make this more than a prompt wrapper:

1. **The numbers are handed in, not generated.** `market.py` computes them; the
   model receives them as fixed text it may quote but not recalculate.
2. **The output is audited against those numbers.** Every figure in the finished
   script is matched back to the fact sheet, and anything unmatched is reported.
   A model that invents a return on a finance channel is a credibility problem,
   so it gets caught here rather than in the edit.
3. **The script feeds the editor.** Its spoken text becomes the transcription
   prior, so captions on footage shot from a script are markedly more accurate.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .arabic import normalize_for_match
from .knowledge import Framework, Studio
from .llm import LLMResponse, generate

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUMBER_RE = re.compile(r"\d[\d,٬]*(?:[.٫]\d+)?")
# Counts, ordinals and month numbers are not claims about data.
TRIVIAL_MAX = 12

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "caption_hook": {"type": "string"},
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "text": {"type": "string"},
                    "seconds": {"type": "number"},
                    "onscreen": {"type": "string"},
                    "broll": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "text", "seconds", "onscreen", "broll"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "caption_hook", "beats"],
    "additionalProperties": False,
}


@dataclass
class Beat:
    role: str
    text: str
    seconds: float = 0.0
    onscreen: str = ""
    broll: list[str] = field(default_factory=list)


@dataclass
class Script:
    topic: str
    framework: str
    language: str
    title: str = ""
    caption_hook: str = ""
    beats: list[Beat] = field(default_factory=list)
    facts_used: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    created: str = ""

    @property
    def spoken_text(self) -> str:
        """Everything you will say, in order. This is the transcription prior."""
        return " ".join(beat.text.strip() for beat in self.beats if beat.text.strip())

    @property
    def seconds(self) -> float:
        return sum(beat.seconds for beat in self.beats)

    @property
    def word_count(self) -> int:
        return len(self.spoken_text.split())

    def broll_keywords(self) -> list[str]:
        seen: list[str] = []
        for beat in self.beats:
            for keyword in beat.broll:
                if keyword and keyword not in seen:
                    seen.append(keyword)
        return seen

    def to_dict(self) -> dict:
        data = asdict(self)
        data["spoken_text"] = self.spoken_text
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        data = dict(data)
        data.pop("spoken_text", None)
        beats = [Beat(**b) for b in data.pop("beats", [])]
        return cls(beats=beats, **data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Script":
        return cls.from_dict(json.loads(Path(path).read_text("utf-8")))

    def to_markdown(self) -> str:
        lines = [f"# {self.title or self.topic}", ""]
        if self.caption_hook:
            lines += [f"**Caption / first line:** {self.caption_hook}", ""]
        lines += [f"*{self.framework} · ~{self.seconds:.0f}s · {self.word_count} words*", ""]
        for index, beat in enumerate(self.beats, start=1):
            lines.append(f"### {index}. {beat.role}  ({beat.seconds:.0f}s)")
            lines.append("")
            lines.append(beat.text)
            lines.append("")
            extras = []
            if beat.onscreen:
                extras.append(f"on screen: **{beat.onscreen}**")
            if beat.broll:
                extras.append(f"b-roll: {', '.join(beat.broll)}")
            if extras:
                lines += ["> " + " · ".join(extras), ""]
        if self.unverified_numbers:
            lines += ["---", "", "**Check these numbers — they are not in your data:** "
                      + ", ".join(self.unverified_numbers), ""]
        return "\n".join(lines)

    def teleprompter(self) -> str:
        return "\n\n".join(beat.text.strip() for beat in self.beats if beat.text.strip())


# ------------------------------------------------------------- number audit

def _numbers_in(text: str) -> list[str]:
    return NUMBER_RE.findall((text or "").translate(ARABIC_DIGITS))


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("٬", "").replace("٫", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def allowed_numbers(facts: dict | None) -> set[float]:
    """Every value a script is permitted to state, drawn from the fact sheet."""
    allowed: set[float] = set()
    if not facts:
        return allowed

    def add(value) -> None:
        if isinstance(value, (int, float)) and value is not None:
            allowed.add(round(float(value), 4))
            allowed.add(round(float(value) * 100, 4))     # fractions quoted as percents

    for key in ("years", "start_price", "end_price", "total_return", "cagr", "volatility"):
        add(facts.get(key))
    for block in ("lump_sum", "monthly_plan"):
        for value in (facts.get(block) or {}).values():
            add(value)
    drawdown = facts.get("max_drawdown") or {}
    for key in ("drawdown", "peak", "trough"):
        add(drawdown.get(key))
    for year, value in (facts.get("calendar_years") or {}).items():
        add(_to_float(str(year)))
        add(value)
    for key in ("best_year", "worst_year"):
        pair = facts.get(key)
        if pair:
            add(_to_float(str(pair[0])))
            add(pair[1])

    # Anything already written into a statement is quotable as-is.
    for language_lines in (facts.get("statements") or {}).values():
        for line in language_lines:
            for token in _numbers_in(line):
                value = _to_float(token)
                if value is not None:
                    allowed.add(round(value, 4))
    return allowed


def _decimals(token: str) -> int:
    for separator in (".", "\u066b"):
        if separator in token:
            return len(token.rsplit(separator, 1)[1])
    return 0


def audit_numbers(text: str, facts: dict | None) -> list[str]:
    """Numbers in the script that cannot be traced back to the data.

    A quoted figure counts as verified when some real value rounds to it at the
    precision it was quoted at - so 5.66% may be said as "5.7%", but 99.9 is not
    excused by a real 99.0. A proportional tolerance was tried first and was too
    generous: one percent of 99 is nearly a whole unit, which let a fabricated
    99.9 pass. Strictness is the right default here; a false flag costs a glance,
    a missed one costs credibility.
    """
    allowed = allowed_numbers(facts)
    unverified: list[str] = []
    for token in _numbers_in(text):
        value = _to_float(token)
        if value is None:
            continue
        if float(value).is_integer() and abs(value) <= TRIVIAL_MAX:
            continue                                   # "3 steps", "5 minutes"
        places = _decimals(token)
        if any(abs(round(candidate, places) - value) < 1e-9 for candidate in allowed):
            continue
        if token not in unverified:
            unverified.append(token)
    return unverified


# ---------------------------------------------------------------- prompting

def build_system(studio: Studio, framework: Framework, language: str) -> str:
    context = studio.context().strip()
    language_name = {"ar": "Arabic", "en": "English"}.get(language, language)
    parts = [
        "You write scripts for short vertical videos (Reels, TikTok, Shorts).",
        f"Write the spoken lines in {language_name}, in the creator's own voice.",
        "",
        "Hard rules:",
        "- Use ONLY the figures given to you in the FACTS section. Never compute, "
        "estimate, adjust or invent a number. If a number you want is not in FACTS, "
        "write the sentence without it.",
        "- Do not promise or predict returns. Past performance is not a forecast.",
        "- Write lines to be spoken aloud, not read. Short sentences, one idea each.",
        "- No greetings, no 'welcome back', no throat-clearing. The first line is the hook.",
        "- `onscreen` is a two- or three-word caption overlay, not a full sentence.",
        "- `broll` is one or two keywords naming footage to cut to, in the same "
        "language as the script.",
        "",
        framework.as_prompt(),
    ]
    if context:
        parts += ["", "About the creator - match this voice and stay inside these limits:",
                  "", context]
    return "\n".join(parts)


def build_prompt(topic: str, *, facts: dict | None, seconds: int,
                 extra: str = "") -> str:
    parts = [f"Topic: {topic}", f"Target length: about {seconds} seconds when spoken."]
    if facts:
        parts += ["", "FACTS - the only numbers you may use:"]
        for line in (facts.get("statements") or {}).get("en", []):
            parts.append(f"- {line}")
        parts.append(f"(Source: {facts.get('name')} [{facts.get('symbol')}], "
                     f"{facts.get('start_day')} to {facts.get('end_day')}.)")
    else:
        parts += ["", "No market data was supplied. Do not state any statistic, "
                  "percentage or price."]
    if extra:
        parts += ["", f"Additional direction: {extra}"]
    parts += ["", "Return the script as JSON matching the required schema."]
    return "\n".join(parts)


# ---------------------------------------------------------------- generation

def skeleton(topic: str, framework: Framework, language: str,
             facts: dict | None) -> Script:
    """What you get with no model: the structure, blank, with the facts placed.

    Deliberately not filled with plausible prose - an obviously empty script is
    safer than a fake one that reads well enough to publish by accident.
    """
    statements = list((facts or {}).get("statements", {}).get(language, [])) or \
        list((facts or {}).get("statements", {}).get("en", []))
    beats: list[Beat] = []
    for index, spec in enumerate(framework.beats):
        text = ""
        if statements and spec.get("role") in ("context", "evidence", "cost", "proof",
                                               "side_a", "side_b", "hook"):
            text = statements.pop(0)
        beats.append(Beat(role=spec.get("role", f"beat{index + 1}"), text=text,
                          seconds=float(spec.get("seconds", 0)),
                          onscreen="", broll=[]))
    return Script(topic=topic, framework=framework.name, language=language,
                  title=topic, beats=beats, provider="stub", model="stub",
                  created=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def write_script(topic: str, studio: Studio, *, framework: str | None = None,
                 facts: dict | None = None, language: str = "ar",
                 seconds: int = 45, provider: str = "auto",
                 model: str | None = None, extra: str = "") -> Script:
    """Generate a script, then check every number in it against the data."""
    chosen = studio.framework(framework)
    system = build_system(studio, chosen, language)
    prompt = build_prompt(topic, facts=facts, seconds=seconds, extra=extra)

    response: LLMResponse = generate(prompt, system=system, schema=SCRIPT_SCHEMA,
                                     provider=provider, model=model)

    if not response.data:
        script = skeleton(topic, chosen, language, facts)
        script.provider = response.provider
        script.model = response.model
    else:
        payload = response.data
        beats = []
        for raw in payload.get("beats", []):
            broll = raw.get("broll") or []
            if isinstance(broll, str):
                broll = [broll]
            beats.append(Beat(role=str(raw.get("role", "")), text=str(raw.get("text", "")),
                              seconds=float(raw.get("seconds") or 0),
                              onscreen=str(raw.get("onscreen", "")),
                              broll=[str(k) for k in broll]))
        script = Script(
            topic=topic, framework=chosen.name, language=language,
            title=str(payload.get("title") or topic),
            caption_hook=str(payload.get("caption_hook") or ""),
            beats=beats, provider=response.provider, model=response.model,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    if facts:
        script.facts_used = list((facts.get("statements") or {}).get("en", []))
    audit_text = " ".join([script.spoken_text, script.caption_hook, script.title]
                          + [b.onscreen for b in script.beats])
    script.unverified_numbers = audit_numbers(audit_text, facts)
    return script


def caption_prior(script: Script, limit: int = 900) -> str:
    """The script as an ASR bias prompt for the editor.

    Knowing roughly what was said is the single cheapest accuracy win available
    to Arabic transcription - names, tickers and figures stop being guesses.
    """
    text = script.spoken_text.strip()
    return text[:limit]


def script_vocabulary(script: Script) -> dict[str, str]:
    """Distinctive words from the script, keyed for the correction map."""
    vocabulary: dict[str, str] = {}
    for word in re.split(r"\s+", script.spoken_text):
        cleaned = word.strip(".,،؛؟!:\"'()[]")
        if len(cleaned) < 4:
            continue
        key = normalize_for_match(cleaned)
        if key and key not in vocabulary:
            vocabulary[key] = cleaned
    return vocabulary

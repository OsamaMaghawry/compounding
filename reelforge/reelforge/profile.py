"""Style profiles: every editing decision is driven by these numbers.

The defaults below are the single source of truth. A profile YAML file only needs
to contain the keys it wants to override, and CLI flags override those in turn.
This matters for the learning loop: `reelforge` tunes a handful of these scalars
from your feedback, so "your style" is data, not code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "description": "Vertical talking-head Reel, fast cuts, Arabic karaoke captions.",

    "output": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "preset": "veryfast",      # x264 preset; 'ultrafast' for preview
        "crf": 20,
        "audio_bitrate": "192k",
        "zoom_headroom": 1.35,     # render canvas oversize so punch-ins stay sharp
    },

    "reframe": {
        "mode": "center",          # center | left | right | pad
        "pad_color": "black",
        "blur_background": False,  # blurred fill instead of crop for wide sources
    },

    # Dead-air removal. The single biggest time saver on talking-head Reels.
    "cuts": {
        "enabled": True,
        "noise_db": -32.0,         # anything quieter counts as silence
        "min_silence": 0.35,       # only cut silences longer than this (seconds)
        "pad_before": 0.10,        # breathing room kept before speech resumes
        "pad_after": 0.12,
        "max_gap_keep": 0.22,      # longest pause allowed to survive a cut
        "min_segment": 0.25,       # drop slivers shorter than this
        "keep_head": 0.0,          # always keep the first N seconds verbatim
    },

    # Punch-ins / pull-outs timed to speech emphasis.
    "zoom": {
        "enabled": True,
        "rate_per_min": 14.0,      # target number of moves per minute
        "max_factor": 1.22,        # strongest punch-in
        "min_factor": 1.04,        # gentlest
        "min_gap": 1.1,            # seconds between moves
        "min_duration": 0.55,
        "max_duration": 2.4,
        "hook_punch": True,        # always punch in on the opening hook
        "hook_window": 1.6,
        "ease": "smooth",          # smooth | linear
        "alternate": True,         # alternate in/out so it doesn't creep
        "score_threshold": 0.45,   # emphasis score required to trigger a move
        "cut_bias": 0.15,          # extra score for moments right after a cut
    },

    # Extra video layers pulled from your own library.
    "broll": {
        "enabled": True,
        "library": "assets/broll",
        "max_per_min": 6.0,
        "min_duration": 1.0,
        "max_duration": 2.6,
        "cooldown": 2.5,
        "mode": "cover",           # cover | pip | band
        "opacity": 1.0,
        "pip_scale": 0.42,
        "pip_pos": "top",          # top | bottom
        "min_score": 0.55,         # keyword match confidence required
        "head_guard": 1.0,         # never cover the first N seconds (the hook)
        "audio": "mute",           # mute | duck | keep
    },

    "captions": {
        "enabled": True,
        "language": "ar",
        "font": "Cairo",
        "font_size": 92,
        "bold": True,
        # How the spoken word is marked while you talk:
        #   karaoke - active word changes colour (default)
        #   box     - active word sits in a filled box (the CapCut look)
        #   pop     - active word scales up as it is spoken
        #   word    - one large word on screen at a time
        #   plain   - no per-word marking at all
        "style": "karaoke",
        "primary": "#FFFFFF",
        "highlight": "#FFD700",    # active-word colour
        "box_color": "#FFD700",    # active-word box fill, for style: box
        "box_text": "#101010",     # text colour inside that box
        "box_padding": 7,
        "pop_scale": 1.12,         # active-word scale, for style: pop
        "word_size_boost": 1.55,   # font multiplier, for style: word
        # Important words stay marked even when they are not being spoken.
        "emphasis": True,
        "emphasis_color": "#3DDC97",
        "emphasis_scale": 1.0,      # >1 makes the word bigger, but it will not reflow
        "emphasis_words": [],      # your own terms, added to the built-in list
        "outline_color": "#101010",
        "outline": 7,
        "shadow": 3,
        "y_pct": 0.72,             # vertical position, 0 = top, 1 = bottom
        "margin_x": 90,
        "max_words": 4,            # words per caption line - Reels want few
        "max_chars": 34,
        "min_duration": 0.55,
        "max_duration": 3.0,
        "gap_split": 0.45,         # a pause this long starts a new line
        "line_gap": 0.08,          # blank time between lines, so they do not run together
        "karaoke": True,           # highlight the word being spoken
        "highlight_scale": 1.0,    # >1 pops the active word (CapCut-style)
        "strip_diacritics": False,
        "normalize_punctuation": True,
        "arabic_percent": False,   # render % as the Arabic ٪ sign
        "safe_area": True,         # keep out of the UI overlay zones
    },

    # Cut-point effects. Short by design - a transition you notice is too long.
    "transitions": {
        "enabled": True,
        "kind": "auto",            # auto | punch | flash | blur | none
        "duration": 0.18,
        "strength": 0.7,           # 0..1, scales the effect
        "min_gap": 0.9,            # never stack two transitions closer than this
        "max_per_min": 24.0,
        "scene_change_only": False,  # only transition where the shot actually changes
    },

    "audio": {
        "loudnorm": True,
        "target_lufs": -14.0,      # what Instagram/TikTok normalise to anyway
        "music_path": None,
        "music_db": -22.0,
        "duck_db": -12.0,
    },

    "asr": {
        "backend": "auto",         # auto | faster-whisper | whispercpp | stub
        "model": "large-v3",       # tiny|base|small|medium|large-v3 (or a local path)
        "device": "auto",          # auto | cpu | cuda
        "compute_type": "auto",    # auto | int8 | int8_float16 | float16 | float32
        "language": "ar",
        "beam_size": 5,
        "vad": True,
        "vad_min_silence_ms": 300,
        "condition_on_previous_text": False,  # curbs Whisper's repetition loops
        "temperature_fallback": True,
        "initial_prompt": None,    # extended automatically with your learned vocab
        "word_timestamps": True,
    },

    "learning": {
        "enabled": True,
        "min_samples": 40,         # labelled examples before the model overrides rules
        "learning_rate": 0.08,
        "ema": 0.25,               # how fast tuned scalars move toward your edits
        "max_vocab_prompt": 220,   # characters of learned vocab fed to the ASR
    },
}

_MISSING = object()


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".json",):
        return json.loads(text)
    try:
        import yaml  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Reading YAML profiles needs PyYAML (`pip install pyyaml`), "
            "or use a .json profile instead."
        ) from exc
    return yaml.safe_load(text) or {}


class StyleProfile:
    """Dotted-path access over a merged settings tree."""

    def __init__(self, data: dict | None = None):
        self.data = _deep_merge(DEFAULTS, data or {})

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None) -> "StyleProfile":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"profile not found: {path}")
        return cls(_load_mapping(path))

    @classmethod
    def resolve(cls, name_or_path: str | None, search_dirs: list[Path] | None = None) -> "StyleProfile":
        """Accept a profile name ('punchy') or an explicit path."""
        if not name_or_path:
            return cls()
        candidate = Path(name_or_path)
        if candidate.exists():
            return cls.load(candidate)
        for directory in search_dirs or []:
            for suffix in (".yml", ".yaml", ".json"):
                probe = Path(directory) / f"{name_or_path}{suffix}"
                if probe.exists():
                    return cls.load(probe)
        raise FileNotFoundError(f"no profile named '{name_or_path}'")

    # -- access ----------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"unknown profile key: {dotted}")
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def merged(self, override: dict) -> "StyleProfile":
        return StyleProfile(_deep_merge(self.data, override))

    def apply_overrides(self, pairs: list[str]) -> "StyleProfile":
        """Apply `--set zoom.max_factor=1.3` style overrides with type coercion."""
        override: dict = {}
        for pair in pairs or []:
            if "=" not in pair:
                raise ValueError(f"override must look like key.path=value, got '{pair}'")
            key, raw = pair.split("=", 1)
            node = override
            parts = key.strip().split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = _coerce(raw.strip())
        return self.merged(override)

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            import yaml  # noqa: PLC0415
            path.write_text(
                yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        return path

    def fingerprint(self) -> str:
        """Stable hash of the settings that affect analysis - used for cache keys."""
        import hashlib
        relevant = {k: self.data[k] for k in ("cuts", "asr", "captions") if k in self.data}
        blob = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StyleProfile(name={self.data.get('name')!r})"


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

"""Your own b-roll library, matched to what you are saying.

No stock-footage API and no cloud search: the library is a folder of your clips.
Keywords come from filenames by default (Arabic filenames work), or from an
optional `library.yml` when you want several keywords per asset.

`library.yml`:
    clips/money.mp4:
      keywords: [فلوس, مال, ارباح, money, profit]
      start: 1.5        # skip the first 1.5s of the asset
    clips/laptop.mp4: [لابتوب, كمبيوتر, شغل]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .arabic import normalize_for_match, similarity, tokens

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STOPWORDS = {
    "في", "من", "على", "عن", "الى", "الي", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "ان", "انا", "انت", "هو", "هي", "ما", "لا", "و", "يا", "كل", "بعد", "قبل",
    "the", "a", "an", "of", "in", "on", "to", "and", "is", "it",
}


@dataclass
class BrollAsset:
    path: Path
    keywords: list[str] = field(default_factory=list)
    start: float = 0.0
    duration: float | None = None

    @property
    def is_image(self) -> bool:
        return self.path.suffix.lower() in IMAGE_EXT

    @property
    def name(self) -> str:
        return self.path.name


def _keywords_from_filename(path: Path) -> list[str]:
    stem = path.stem
    for separator in ("_", "-", ".", ","):
        stem = stem.replace(separator, " ")
    parts = [p for p in stem.split() if p and not p.isdigit()]
    return [normalize_for_match(p) for p in parts if normalize_for_match(p)]


class BrollLibrary:
    """A searchable set of local clips and stills."""

    def __init__(self, assets: list[BrollAsset] | None = None):
        self.assets = assets or []

    def __len__(self) -> int:
        return len(self.assets)

    @classmethod
    def load(cls, directory: str | Path | None) -> "BrollLibrary":
        if not directory:
            return cls()
        directory = Path(directory)
        if not directory.exists():
            return cls()

        overrides: dict[str, dict] = {}
        for candidate in ("library.yml", "library.yaml", "library.json"):
            manifest = directory / candidate
            if not manifest.exists():
                continue
            try:
                if manifest.suffix == ".json":
                    import json  # noqa: PLC0415
                    raw = json.loads(manifest.read_text("utf-8"))
                else:
                    import yaml  # noqa: PLC0415
                    raw = yaml.safe_load(manifest.read_text("utf-8")) or {}
            except Exception:
                raw = {}
            for key, value in (raw or {}).items():
                if isinstance(value, list):
                    overrides[key] = {"keywords": value}
                elif isinstance(value, dict):
                    overrides[key] = value
            break

        assets: list[BrollAsset] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in VIDEO_EXT and suffix not in IMAGE_EXT:
                continue
            relative = path.relative_to(directory).as_posix()
            config = overrides.get(relative) or overrides.get(path.name) or {}
            keywords = [normalize_for_match(k) for k in config.get("keywords", [])]
            keywords = [k for k in keywords if k] or _keywords_from_filename(path)
            assets.append(BrollAsset(path=path, keywords=keywords,
                                     start=float(config.get("start", 0.0)),
                                     duration=config.get("duration")))
        return cls(assets)

    def match(self, text: str, *, weights: dict[str, float] | None = None,
              min_score: float = 0.55) -> tuple[BrollAsset, str, float] | None:
        """Best (asset, keyword, score) for a caption line, or None."""
        line_tokens = [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]
        if not line_tokens or not self.assets:
            return None

        best: tuple[BrollAsset, str, float] | None = None
        for asset in self.assets:
            for keyword in asset.keywords:
                if not keyword or keyword in STOPWORDS:
                    continue
                score = max((similarity(token, keyword) for token in line_tokens), default=0.0)
                if score <= 0:
                    continue
                # Learned preference: assets you keep get promoted, ones you delete sink.
                score *= (weights or {}).get(f"{asset.name}|{keyword}", 1.0)
                if score >= min_score and (best is None or score > best[2]):
                    best = (asset, keyword, min(score, 1.0))
        return best

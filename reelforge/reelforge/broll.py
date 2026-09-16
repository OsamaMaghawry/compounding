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
            # Skip hidden files and anything inside a hidden folder: that is where
            # the thumbnails live, and a thumbnail is an image, so without this the
            # library would happily cut its own preview pictures into your video.
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
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


# ------------------------------------------------------------------ managing

MANIFEST = "library.json"


def read_manifest(directory: str | Path) -> dict:
    """The keyword file, as a plain mapping. Missing or broken reads as empty."""
    path = Path(directory) / MANIFEST
    if not path.exists():
        return {}
    try:
        import json  # noqa: PLC0415
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def write_manifest(directory: str | Path, data: dict) -> Path:
    import json  # noqa: PLC0415
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def is_supported(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in VIDEO_EXT or suffix in IMAGE_EXT


def thumbnail(asset: str | Path, dest: str | Path, *, width: int = 240) -> Path | None:
    """One frame from a clip, or a scaled copy of a still.

    A library listed by filename alone is nearly useless - you end up opening
    things to remember what they are. A picture makes it a library.
    """
    from .ffmpeg import run_ffmpeg  # noqa: PLC0415 - avoid a cycle at import time
    asset, dest = Path(asset), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = []
    if asset.suffix.lower() in VIDEO_EXT:
        # A frame from just inside the clip: the very first one is often black.
        args += ["-ss", "0.5"]
    args += ["-i", str(asset), "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-y", str(dest)]
    try:
        run_ffmpeg(args)
    except Exception:
        return None
    return dest if dest.exists() and dest.stat().st_size > 0 else None


def make_proxy(asset: str | Path, dest: str | Path, *, height: int = 480) -> Path | None:
    """A small copy of a library clip, for the preview to lay over you.

    The original is whatever came off a phone. Streaming that every time a
    two-second cutaway appears is a lot of bytes for a picture nobody keeps.
    """
    from .render import build_proxy  # noqa: PLC0415 - avoid a cycle at import time
    asset, dest = Path(asset), Path(dest)
    if asset.suffix.lower() not in VIDEO_EXT:
        return None
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        return build_proxy(asset, dest, height=height)
    except Exception:
        return None

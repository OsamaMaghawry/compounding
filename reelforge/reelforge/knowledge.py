"""Your studio: the files that make the writing sound like you.

Plain markdown and YAML in a folder you own and edit by hand. No database, no
hidden state - if the writing is wrong, you can see exactly which sentence in
which file caused it.

The defaults are deliberately empty prompts rather than invented biography. A
made-up credential that reads plausibly is worse than a blank, because it ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_FRAMEWORKS = PACKAGE_ROOT / "frameworks"

STUDIO_TEMPLATE = PACKAGE_ROOT / "studio_template"
STUDIO_FILES = ("background.md", "voice.md", "audience.md")


def template(name: str) -> str:
    """The starting text for one studio file.

    These live as real markdown in `studio_template/` rather than as strings in
    here, so they can be read and edited like any other file in the repo - which
    is where anyone looking for them will look first.
    """
    path = STUDIO_TEMPLATE / name
    return path.read_text("utf-8") if path.exists() else ""


# Kept as module constants for convenience; the files are the source of truth.
BACKGROUND = template("background.md")
VOICE = template("voice.md")
AUDIENCE = template("audience.md")


@dataclass
class Framework:
    name: str
    description: str
    beats: list[dict] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    best_for: str = ""
    path: Path | None = None

    @property
    def seconds(self) -> int:
        return int(sum(b.get("seconds", 0) for b in self.beats))

    def as_prompt(self) -> str:
        lines = [f"Framework: {self.name} - {self.description}", "", "Beats, in order:"]
        for index, beat in enumerate(self.beats, start=1):
            lines.append(f"{index}. {beat.get('role')} (~{beat.get('seconds')}s): "
                         f"{beat.get('purpose')}")
        if self.rules:
            lines += ["", "Rules for this framework:"]
            lines += [f"- {rule}" for rule in self.rules]
        return "\n".join(lines)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("frameworks need PyYAML (`pip install pyyaml`)") from exc
    return yaml.safe_load(path.read_text("utf-8")) or {}


def load_framework(path: Path) -> Framework:
    data = _load_yaml(path)
    return Framework(
        name=data.get("name", path.stem), description=data.get("description", ""),
        beats=data.get("beats", []), rules=data.get("rules", []),
        best_for=data.get("best_for", ""), path=path,
    )


class Studio:
    """The folder holding everything the writer knows about you."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def frameworks_dir(self) -> Path:
        return self.root / "frameworks"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def exists(self) -> bool:
        return (self.root / "background.md").exists()

    def init(self, *, force: bool = False) -> list[Path]:
        """Create the studio files, copying the built-in frameworks in to be edited."""
        written: list[Path] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.frameworks_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        for name in STUDIO_FILES:
            path = self.root / name
            if path.exists() and not force:
                continue
            body = template(name)
            if not body:
                continue
            path.write_text(body, encoding="utf-8")
            written.append(path)

        if BUILTIN_FRAMEWORKS.exists():
            for source in sorted(BUILTIN_FRAMEWORKS.glob("*.yml")):
                target = self.frameworks_dir / source.name
                if target.exists() and not force:
                    continue
                target.write_text(source.read_text("utf-8"), encoding="utf-8")
                written.append(target)
        return written

    # -- reading ---------------------------------------------------------
    def context(self) -> str:
        """Everything the writer should know about you, as one block of text."""
        parts: list[str] = []
        for name in ("background.md", "voice.md", "audience.md"):
            path = self.root / name
            if path.exists():
                body = path.read_text("utf-8").strip()
                if body:
                    parts.append(body)
        return "\n\n---\n\n".join(parts)

    def is_unedited(self) -> bool:
        """True while the studio still holds the placeholder text."""
        background = self.root / "background.md"
        if not background.exists():
            return True
        return "(Your name, and what you actually do.)" in background.read_text("utf-8")

    def frameworks(self) -> list[Framework]:
        seen: dict[str, Framework] = {}
        for directory in (BUILTIN_FRAMEWORKS, self.frameworks_dir):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.yml")):
                framework = load_framework(path)
                seen[framework.name] = framework      # the studio copy wins
        return sorted(seen.values(), key=lambda f: f.name)

    def framework(self, name: str | None) -> Framework:
        available = self.frameworks()
        if not available:
            raise FileNotFoundError("no frameworks found - run `reelforge studio init`")
        if not name:
            for framework in available:
                if framework.path and _load_yaml(framework.path).get("default"):
                    return framework
            return available[0]
        for framework in available:
            if framework.name.lower() == name.lower():
                return framework
        raise FileNotFoundError(
            f"no framework called '{name}'. Available: "
            f"{', '.join(f.name for f in available)}"
        )

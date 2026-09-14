"""Curated Arabic fonts for captions, with on-demand download.

All of these are SIL Open Font License, fetched from the official Google Fonts
repository. The family name is what goes in `captions.font`; the file is what
libass loads out of `assets/fonts/`.

Picked for burned-in captions specifically: heavy enough to read at speed over
moving footage, with real Arabic coverage rather than a Latin font that happens
to render Arabic badly.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

GOOGLE = "https://raw.githubusercontent.com/google/fonts/main/ofl"


@dataclass
class FontEntry:
    family: str          # the name you put in captions.font
    filename: str
    url: str
    note: str
    default: bool = False


CATALOG: list[FontEntry] = [
    FontEntry("Cairo", "Cairo.ttf", f"{GOOGLE}/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
              "The safe default. Clean, modern, reads at any size.", default=True),
    FontEntry("Tajawal", "Tajawal-Bold.ttf", f"{GOOGLE}/tajawal/Tajawal-Bold.ttf",
              "Geometric and friendly. Good for explainers.", default=True),
    FontEntry("Almarai", "Almarai-ExtraBold.ttf", f"{GOOGLE}/almarai/Almarai-ExtraBold.ttf",
              "Very heavy. Best where captions sit over busy footage.", default=True),
    FontEntry("Changa", "Changa.ttf", f"{GOOGLE}/changa/Changa%5Bwght%5D.ttf",
              "Condensed - fits more words per line. Avoid if you say percentages: its % glyph crowds the next word."),
    FontEntry("Alexandria", "Alexandria.ttf", f"{GOOGLE}/alexandria/Alexandria%5Bwght%5D.ttf",
              "Wide and confident. Strong for one-word-at-a-time captions."),
    FontEntry("El Messiri", "ElMessiri.ttf", f"{GOOGLE}/elmessiri/ElMessiri%5Bwght%5D.ttf",
              "Softer, more editorial. Suits calmer content."),
    FontEntry("Reem Kufi", "ReemKufi.ttf", f"{GOOGLE}/reemkufi/ReemKufi%5Bwght%5D.ttf",
              "Geometric Kufi. Distinctive for titles and hooks."),
    FontEntry("Noto Kufi Arabic", "NotoKufiArabic.ttf",
              f"{GOOGLE}/notokufiarabic/NotoKufiArabic%5Bwght%5D.ttf",
              "Neutral Kufi with the widest character coverage."),
    FontEntry("Marhey", "Marhey.ttf", f"{GOOGLE}/marhey/Marhey%5Bwght%5D.ttf",
              "Playful and rounded. Good for light, fun content."),
    FontEntry("Baloo Bhaijaan 2", "BalooBhaijaan2.ttf",
              f"{GOOGLE}/baloobhaijaan2/BalooBhaijaan2%5Bwght%5D.ttf",
              "Chunky and rounded. Very legible on small screens."),
    FontEntry("Lalezar", "Lalezar-Regular.ttf", f"{GOOGLE}/lalezar/Lalezar-Regular.ttf",
              "Display weight only. Loud - use for hooks, not paragraphs."),
    FontEntry("Rakkas", "Rakkas-Regular.ttf", f"{GOOGLE}/rakkas/Rakkas-Regular.ttf",
              "Decorative display. Distinctive but harder to read fast."),
    FontEntry("Amiri", "Amiri-Bold.ttf", f"{GOOGLE}/amiri/Amiri-Bold.ttf",
              "Classical Naskh. For traditional or literary content."),
    FontEntry("Aref Ruqaa", "ArefRuqaa-Bold.ttf", f"{GOOGLE}/arefruqaa/ArefRuqaa-Bold.ttf",
              "Ruqaa calligraphy. Beautiful, but slow to read."),
]

BY_FAMILY = {entry.family.lower(): entry for entry in CATALOG}


def installed(fonts_dir: Path) -> set[str]:
    """Families already present in the fonts folder."""
    if not Path(fonts_dir).exists():
        return set()
    names = {p.name for p in Path(fonts_dir).glob("*") if p.suffix.lower() in (".ttf", ".otf")}
    return {entry.family for entry in CATALOG if entry.filename in names}


def resolve(name: str) -> FontEntry | None:
    return BY_FAMILY.get((name or "").strip().lower())


def download(entry: FontEntry, fonts_dir: Path, *, force: bool = False,
             timeout: int = 60) -> tuple[bool, str]:
    """Fetch one font. Returns (changed, message)."""
    fonts_dir = Path(fonts_dir)
    fonts_dir.mkdir(parents=True, exist_ok=True)
    target = fonts_dir / entry.filename
    if target.exists() and not force:
        return False, f"have {entry.family}"
    try:
        with urllib.request.urlopen(entry.url, timeout=timeout) as response:
            data = response.read()
        if len(data) < 2048:
            return False, f"{entry.family}: download looked empty, skipped"
        target.write_bytes(data)
        return True, f"installed {entry.family}"
    except Exception as exc:
        return False, f"{entry.family}: {exc}"


def install(names: list[str] | None, fonts_dir: Path, *, force: bool = False):
    """Install named families, the defaults if none named, or all with ['all']."""
    if names and len(names) == 1 and names[0].lower() == "all":
        chosen = CATALOG
    elif names:
        chosen = []
        for name in names:
            entry = resolve(name)
            if entry is None:
                yield False, (f"unknown font '{name}'. "
                              f"Run `reelforge fonts` to see the list.")
            else:
                chosen.append(entry)
    else:
        chosen = [entry for entry in CATALOG if entry.default]

    for entry in chosen:
        yield download(entry, fonts_dir, force=force)

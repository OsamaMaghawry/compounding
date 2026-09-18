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


# ------------------------------------------------------------------- weights

WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular",
                500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold",
                900: "Black"}


def _stem(entry: FontEntry) -> str:
    """The file stem Google uses for this family: Cairo, ElMessiri, Amiri."""
    name = Path(entry.filename).stem
    for cut in ("-", "["):
        name = name.split(cut)[0]
    return name


def _google_dir(entry: FontEntry) -> str:
    return entry.url.split("/ofl/")[1].split("/")[0]


def weight_file(entry: FontEntry, weight: int, fonts_dir: Path) -> Path | None:
    """A face of this family at this weight, made or fetched, or None.

    libass does not read a variable font's weight axis: asked for 300 or 900 on
    a variable Cairo it draws the same synthetic bold either way, which was
    measured before believing it. So a variable font is cut to a real static
    face at the weight wanted, with its weight class set so libass picks it. A
    family that ships one file per weight gets that file fetched instead.
    """
    weight = int(round(weight / 100.0) * 100)
    if weight not in WEIGHT_NAMES:
        return None
    fonts_dir = Path(fonts_dir)
    base = fonts_dir / entry.filename
    target = fonts_dir / f"{_stem(entry)}-{WEIGHT_NAMES[weight]}.ttf"
    if target.exists():
        return target
    if base.exists() and _is_variable(base):
        return _instance(base, target, weight, family=entry.family)
    # A static family: Google keeps one file per weight where the weight exists.
    for candidate in (f"{GOOGLE}/{_google_dir(entry)}/{target.name}",):
        try:
            with urllib.request.urlopen(candidate, timeout=60) as response:
                data = response.read()
            if len(data) > 2048:
                target.write_bytes(data)
                return target
        except Exception:
            continue
    return None


def _is_variable(path: Path) -> bool:
    try:
        from fontTools.ttLib import TTFont  # noqa: PLC0415
        with TTFont(str(path), lazy=True) as font:
            return "fvar" in font
    except Exception:
        return False


def _instance(base: Path, target: Path, weight: int, *, family: str) -> Path | None:
    """Cut a variable font to one weight. Needs fontTools; None without it."""
    try:
        from fontTools.ttLib import TTFont  # noqa: PLC0415
        from fontTools.varLib import instancer  # noqa: PLC0415
    except ImportError:
        return None
    try:
        font = TTFont(str(base))
        axes = {a.axisTag: (a.minValue, a.maxValue) for a in font["fvar"].axes}
        if "wght" not in axes:
            return None
        low, high = axes["wght"]
        pinned = {"wght": max(low, min(high, float(weight)))}
        for tag in axes:
            if tag != "wght":
                pinned[tag] = 0.0 if tag == "slnt" else None    # leave others at default
        pinned = {k: v for k, v in pinned.items() if v is not None}
        cut = instancer.instantiateVariableFont(font, pinned, inplace=False,
                                                updateFontNames=True)
        # What libass keys on: the family name stays, the weight class says
        # which face this is.
        cut["OS/2"].usWeightClass = weight
        style = WEIGHT_NAMES[weight]
        for record in cut["name"].names:
            if record.nameID in (1, 16):
                record.string = family
            elif record.nameID in (2, 17):
                record.string = style
            elif record.nameID == 4:
                record.string = f"{family} {style}"
            elif record.nameID == 6:
                record.string = f"{family.replace(' ', '')}-{style}"
        cut.save(str(target))
        return target
    except Exception:
        return None

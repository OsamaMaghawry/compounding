"""Caption construction: group words into readable lines, then write ASS/SRT.

Word grouping is the part that decides whether captions feel professional. Reels
captions want very few words on screen at a time, broken at natural pauses, never
mid-phrase, and never flickering between lines.

Rendering relies on libass for Arabic shaping and bidi, so text is emitted in
logical (spoken) order and never pre-reversed - pre-reversing is the classic way
Arabic subtitles end up broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .arabic import (clean_for_display, emphasis_set, is_arabic, is_emphatic,
                     visual_order)
from .speech import Word

# Vertical space the Instagram/TikTok UI covers at the bottom of a 1920-tall frame.
SAFE_BOTTOM = 420


@dataclass
class CaptionLine:
    words: list[Word]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [{"text": w.text, "start": round(w.start, 3),
                       "end": round(w.end, 3), "prob": round(w.prob, 3)} for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CaptionLine":
        words = [Word(text=w["text"], start=w["start"], end=w["end"], prob=w.get("prob", 1.0))
                 for w in data.get("words", [])]
        if not words:
            words = [Word(text=data.get("text", ""), start=data["start"], end=data["end"])]
        return cls(words=words, start=data["start"], end=data["end"])


def group_words(words: list[Word], profile) -> list[CaptionLine]:
    """Break a word stream into caption lines at pauses and length limits."""
    max_words = int(profile.get("captions.max_words"))
    max_chars = int(profile.get("captions.max_chars"))
    gap_split = float(profile.get("captions.gap_split"))
    min_duration = float(profile.get("captions.min_duration"))
    max_duration = float(profile.get("captions.max_duration"))

    lines: list[CaptionLine] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        lines.append(CaptionLine(words=list(current), start=current[0].start, end=current[-1].end))
        current.clear()

    for word in words:
        if not word.text:
            continue
        if current:
            gap = word.start - current[-1].end
            span = word.end - current[0].start
            candidate_chars = len(" ".join(w.text for w in current)) + 1 + len(word.text)
            if (gap >= gap_split or len(current) >= max_words
                    or candidate_chars > max_chars or span > max_duration):
                flush()
        current.append(word)
    flush()

    line_gap = float(profile.get("captions.line_gap", 0.0))

    # A line that flashes by is unreadable; borrow time from the gap that follows,
    # but never all of it - the next line needs somewhere to begin.
    for index, line in enumerate(lines):
        if line.duration < min_duration:
            if index + 1 < len(lines):
                limit = lines[index + 1].start - line_gap
            else:
                limit = line.end + min_duration
            line.end = min(max(line.end, line.start + min_duration), limit)

    # Two lines on screen at once reads as a duplicated word, and lines that touch
    # never blink off, so one statement runs into the next.
    for index in range(len(lines) - 1):
        latest = lines[index + 1].start - line_gap
        if lines[index].end > latest:
            lines[index].end = max(lines[index].start + 0.12, latest)

    return [line for line in lines if line.text and line.end > line.start]


# ------------------------------------------------------------------ ASS output

def _ass_color(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR (ASS stores colours as alpha + reversed RGB)."""
    value = (hex_color or "#FFFFFF").lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        value = "FFFFFF"
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{blue}{green}{red}".upper()


def _inline_color(hex_color: str) -> str:
    value = (hex_color or "#FFFFFF").lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H{blue}{green}{red}&".upper()


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _wrap_rows(words: list[str], max_chars: int) -> list[list[int]]:
    """Split word indices into display rows, balancing row length."""
    if not words:
        return []
    total = len(" ".join(words))
    if total <= max_chars:
        return [list(range(len(words)))]
    rows: list[list[int]] = []
    current: list[int] = []
    length = 0
    for index, word in enumerate(words):
        add = len(word) + (1 if current else 0)
        if current and length + add > max_chars:
            rows.append(current)
            current, length = [index], len(word)
        else:
            current.append(index)
            length += add
    if current:
        rows.append(current)
    return rows


def _style_spec(profile, font_size: int) -> dict:
    """Per-style ASS parameters and the inline tags that mark the spoken word."""
    style = (profile.get("captions.style") or "karaoke").lower()
    primary = _inline_color(profile.get("captions.primary"))
    highlight = _inline_color(profile.get("captions.highlight"))
    pop = float(profile.get("captions.pop_scale"))

    spec = {
        "style": style,
        "font_size": font_size,
        "border_style": 1,
        "outline": float(profile.get("captions.outline")),
        "shadow": float(profile.get("captions.shadow")),
        "outline_color": _ass_color(profile.get("captions.outline_color")),
        "per_word": True,     # emit one event per word so the active one can change
        "one_word": False,    # show only the active word, nothing else
        "active_open": f"{{\\c{highlight}}}",
        "active_close": f"{{\\c{primary}}}",
        "line_prefix": "",
    }

    if style == "plain":
        spec["per_word"] = False
        spec["active_open"] = spec["active_close"] = ""
    elif style == "pop":
        # Scaling a single word inline does not reflow the rest of the line, so the
        # word grows straight into its neighbour. Pulse the whole line instead: the
        # colour change still tracks the spoken word, and nothing can collide.
        scale = int(round(pop * 100))
        spec["line_prefix"] = (f"{{\\fscx{scale}\\fscy{scale}"
                               f"\\t(0,150,\\fscx100\\fscy100)}}")
        spec["active_open"] = f"{{\\c{highlight}}}"
        spec["active_close"] = f"{{\\c{primary}}}"
    elif style == "box":
        # BorderStyle 3 turns the outline into a filled box. The style's box is
        # transparent, so only the word that opts in with \3a is visibly boxed.
        box_fill = _inline_color(profile.get("captions.box_color"))
        box_text = _inline_color(profile.get("captions.box_text"))
        spec["border_style"] = 3
        spec["outline"] = float(profile.get("captions.box_padding"))
        spec["shadow"] = 0.0
        spec["outline_color"] = "&HFF000000"          # fully transparent
        spec["active_open"] = f"{{\\3a&H00&\\3c{box_fill}\\c{box_text}}}"
        spec["active_close"] = f"{{\\3a&HFF&\\c{primary}}}"
    elif style == "word":
        spec["one_word"] = True
        spec["font_size"] = int(round(font_size * float(profile.get("captions.word_size_boost"))))
        spec["outline"] = float(profile.get("captions.outline")) + 2
        # A single word needs no colour change to read as active; a scale-in
        # gives it the beat instead.
        spec["active_open"] = "{\\fscx88\\fscy88\\t(0,120,\\fscx100\\fscy100)}"
        spec["active_close"] = ""
    return spec


def build_ass(lines: list[CaptionLine], profile, *, width: int | None = None,
              height: int | None = None) -> str:
    """Render caption lines to an ASS subtitle script."""
    width = width or int(profile.get("output.width"))
    height = height or int(profile.get("output.height"))

    font = profile.get("captions.font")
    bold = -1 if profile.get("captions.bold") else 0
    primary = _ass_color(profile.get("captions.primary"))
    margin_x = int(profile.get("captions.margin_x"))
    y_pct = float(profile.get("captions.y_pct"))
    max_chars = int(profile.get("captions.max_chars"))
    primary_inline = _inline_color(profile.get("captions.primary"))

    spec = _style_spec(profile, int(profile.get("captions.font_size")))
    # The letters' own shape: width and height as percentages, spacing in
    # pixels. These are libass's ScaleX, ScaleY and Spacing style fields.
    scale_x = max(20.0, min(400.0, float(profile.get("captions.scale_x", 100))))
    scale_y = max(20.0, min(400.0, float(profile.get("captions.scale_y", 100))))
    spacing = max(-20.0, min(100.0, float(profile.get("captions.spacing", 0))))

    use_emphasis = bool(profile.get("captions.emphasis"))
    extra_emphasis = emphasis_set(profile.get("captions.emphasis_words"))
    emphasis_inline = _inline_color(profile.get("captions.emphasis_color"))
    # Scaling one word does not reflow the line, so any scale above 1 risks the word
    # overlapping its neighbour. Colour alone marks importance safely; the scale knob
    # is there for anyone who wants it and accepts the tighter spacing.
    emphasis_scale = int(round(float(profile.get("captions.emphasis_scale")) * 100))
    if emphasis_scale == 100:
        emphasis_open = f"{{\\c{emphasis_inline}}}"
        emphasis_close = f"{{\\c{primary_inline}}}"
    else:
        emphasis_open = f"{{\\c{emphasis_inline}\\fscx{emphasis_scale}\\fscy{emphasis_scale}}}"
        emphasis_close = f"{{\\c{primary_inline}\\fscx100\\fscy100}}"

    margin_v = int(round(height * (1.0 - y_pct)))
    if profile.get("captions.safe_area"):
        margin_v = max(margin_v, int(SAFE_BOTTOM * height / 1920))

    header = [
        "[Script Info]",
        "; Generated by ReelForge",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",              # we do our own wrapping
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Reel,{font},{spec['font_size']},{primary},{primary},{spec['outline_color']},"
        f"&H80000000,{bold},0,0,0,{scale_x:g},{scale_y:g},{spacing:g},0,{spec['border_style']},"
        f"{spec['outline']:g},{spec['shadow']:g},2,{margin_x},{margin_x},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    for line in lines:
        texts = [_escape(w.text) for w in line.words]
        texts_raw = [w.text for w in line.words]
        emphatic = [use_emphasis and is_emphatic(w.text, extra_emphasis) for w in line.words]
        rows = _wrap_rows(texts_raw, max_chars)
        # Any override tag in the line - the active-word marker or an emphasised
        # word - means we must order the words for display ourselves.
        tagged = spec["per_word"] or any(emphatic)

        def render(active: int | None) -> str:
            """The full line, with the active word marked and important words kept marked."""
            if spec["one_word"]:
                index = active if active is not None else 0
                return (f"{spec['line_prefix']}{spec['active_open']}"
                        f"{texts[index]}{spec['active_close']}")
            parts: list[str] = []
            for row in rows:
                chunk: list[str] = []
                # Override tags cost us libass's bidi, so lay the words out
                # ourselves whenever the line carries any.
                order = row
                if tagged:
                    order = [row[i] for i in visual_order([texts_raw[j] for j in row])]
                for index in order:
                    word = texts[index]
                    if active is not None and index == active:
                        chunk.append(f"{spec['active_open']}{word}{spec['active_close']}")
                    elif emphatic[index]:
                        chunk.append(f"{emphasis_open}{word}{emphasis_close}")
                    else:
                        chunk.append(word)
                parts.append(" ".join(chunk))
            return spec["line_prefix"] + "\\N".join(parts)

        if not spec["per_word"] or len(line.words) == 1:
            active = 0 if (spec["one_word"] or len(line.words) == 1) else None
            if not spec["per_word"]:
                active = None
            events.append(
                f"Dialogue: 0,{_timestamp(line.start)},{_timestamp(line.end)},Reel,,0,0,0,,"
                f"{render(active)}"
            )
            continue

        # One event per word. Events tile the line exactly - any gap would make
        # the caption blink between words.
        for index, word in enumerate(line.words):
            start = line.start if index == 0 else max(line.start, word.start)
            if index + 1 < len(line.words):
                end = max(start + 0.02, min(line.end, line.words[index + 1].start))
            else:
                end = line.end
            if end <= start:
                continue
            events.append(
                f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},Reel,,0,0,0,,{render(index)}"
            )

    return "\n".join(header + events) + "\n"


def build_srt(lines: list[CaptionLine]) -> str:
    """Plain SRT for uploading as a separate subtitle track."""
    def stamp(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    blocks = []
    for index, line in enumerate(lines, start=1):
        blocks.append(f"{index}\n{stamp(line.start)} --> {stamp(line.end)}\n{line.text}\n")
    return "\n".join(blocks)


def write_captions(lines: list[CaptionLine], profile, out_dir: Path, stem: str,
                   *, width: int | None = None, height: int | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ass_path = out_dir / f"{stem}.ass"
    srt_path = out_dir / f"{stem}.srt"
    ass_path.write_text(build_ass(lines, profile, width=width, height=height), encoding="utf-8")
    srt_path.write_text(build_srt(lines), encoding="utf-8")
    return {"ass": ass_path, "srt": srt_path}


def apply_text_edit(line: CaptionLine, new_text: str) -> CaptionLine:
    """Rewrite a caption line's text, keeping the timing sensible.

    When the word count is unchanged each word keeps its own timing - the common
    case when you fix a misheard word, and the one that preserves karaoke sync.
    Otherwise the line's span is redistributed proportionally to word length.
    """
    new_text = clean_for_display(new_text)
    words = [w for w in new_text.split(" ") if w]
    if not words:
        return line

    if len(words) == len(line.words):
        for word, text in zip(line.words, words):
            word.text = text
        return line

    span = max(0.2, line.end - line.start)
    total = sum(len(w) for w in words) or len(words)
    rebuilt: list[Word] = []
    cursor = line.start
    for text in words:
        share = span * (len(text) / total)
        rebuilt.append(Word(text=text, start=round(cursor, 3),
                            end=round(cursor + share, 3), prob=1.0))
        cursor += share
    line.words = rebuilt
    line.end = max(line.end, cursor)
    return line

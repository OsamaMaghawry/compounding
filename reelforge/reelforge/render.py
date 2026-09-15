"""EDL -> ffmpeg filtergraph -> finished vertical video.

One ffmpeg process does everything: retime around the cuts, reframe to 9:16,
apply the zoom curve, composite b-roll layers, burn captions, normalise loudness.

The zoom curve is the subtle part. Moves are chained end-to-start and the factor
holds between them, so the expression below is continuous - the framing never
snaps back to 1.0 between punch-ins.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .captions import build_ass
from .edl import EDL, Overlay, Transition, Zoom
from .ffmpeg import FFmpegError, probe, run, run_filtergraph
from .profile import StyleProfile

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _escape_path(path: str | Path) -> str:
    """Escape a path for use inside a filtergraph argument.

    Only used for paths we cannot make relative. A Windows path carries both a
    drive colon and backslashes, which are the filtergraph's own separator and
    escape characters - so the renderer avoids embedding absolute paths at all
    (see `Renderer.render`, which runs ffmpeg inside the work directory and
    refers to the subtitle file and font folder by bare name). This remains for
    the cases that cannot be made relative.
    """
    text = str(path).replace("\\", "/")          # Windows accepts forward slashes
    for char in ("'", ":", "[", "]", ",", ";"):
        text = text.replace(char, "\\" + char)
    return text


def sync_fonts(fonts_dir: Path | None, work_dir: Path) -> str | None:
    """Put the fonts next to the subtitle file so libass needs no path.

    Copying a few hundred KB per render is a cheap price for never having to
    escape a font folder path inside a filtergraph.
    """
    if not fonts_dir or not Path(fonts_dir).exists():
        return None
    local = work_dir / "fonts"
    local.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in Path(fonts_dir).iterdir():
        if source.suffix.lower() not in (".ttf", ".otf", ".ttc"):
            continue
        target = local / source.name
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            shutil.copy2(source, target)
        copied += 1
    return "fonts" if copied else None


# --------------------------------------------------------------- zoom curve

def zoom_expression(zooms: list[Zoom], *, ease: str = "smooth", var: str = "in_time") -> str:
    """Piecewise-continuous zoom factor as an ffmpeg expression."""
    moves = sorted([z for z in zooms if z.enabled and z.out_end > z.out_start],
                   key=lambda z: z.out_start)
    if not moves:
        return "1.0"

    def ramp(move: Zoom) -> str:
        span = max(1e-3, move.out_end - move.out_start)
        progress = f"(({var}-{move.out_start:.3f})/{span:.3f})"
        shaped = f"({progress}*{progress}*(3-2*{progress}))" if ease == "smooth" else progress
        delta = move.end_factor - move.start_factor
        return f"({move.start_factor:.4f}+({delta:.4f})*{shaped})"

    # Build the nested conditional from the last move backwards.
    expression = f"{moves[-1].end_factor:.4f}"
    for move in reversed(moves):
        expression = (f"if(lt({var},{move.out_start:.3f}),HOLD,"
                      f"if(lt({var},{move.out_end:.3f}),{ramp(move)},{expression}))")
        expression = expression.replace("HOLD", f"{move.start_factor:.4f}", 1)
    return expression


def punch_expression(transitions: list[Transition], *, var: str = "in_time") -> str:
    """A multiplier that spikes on each punch transition and decays back to 1.

    Multiplying this into the zoom curve keeps the two independent: a punch lands
    on top of whatever framing the zoom happens to be holding.
    """
    punches = [t for t in transitions if t.enabled and t.kind == "punch" and t.duration > 0]
    if not punches:
        return ""
    expression = "1"
    for punch in sorted(punches, key=lambda t: t.out_time):
        amp = 0.16 * max(0.0, min(1.0, punch.strength))
        start, end = punch.out_time, punch.out_time + punch.duration
        decay = f"(1-(({var}-{start:.3f})/{punch.duration:.3f}))"
        expression = (f"if(between({var},{start:.3f},{end:.3f}),"
                      f"(1+{amp:.4f}*{decay}),{expression})")
    return expression


def max_punch_factor(transitions: list[Transition]) -> float:
    punches = [t for t in transitions if t.enabled and t.kind == "punch"]
    if not punches:
        return 1.0
    return 1.0 + 0.16 * max(max(0.0, min(1.0, t.strength)) for t in punches)


def flash_expression(transitions: list[Transition]) -> str:
    """Brightness lift that decays over each flash transition."""
    flashes = [t for t in transitions if t.enabled and t.kind == "flash" and t.duration > 0]
    if not flashes:
        return ""
    expression = "0"
    for flash in sorted(flashes, key=lambda t: t.out_time):
        amp = 0.75 * max(0.0, min(1.0, flash.strength))
        start, end = flash.out_time, flash.out_time + flash.duration
        decay = f"(1-((t-{start:.3f})/{flash.duration:.3f}))"
        expression = (f"if(between(t,{start:.3f},{end:.3f}),"
                      f"({amp:.4f}*{decay}),{expression})")
    return expression


def blur_enable_expression(transitions: list[Transition]) -> str:
    """A single enable expression covering every blur window."""
    blurs = [t for t in transitions if t.enabled and t.kind == "blur" and t.duration > 0]
    if not blurs:
        return ""
    return "+".join(f"between(t,{b.out_time:.3f},{b.out_time + b.duration:.3f})" for b in blurs)


# ------------------------------------------------------------------ graph

@dataclass
class RenderPlan:
    args: list[str]
    filtergraph: str
    ass_path: Path | None
    output: Path


class Renderer:
    def __init__(self, profile: StyleProfile, *, work_dir: Path, fonts_dir: Path | None = None):
        self.profile = profile
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir = Path(fonts_dir) if fonts_dir else None

    # -- pieces ----------------------------------------------------------
    def _reframe(self, label_in: str, label_out: str, canvas_w: int, canvas_h: int) -> str:
        mode = self.profile.get("reframe.mode")
        if self.profile.get("reframe.blur_background"):
            # Blurred fill behind a contained copy - for sources that are not 9:16.
            return (
                f"[{label_in}]split=2[bg_src][fg_src];"
                f"[bg_src]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},gblur=sigma=28[bg];"
                f"[fg_src]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[{label_out}];"
            )
        if mode == "pad":
            color = self.profile.get("reframe.pad_color")
            return (f"[{label_in}]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
                    f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color={color}[{label_out}];")
        x = {"left": "0", "right": "iw-ow"}.get(mode, "(iw-ow)/2")
        return (f"[{label_in}]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h}:{x}:(ih-oh)/2[{label_out}];")

    def _overlay_chain(self, overlay: Overlay, input_index: int, base: str, out_label: str,
                       width: int, height: int) -> str:
        """Scale and position one b-roll layer, then composite it for its window."""
        src = f"{input_index}:v"
        prepared = f"ov{input_index}"
        mode = overlay.mode

        if mode == "pip":
            scale = float(self.profile.get("broll.pip_scale"))
            pip_w, pip_h = _even(width * scale), _even(height * scale * 0.62)
            chain = (f"[{src}]scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
                     f"crop={pip_w}:{pip_h},setsar=1")
            y = f"{int(height * 0.12)}" if self.profile.get("broll.pip_pos") == "top" \
                else f"{int(height * 0.62)}"
            position = f"(W-w)/2:{y}"
        elif mode == "band":
            band_h = _even(height * 0.34)
            chain = (f"[{src}]scale={width}:{band_h}:force_original_aspect_ratio=increase,"
                     f"crop={width}:{band_h},setsar=1")
            position = f"0:{int(height * 0.08)}"
        else:  # cover
            chain = (f"[{src}]scale={width}:{height}:force_original_aspect_ratio=increase,"
                     f"crop={width}:{height},setsar=1")
            position = "0:0"

        if overlay.opacity < 0.999:
            chain += f",format=yuva420p,colorchannelmixer=aa={overlay.opacity:.3f}"
        # Shift the layer's own timeline so frame 0 of the asset lands at out_start.
        chain += f",setpts=PTS-STARTPTS+{overlay.out_start:.3f}/TB[{prepared}];"

        composite = (f"[{base}][{prepared}]overlay={position}:"
                     f"enable='between(t,{overlay.out_start:.3f},{overlay.out_end:.3f})':"
                     f"eof_action=pass:shortest=0[{out_label}];")
        return chain + composite

    def _audio_chain(self, edl: EDL, label_in: str, label_out: str) -> str:
        parts: list[str] = []
        current = label_in

        ducking = [o for o in edl.active_overlays() if o.audio == "duck"]
        if ducking:
            duck_db = float(edl.audio.get("duck_db", -12.0))
            gain = 10 ** (duck_db / 20.0)
            windows = "+".join(f"between(t,{o.out_start:.3f},{o.out_end:.3f})" for o in ducking)
            parts.append(f"[{current}]volume='if({windows},{gain:.4f},1.0)':eval=frame[aduck];")
            current = "aduck"

        if edl.audio.get("loudnorm", True):
            target = float(edl.audio.get("target_lufs", -14.0))
            parts.append(f"[{current}]loudnorm=I={target}:TP=-1.5:LRA=11[{label_out}];")
        else:
            parts.append(f"[{current}]anull[{label_out}];")
        return "".join(parts)

    # -- plan ------------------------------------------------------------
    def plan(self, edl: EDL, output: Path, *, preview: bool = False,
             burn_captions: bool = True) -> RenderPlan:
        source = Path(edl.source).expanduser().resolve()
        if not source.exists():
            raise FFmpegError(f"source file is gone: {source}")
        info = probe(source)

        width = int(edl.output.get("width", self.profile.get("output.width")))
        height = int(edl.output.get("height", self.profile.get("output.height")))
        fps = int(edl.output.get("fps", self.profile.get("output.fps")))
        if preview:
            width, height = _even(width / 2), _even(height / 2)

        headroom = float(self.profile.get("output.zoom_headroom"))
        zooms = edl.active_zooms()
        transitions = edl.active_transitions()
        max_factor = max([z.end_factor for z in zooms] + [z.start_factor for z in zooms] + [1.0])
        max_factor *= max_punch_factor(transitions)
        headroom = max(1.0, min(headroom, max_factor)) if (zooms or transitions) else 1.0
        canvas_w, canvas_h = _even(width * headroom), _even(height * headroom)

        args: list[str] = ["-i", str(source)]
        overlays = edl.active_overlays()
        for overlay in overlays:
            asset = Path(overlay.asset).expanduser().resolve()
            if not asset.exists():
                continue
            if asset.suffix.lower() in IMAGE_EXT:
                args += ["-loop", "1", "-framerate", str(fps),
                         "-t", f"{overlay.duration:.3f}", "-i", str(asset)]
            else:
                args += ["-ss", f"{overlay.asset_start:.3f}",
                         "-t", f"{overlay.duration:.3f}", "-i", str(asset)]

        graph: list[str] = []
        has_audio = info.has_audio

        # 1. Retime: keep only the surviving slices, then join them.
        cuts = edl.cuts or []
        whole = (len(cuts) == 1 and cuts[0].src_start <= 0.02
                 and cuts[0].src_end >= info.duration - 0.05)
        if not cuts or whole:
            graph.append("[0:v]null[cv];")
            if has_audio:
                graph.append("[0:a]anull[ca];")
        else:
            for index, cut in enumerate(cuts):
                graph.append(
                    f"[0:v]trim=start={cut.src_start:.3f}:end={cut.src_end:.3f},"
                    f"setpts=PTS-STARTPTS[v{index}];"
                )
                if has_audio:
                    graph.append(
                        f"[0:a]atrim=start={cut.src_start:.3f}:end={cut.src_end:.3f},"
                        f"asetpts=PTS-STARTPTS[a{index}];"
                    )
            pairs = "".join(f"[v{i}][a{i}]" if has_audio else f"[v{i}]" for i in range(len(cuts)))
            if has_audio:
                graph.append(f"{pairs}concat=n={len(cuts)}:v=1:a=1[cv][ca];")
            else:
                graph.append(f"{pairs}concat=n={len(cuts)}:v=1:a=0[cv];")

        graph.append(f"[cv]fps={fps},setsar=1[cvf];")

        # 2. Reframe to the vertical canvas (oversized when we plan to zoom).
        graph.append(self._reframe("cvf", "canvas", canvas_w, canvas_h))

        # 3. Zoom curve, with any punch transitions multiplied on top of it.
        punch = punch_expression(transitions)
        if zooms or punch:
            expression = zoom_expression(zooms, ease=self.profile.get("zoom.ease"))
            if punch:
                expression = f"({expression})*({punch})"
            graph.append(
                f"[canvas]zoompan=z='{expression}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:s={width}x{height}:fps={fps}[zv];"
            )
        else:
            graph.append(f"[canvas]scale={width}:{height}[zv];")

        # 3b. Flash and blur transitions.
        base_tr = "zv"
        flash = flash_expression(transitions)
        if flash:
            graph.append(f"[{base_tr}]eq=brightness='{flash}':eval=frame[trf];")
            base_tr = "trf"
        blur_windows = blur_enable_expression(transitions)
        if blur_windows:
            sigma = 6.0 + 14.0 * max((t.strength for t in transitions if t.kind == "blur"),
                                     default=0.7)
            graph.append(f"[{base_tr}]gblur=sigma={sigma:.1f}:enable='{blur_windows}'[trb];")
            base_tr = "trb"

        # 4. B-roll layers.
        base = base_tr
        input_index = 1
        for number, overlay in enumerate(overlays):
            if not Path(overlay.asset).exists():
                continue
            label = f"ovl{number}"
            graph.append(self._overlay_chain(overlay, input_index, base, label, width, height))
            base = label
            input_index += 1

        # 5. Burned captions.
        ass_path = None
        if burn_captions and edl.captions and self.profile.get("captions.enabled"):
            ass_path = self.work_dir / "captions.ass"
            ass_path.write_text(
                build_ass(edl.captions, self.profile, width=width, height=height),
                encoding="utf-8",
            )
            # Referenced by bare name: ffmpeg is run with the work directory as its
            # cwd, so no absolute path - and therefore no drive colon or backslash -
            # ever reaches the filtergraph parser.
            ass_filter = f"ass={ass_path.name}"
            local_fonts = sync_fonts(self.fonts_dir, self.work_dir)
            if local_fonts:
                ass_filter += f":fontsdir={local_fonts}"
            graph.append(f"[{base}]{ass_filter}[sub];")
            base = "sub"

        graph.append(f"[{base}]format=yuv420p[outv];")
        if has_audio:
            graph.append(self._audio_chain(edl, "ca", "outa"))

        filtergraph = "".join(graph).rstrip(";")

        preset = "ultrafast" if preview else self.profile.get("output.preset")
        crf = 28 if preview else int(self.profile.get("output.crf"))

        args += ["-map", "[outv]"]
        if has_audio:
            args += ["-map", "[outa]", "-c:a", "aac",
                     "-b:a", self.profile.get("output.audio_bitrate")]
        else:
            args += ["-an"]
        args += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                 "-profile:v", "high", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(output)]

        return RenderPlan(args=args, filtergraph=filtergraph, ass_path=ass_path, output=Path(output))

    # -- run -------------------------------------------------------------
    def render(self, edl: EDL, output: str | Path, *, preview: bool = False,
               burn_captions: bool = True, on_status=None) -> Path:
        # Absolute, because ffmpeg runs with the work directory as its cwd.
        output = Path(output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        plan = self.plan(edl, output, preview=preview, burn_captions=burn_captions)

        # Filtergraphs get long; pass via file so we never hit an argv limit.
        script = self.work_dir / ("filtergraph_preview.txt" if preview else "filtergraph.txt")

        if on_status:
            on_status(f"rendering {'preview' if preview else 'final'} -> {output.name}")

        # Inputs first, then the filtergraph, then mapping/encoding args.
        split = plan.args.index("-map")
        run_filtergraph(plan.args[:split], plan.filtergraph, script, plan.args[split:],
                        cwd=self.work_dir)
        if not output.exists() or output.stat().st_size == 0:
            raise FFmpegError(f"render produced no output at {output}")
        return output


def check_font(profile: StyleProfile, fonts_dir: Path | None) -> str | None:
    """Warn when the caption font is missing - Arabic falls back to boxes otherwise."""
    name = (profile.get("captions.font") or "").strip()
    if not name:
        return None
    if fonts_dir and Path(fonts_dir).exists():
        for path in Path(fonts_dir).glob("*"):
            if path.suffix.lower() in (".ttf", ".otf", ".ttc") and \
               name.lower().replace(" ", "") in path.stem.lower().replace(" ", ""):
                return None
    if shutil.which("fc-list"):
        proc = run(["fc-list", ":", "family"], check=False)
        if name.lower() in (proc.stdout or "").lower():
            return None
    return (f"font '{name}' was not found in {fonts_dir or 'the system font list'}. "
            f"Run `reelforge setup` to fetch an Arabic font, or set captions.font "
            f"to one you have installed.")

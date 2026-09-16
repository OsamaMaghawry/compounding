"""Joining several takes into one timeline before editing.

People do not shoot a Reel as one file. They shoot six takes on a phone and
expect the editor to treat them as one video. Clips from a phone can differ in
resolution, frame rate, rotation and audio sample rate, and one of them is
usually a silent b-roll shot - so everything is normalised to a common shape
before concatenation, and a clip with no audio gets real silence rather than
being dropped or knocking the audio out of sync.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ffmpeg import (FFmpegError, MediaInfo, ffmpeg_bin, probe, run, run_ffmpeg,
                     run_watched)


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def target_shape(infos: list[MediaInfo], *,
                 max_height: int | None = None) -> tuple[int, int, int, int]:
    """A canvas every clip fits inside, plus a frame rate and audio rate.

    Sized from the clips that share the dominant orientation, not from all of
    them: taking the maximum width and height across a mixed set turns six
    portrait phone takes plus one landscape screen recording into a square
    canvas, padding every clip and wasting most of the frame. Ties go to
    portrait - the output is vertical either way.
    """
    portrait = [info for info in infos if info.height >= info.width]
    landscape = [info for info in infos if info.height < info.width]
    dominant = portrait if len(portrait) >= len(landscape) else landscape

    width = _even(max(info.width for info in dominant))
    height = _even(max(info.height for info in dominant))

    # Phones shoot 4K. The renderer never samples above the output size times the
    # zoom headroom, so joining at 2160x3840 spends minutes encoding pixels that
    # are thrown away on the next pass. Cap the long edge at what can actually be
    # used, keeping the aspect ratio.
    if max_height and height > max_height:
        scale = max_height / height
        width, height = _even(width * scale), _even(max_height)

    fps = max(1, int(round(max(info.fps for info in infos))))
    rates = [info.audio_rate for info in infos if info.audio_rate]
    return width, height, min(fps, 60), (max(rates) if rates else 48000)


def can_be_stuck_together(infos: list[MediaInfo], width: int, height: int,
                          fps: int, audio_rate: int) -> bool:
    """Whether these clips can simply follow one another, untouched.

    Takes shot back to back on one phone almost always can: same camera, same
    mode, same everything. Noticing that turns minutes of re-encoding into a
    copy, which is the difference between waiting and not.
    """
    first = infos[0]
    if not first.has_audio or first.vcodec != "h264" or first.acodec != "aac":
        return False
    if (first.width, first.height) != (width, height):
        return False                      # it would have to be resized anyway
    if abs(first.fps - fps) > 0.02 or first.audio_rate != audio_rate:
        return False
    return all(
        info.has_audio
        and (info.width, info.height) == (width, height)
        and info.vcodec == first.vcodec and info.acodec == first.acodec
        and info.pix_fmt == first.pix_fmt and info.channels == first.channels
        and info.audio_rate == first.audio_rate
        and abs(info.fps - first.fps) <= 0.02
        and info.rotation == first.rotation
        for info in infos
    )


def _copy_together(sources: list[Path], dest: Path, expected: float,
                   on_status=None) -> bool:
    """Stick the clips end to end without re-encoding. False if it would not do."""
    listing = dest.with_suffix(".concat.txt")
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in sources), encoding="utf-8")
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(dest)])
    except FFmpegError:
        return False
    finally:
        listing.unlink(missing_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        return False
    # Trust it only if the result is as long as the parts. A copy that silently
    # produced something shorter is worse than the slow path.
    try:
        made = probe(dest).duration
    except FFmpegError:
        return False
    if abs(made - expected) > max(0.6, expected * 0.02):
        return False
    if on_status:
        on_status(f"the clips already match, so they were joined without "
                  f"re-encoding ({made:.0f}s)")
    return True


def join_clips(paths: list[str | Path], dest: str | Path, *, preset: str = "superfast",
               crf: int = 20, max_height: int | None = None, on_status=None,
               on_fraction=None, owner: int | None = None) -> Path:
    """Concatenate clips into one file, normalising shape, rate and audio."""
    sources = [Path(p) for p in paths]
    if not sources:
        raise FFmpegError("no clips to join")
    if len(sources) == 1:
        return sources[0]

    infos = [probe(path) for path in sources]
    width, height, fps, audio_rate = target_shape(infos, max_height=max_height)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Joining re-encodes every frame, which on 4K phone clips is minutes of work.
    # Doing it again on an unchanged set of clips is pure waste, so remember it.
    signature = _signature(sources, width, height, fps, audio_rate)
    stamp = dest.with_suffix(".join.json")
    if dest.exists() and stamp.exists():
        try:
            if json.loads(stamp.read_text("utf-8")).get("signature") == signature:
                if on_status:
                    on_status(f"reusing the joined timeline from last time ({dest.name})")
                return dest
        except (json.JSONDecodeError, OSError):
            pass

    total = sum(info.duration for info in infos)
    if can_be_stuck_together(infos, width, height, fps, audio_rate) and \
            _copy_together(sources, dest, total, on_status=on_status):
        stamp.write_text(json.dumps({"signature": signature}), encoding="utf-8")
        return dest

    if on_status:
        on_status(f"joining {len(sources)} clips into one {width}x{height} timeline "
                  f"({total:.0f}s) - every frame is re-encoded, which on 4K takes "
                  f"minutes; recording at 1080p makes this step almost free")

    args: list[str] = []
    for path in sources:
        args += ["-i", str(path)]

    silent = [index for index, info in enumerate(infos) if not info.has_audio]
    silence_input = None
    if silent:
        silence_input = len(sources)
        args += ["-f", "lavfi", "-t", "0.1", "-i",
                 f"anullsrc=channel_layout=stereo:sample_rate={audio_rate}"]

    graph: list[str] = []
    for index, info in enumerate(infos):
        # Fit inside the canvas and pad - never crop, the vertical reframe does that later.
        graph.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{index}];"
        )
        if info.has_audio:
            graph.append(f"[{index}:a]aresample={audio_rate},aformat="
                         f"sample_fmts=fltp:channel_layouts=stereo[a{index}];")
        else:
            # Real silence for the clip's full length keeps every later clip in sync.
            graph.append(
                f"[{silence_input}:a]atrim=0:{max(info.duration, 0.05):.3f},asetpts=PTS-STARTPTS,"
                f"aresample={audio_rate},aformat=sample_fmts=fltp:channel_layouts=stereo"
                f"[a{index}];"
            )

    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(sources)))
    graph.append(f"{pairs}concat=n={len(sources)}:v=1:a=1[outv][outa]")

    # This is an intermediate - the renderer encodes the real output from it -
    # so it is worth trading file size for time here. And it reports where it has
    # got to, because minutes of silence is indistinguishable from a hang.
    run_watched([ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                 *args, "-filter_complex", "".join(graph),
                 "-map", "[outv]", "-map", "[outa]",
                 "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                 "-movflags", "+faststart", str(dest)],
                total=total, on_fraction=on_fraction, owner=owner)

    if not dest.exists() or dest.stat().st_size == 0:
        raise FFmpegError(f"joining produced no output at {dest}")
    stamp.write_text(json.dumps({"signature": signature}), encoding="utf-8")
    return dest


def _signature(sources: list[Path], width: int, height: int, fps: int,
               audio_rate: int) -> str:
    """Identifies this exact set of clips joined to this exact shape."""
    parts = []
    for path in sources:
        stat = path.stat()
        parts.append(f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}")
    parts.append(f"{width}x{height}@{fps}:{audio_rate}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def clip_boundaries(paths: list[str | Path]) -> list[float]:
    """Where each clip starts in the joined timeline - the natural cut points."""
    boundaries: list[float] = []
    cursor = 0.0
    for path in paths[:-1]:
        cursor += probe(path).duration
        boundaries.append(round(cursor, 3))
    return boundaries

"""Thin, dependency-free wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


class FFmpegMissing(FFmpegError):
    pass


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegMissing(
            f"`{name}` was not found on PATH.\n"
            "Install it first:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg"
        )
    return path


def ffmpeg_bin() -> str:
    return _binary("ffmpeg")


def ffprobe_bin() -> str:
    return _binary("ffprobe")


def run(args: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising FFmpegError with the tail of stderr on failure."""
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        errors="replace",
    )
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise FFmpegError(f"command failed ({proc.returncode}): {' '.join(args[:6])} ...\n{tail}")
    return proc


def run_ffmpeg(args: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with sane defaults. Returns the completed process (stderr captured)."""
    base = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y"]
    if quiet:
        base += ["-loglevel", "error"]
    return run(base + args)


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_rate: int
    rotation: int
    size_bytes: int

    @property
    def is_vertical(self) -> bool:
        return self.height >= self.width

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def _parse_fps(rate: str) -> float:
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    """Read stream/format metadata for a media file."""
    path = Path(path)
    if not path.exists():
        raise FFmpegError(f"file not found: {path}")
    proc = run([
        ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise FFmpegError(f"no video stream in {path}")

    rotation = 0
    for entry in video.get("side_data_list", []) or []:
        if "rotation" in entry:
            rotation = int(entry["rotation"]) % 360
    if not rotation:
        tag = (video.get("tags") or {}).get("rotate")
        if tag:
            try:
                rotation = int(tag) % 360
            except ValueError:
                rotation = 0

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    # ffmpeg auto-applies display rotation when decoding, so report post-rotation dims.
    if rotation in (90, 270):
        width, height = height, width

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")

    return MediaInfo(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps or 30.0,
        has_audio=audio is not None,
        audio_rate=int(audio.get("sample_rate") or 0) if audio else 0,
        rotation=rotation,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
    )


def extract_pcm(path: str | Path, *, rate: int = 16000, start: float | None = None,
                duration: float | None = None) -> bytes:
    """Decode audio to raw mono 16-bit PCM at `rate` Hz and return the bytes."""
    args = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
        raise FFmpegError("audio decode failed:\n" + "\n".join(tail))
    return proc.stdout


def extract_wav(path: str | Path, dest: str | Path, *, rate: int = 16000) -> Path:
    """Write a mono 16 kHz WAV next to the source - the format every ASR engine wants."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
                "-c:a", "pcm_s16le", str(dest)])
    return dest


def has_encoder(name: str) -> bool:
    try:
        proc = run([ffmpeg_bin(), "-hide_banner", "-encoders"], check=False)
    except FFmpegMissing:
        return False
    return name in (proc.stdout or "")


def build_config() -> str:
    """The ffmpeg `configuration:` line - used to check for libass/fribidi/harfbuzz."""
    proc = run([ffmpeg_bin(), "-version"], check=False)
    return proc.stdout or ""

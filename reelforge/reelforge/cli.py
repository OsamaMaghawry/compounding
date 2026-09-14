"""Command line interface.

    reelforge auto clip.mp4                 # the one command you will actually use
    reelforge auto clip.mp4 --review        # ...and open the review page
    reelforge captions clip.mp4             # accurate Arabic subtitles only
    reelforge learn                         # what has it picked up from you so far
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from . import __version__
from .edl import EDL
from .ffmpeg import FFmpegError, FFmpegMissing, build_config, probe
from .learn import FeedbackStore
from .pipeline import PACKAGE_ROOT, AutoEditor
from .profile import StyleProfile
from .speech import available_backend

PROFILE_DIR = PACKAGE_ROOT / "profiles"
FONTS = {
    "Cairo.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
    "Tajawal-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/tajawal/Tajawal-Bold.ttf",
    "Almarai-ExtraBold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/almarai/Almarai-ExtraBold.ttf",
}


def _print(message: str) -> None:
    print(f"  {message}", flush=True)


def _profile_from_args(args) -> StyleProfile:
    profile = StyleProfile.resolve(getattr(args, "profile", None), [PROFILE_DIR])
    overrides: list[str] = list(getattr(args, "set", None) or [])

    flag_map = {
        "no_captions": "captions.enabled=false",
        "no_zoom": "zoom.enabled=false",
        "no_broll": "broll.enabled=false",
        "no_cuts": "cuts.enabled=false",
        "no_learning": "learning.enabled=false",
    }
    for flag, override in flag_map.items():
        if getattr(args, flag, False):
            overrides.append(override)

    for attr, key in (("asr_backend", "asr.backend"), ("model", "asr.model"),
                      ("lang", "asr.language"), ("broll_dir", "broll.library"),
                      ("font", "captions.font")):
        value = getattr(args, attr, None)
        if value:
            overrides.append(f"{key}={value}")
    if getattr(args, "vertical", None):
        width, _, height = args.vertical.partition("x")
        overrides += [f"output.width={width}", f"output.height={height}"]
    return profile.apply_overrides(overrides)


def _editor(args, profile: StyleProfile) -> AutoEditor:
    project = Path(args.project) if getattr(args, "project", None) else Path.cwd() / ".reelforge"
    return AutoEditor(profile, project_dir=project,
                      fonts_dir=PACKAGE_ROOT / "assets" / "fonts", on_status=_print)


# ------------------------------------------------------------------ commands

def cmd_auto(args) -> int:
    profile = _profile_from_args(args)
    editor = _editor(args, profile)
    source = Path(args.video).expanduser()

    started = time.time()
    result = editor.plan(source, refresh=args.refresh)
    edl = result.edl
    summary = edl.summary()

    for warning in result.warnings:
        _print(f"note: {warning}")
    _print(f"{summary['source_duration']}s -> {summary['output_duration']}s "
           f"({summary['removed']}s of dead air removed)")
    _print(f"{summary['cuts']} segments, {summary['zooms']} zoom moves, "
           f"{summary['overlays']} b-roll layers, {summary['caption_lines']} caption lines")

    output = Path(args.out) if args.out else source.with_name(f"{source.stem}-reel.mp4")
    edl_path = editor.runs_dir / f"run-{result.run_id}.edl.json"

    if args.plan_only:
        _print(f"edit plan: {edl_path}")
        print(json.dumps(summary, indent=2))
        return 0

    preview_path = editor.work_dir / "preview.mp4"
    if args.review:
        editor.render(edl, preview_path, preview=True)
        from .review import ReviewServer  # noqa: PLC0415 - only needed interactively
        server = ReviewServer(editor, result, output=output,
                              preview=preview_path, port=args.port)
        outcome = server.serve(open_browser=not args.no_browser)
        if not outcome:
            _print("nothing exported - the edit plan is still saved at "
                   f"{edl_path}")
            return 0
        _print(f"exported {outcome['output']}")
    else:
        editor.render(edl, output, preview=args.preview)
        _print(f"exported {output}")
        if args.accept:
            learned = editor.accept(result.run_id, edl)
            _print(f"recorded as accepted ({learned['zooms_kept']} zoom moves kept)")

    if args.srt:
        paths = editor.export_captions(edl, output.parent, output.stem)
        _print(f"subtitles: {paths['srt'].name}, {paths['ass'].name}")

    _print(f"done in {time.time() - started:.1f}s")
    return 0


def cmd_captions(args) -> int:
    args.no_zoom = args.no_broll = args.no_cuts = True
    profile = _profile_from_args(args)
    editor = _editor(args, profile)
    source = Path(args.video).expanduser()

    result = editor.plan(source, refresh=args.refresh)
    for warning in result.warnings:
        _print(f"note: {warning}")

    out_dir = Path(args.out).parent if args.out else source.parent
    stem = Path(args.out).stem if args.out else source.stem
    paths = editor.export_captions(result.edl, out_dir, stem)
    _print(f"{len(result.edl.captions)} caption lines -> {paths['srt']} / {paths['ass']}")

    if args.burn:
        output = Path(args.out) if args.out else source.with_name(f"{source.stem}-subbed.mp4")
        editor.render(result.edl, output, preview=args.preview)
        _print(f"burned into {output}")
    return 0


def cmd_render(args) -> int:
    profile = _profile_from_args(args)
    editor = _editor(args, profile)
    edl = EDL.load(args.edl)
    output = Path(args.out) if args.out else Path(edl.source).with_name(
        f"{Path(edl.source).stem}-reel.mp4")
    editor.render(edl, output, preview=args.preview)
    _print(f"exported {output}")
    return 0


def cmd_review(args) -> int:
    profile = _profile_from_args(args)
    editor = _editor(args, profile)
    store = editor.store
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    if not run:
        _print("no runs recorded yet - run `reelforge auto` first")
        return 1

    from .pipeline import PlanResult  # noqa: PLC0415
    from .review import ReviewServer  # noqa: PLC0415

    edl = EDL.from_dict(json.loads(run["proposed_edl"]))
    result = PlanResult(edl=edl, run_id=run["id"], analysis=None, transcript=None)
    preview_path = editor.work_dir / "preview.mp4"
    editor.render(edl, preview_path, preview=True)
    output = Path(args.out) if args.out else Path(edl.source).with_name(
        f"{Path(edl.source).stem}-reel.mp4")
    server = ReviewServer(editor, result, output=output, preview=preview_path, port=args.port)
    server.serve(open_browser=not args.no_browser)
    return 0


def cmd_learn(args) -> int:
    project = Path(args.project) if args.project else Path.cwd() / ".reelforge"
    store = FeedbackStore(project)

    if args.add_vocab:
        for pair in args.add_vocab:
            if "=" not in pair:
                _print(f"skipping '{pair}' - expected wrong=right")
                continue
            wrong, right = pair.split("=", 1)
            store.add_vocab(wrong.strip(), right.strip())
            _print(f"learned: {wrong.strip()} -> {right.strip()}")
        return 0

    if args.retrain:
        model = store.train_zoom_model(min_samples=args.min_samples)
        if not model:
            rows = store.training_rows("zoom")
            _print(f"not enough labelled examples yet ({len(rows)}/{args.min_samples}) - "
                   "review a few more edits first")
            return 0
        _print(f"trained on {model.samples} examples, accuracy {model.accuracy:.2f}")
        for name, weight in model.explain():
            _print(f"    {name:<18} {weight:+.3f}")
        return 0

    stats = store.stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["vocab_terms"]:
        _print(f"{stats['vocab_terms']} learned vocabulary fixes are biasing the next transcript")
    return 0


def cmd_setup(args) -> int:
    fonts_dir = PACKAGE_ROOT / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FONTS.items():
        target = fonts_dir / name
        if target.exists() and not args.force:
            _print(f"have {name}")
            continue
        try:
            _print(f"downloading {name}")
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        except Exception as exc:
            _print(f"could not fetch {name}: {exc}")
    _print(f"fonts in {fonts_dir}")
    return cmd_doctor(args)


def cmd_doctor(args) -> int:
    ok = True
    try:
        config = build_config()
        first = config.splitlines()[0] if config else "ffmpeg"
        _print(f"ffmpeg: {first.split('Copyright')[0].strip()}")
        for feature in ("libass", "libfribidi", "libharfbuzz", "libfreetype"):
            present = feature in config
            _print(f"  {feature:<12} {'yes' if present else 'NO - Arabic captions will break'}")
            ok = ok and present
    except FFmpegMissing as exc:
        _print(str(exc))
        return 1

    backend = available_backend("auto")
    _print(f"speech backend: {backend}")
    if backend == "stub":
        _print("  no real model installed. For accurate Arabic:")
        _print("  pip install -r requirements-asr.txt")
        ok = False
    else:
        try:
            import ctranslate2  # noqa: PLC0415
            count = ctranslate2.get_cuda_device_count()
            _print(f"  gpu: {'cuda x' + str(count) if count else 'cpu only (slower)'}")
        except Exception:
            pass

    fonts_dir = PACKAGE_ROOT / "assets" / "fonts"
    found = sorted(p.name for p in fonts_dir.glob("*")
                   if p.suffix.lower() in (".ttf", ".otf", ".ttc")) if fonts_dir.exists() else []
    _print(f"fonts: {', '.join(found) if found else 'none - run `reelforge setup`'}")
    ok = ok and bool(found)

    project = Path(args.project) if getattr(args, "project", None) else Path.cwd() / ".reelforge"
    if project.exists():
        stats = FeedbackStore(project).stats()
        _print(f"learning: {stats['runs']} runs, {stats['reviewed']} reviewed, "
               f"{stats['vocab_terms']} vocabulary fixes")
    _print("ready" if ok else "usable, but see the notes above")
    return 0 if ok else 0


def cmd_probe(args) -> int:
    info = probe(args.video)
    print(json.dumps({
        "duration": round(info.duration, 2), "width": info.width, "height": info.height,
        "fps": round(info.fps, 2), "audio": info.has_audio, "rotation": info.rotation,
        "vertical": info.is_vertical,
    }, indent=2))
    return 0


# -------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelforge",
        description="Local AI auto-editor for vertical short-form video.",
    )
    parser.add_argument("--version", action="version", version=f"reelforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target, *, with_video: bool = True):
        if with_video:
            target.add_argument("video", help="input video file")
        target.add_argument("-o", "--out", help="output path")
        target.add_argument("-p", "--profile", help="profile name or path "
                            "(default, punchy, calm, captions_only)")
        target.add_argument("--set", action="append", metavar="KEY=VALUE",
                            help="override any profile value, repeatable")
        target.add_argument("--project", help="where to keep cache and learning data "
                            "(default ./.reelforge)")
        target.add_argument("--preview", action="store_true",
                            help="half resolution, fastest encode")
        target.add_argument("--refresh", action="store_true",
                            help="ignore cached analysis and transcript")
        target.add_argument("--asr-backend", choices=["auto", "faster-whisper", "whispercpp", "stub"])
        target.add_argument("--model", help="speech model, e.g. large-v3 or small")
        target.add_argument("--lang", help="spoken language code (default ar)")
        target.add_argument("--font", help="caption font family")
        target.add_argument("--broll-dir", help="folder of b-roll clips")
        target.add_argument("--vertical", metavar="WxH", help="output size, default 1080x1920")
        target.add_argument("--no-captions", action="store_true")
        target.add_argument("--no-zoom", action="store_true")
        target.add_argument("--no-broll", action="store_true")
        target.add_argument("--no-cuts", action="store_true")
        target.add_argument("--no-learning", action="store_true")

    auto = sub.add_parser("auto", help="analyse and edit a clip end to end")
    add_common(auto)
    auto.add_argument("--review", action="store_true",
                      help="open the local review page before exporting")
    auto.add_argument("--accept", action="store_true",
                      help="record this edit as approved without reviewing")
    auto.add_argument("--plan-only", action="store_true", help="decide but do not render")
    auto.add_argument("--srt", action="store_true", help="also write .srt and .ass files")
    auto.add_argument("--port", type=int, default=8733)
    auto.add_argument("--no-browser", action="store_true")
    auto.set_defaults(func=cmd_auto)

    captions = sub.add_parser("captions", help="Arabic subtitles only")
    add_common(captions)
    captions.add_argument("--burn", action="store_true", help="also render a subtitled video")
    captions.set_defaults(func=cmd_captions)

    render = sub.add_parser("render", help="render a saved edit plan")
    render.add_argument("edl", help="path to a run-N.edl.json file")
    add_common(render, with_video=False)
    render.set_defaults(func=cmd_render)

    review = sub.add_parser("review", help="reopen the review page for a run")
    review.add_argument("run_id", nargs="?", type=int, help="run id (default: the latest)")
    add_common(review, with_video=False)
    review.add_argument("--port", type=int, default=8733)
    review.add_argument("--no-browser", action="store_true")
    review.set_defaults(func=cmd_review)

    learn = sub.add_parser("learn", help="inspect or update what it has learned")
    learn.add_argument("--project")
    learn.add_argument("--retrain", action="store_true", help="retrain the zoom model now")
    learn.add_argument("--min-samples", type=int, default=40)
    learn.add_argument("--add-vocab", action="append", metavar="WRONG=RIGHT",
                       help="teach a correction directly, repeatable")
    learn.set_defaults(func=cmd_learn)

    setup = sub.add_parser("setup", help="download fonts and check the install")
    setup.add_argument("--force", action="store_true")
    setup.add_argument("--project")
    setup.set_defaults(func=cmd_setup)

    doctor = sub.add_parser("doctor", help="check ffmpeg, fonts and models")
    doctor.add_argument("--project")
    doctor.set_defaults(func=cmd_doctor)

    probe_cmd = sub.add_parser("probe", help="show what ffmpeg sees in a file")
    probe_cmd.add_argument("video")
    probe_cmd.set_defaults(func=cmd_probe)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FFmpegMissing as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except (FFmpegError, FileNotFoundError, ValueError) as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

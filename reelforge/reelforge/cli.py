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
from pathlib import Path

from . import __version__
from .edl import EDL
from .ffmpeg import FFmpegError, FFmpegMissing, build_config, probe
from .learn import FeedbackStore
from .knowledge import Studio
from .llm import LLMError, describe_providers
from .market import MarketError, MarketStore, compare, fact_sheet
from .pipeline import PACKAGE_ROOT, AutoEditor
from .profile import StyleProfile
from .fonts import CATALOG, install as install_fonts, installed as installed_fonts
from .speech import available_backend

TEMPLATE_DIR = PACKAGE_ROOT / "templates"
PROFILE_DIR = PACKAGE_ROOT / "profiles"
FONTS_DIR = PACKAGE_ROOT / "assets" / "fonts"
SEARCH_DIRS = [TEMPLATE_DIR, PROFILE_DIR]


def _print(message: str) -> None:
    print(f"  {message}", flush=True)


def _profile_from_args(args) -> StyleProfile:
    chosen = getattr(args, "template", None) or getattr(args, "profile", None)
    profile = StyleProfile.resolve(chosen, SEARCH_DIRS)
    overrides: list[str] = list(getattr(args, "set", None) or [])

    flag_map = {
        "no_captions": "captions.enabled=false",
        "no_zoom": "zoom.enabled=false",
        "no_broll": "broll.enabled=false",
        "no_cuts": "cuts.enabled=false",
        "no_transitions": "transitions.enabled=false",
        "no_learning": "learning.enabled=false",
    }
    for flag, override in flag_map.items():
        if getattr(args, flag, False):
            overrides.append(override)

    for attr, key in (("asr_backend", "asr.backend"), ("model", "asr.model"),
                      ("lang", "asr.language"), ("broll_dir", "broll.library"),
                      ("font", "captions.font"), ("caption_style", "captions.style"),
                      ("transitions", "transitions.kind")):
        value = getattr(args, attr, None)
        if value:
            overrides.append(f"{key}={value}")
    if getattr(args, "vertical", None):
        width, _, height = args.vertical.partition("x")
        overrides += [f"output.width={width}", f"output.height={height}"]
    return profile.apply_overrides(overrides)


VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")


def _join_cap(profile: StyleProfile) -> int:
    """Tallest canvas the renderer can actually sample from."""
    height = int(profile.get("output.height"))
    headroom = float(profile.get("output.zoom_headroom"))
    return int(round(height * max(1.0, headroom)))


def _resolve_videos(raw: list[str] | str, *, order: str = "given") -> list[Path]:
    """Expand what the user typed into a list of real video files.

    PowerShell does not expand `*.mp4` for external commands the way a Unix
    shell does, so the pattern arrives here literally and we expand it ourselves.
    """
    if isinstance(raw, str):
        raw = [raw]

    resolved: list[Path] = []
    for entry in raw:
        path = Path(entry).expanduser()
        if any(ch in entry for ch in "*?["):
            base = path.parent if str(path.parent) not in ("", ".") else Path.cwd()
            matches = sorted(base.glob(path.name))
            matches = [m for m in matches if m.suffix.lower() in VIDEO_SUFFIXES]
            if not matches:
                raise FileNotFoundError(f"nothing matched '{entry}' in {base}")
            resolved.extend(matches)
        else:
            resolved.append(_resolve_video(entry))

    if order == "name":
        resolved.sort(key=lambda p: p.name.lower())
    elif order == "time":
        resolved.sort(key=lambda p: p.stat().st_mtime)

    seen: list[Path] = []
    for path in resolved:
        if path not in seen:
            seen.append(path)
    return seen


def _resolve_video(raw: str) -> Path:
    """Check the input file exists, and if not, say what is actually here.

    Getting the filename slightly wrong is the commonest first-run stumble, and
    "file not found" alone leaves you guessing at spelling, extension and folder.
    """
    path = Path(raw).expanduser()
    if path.exists():
        return path

    nearby = sorted(p for p in Path.cwd().iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    message = [f"no file called '{raw}' in {Path.cwd()}"]
    if nearby:
        message.append("")
        message.append("Videos in this folder:")
        message += [f"  {p.name}" for p in nearby[:12]]
        message.append("")
        message.append(f'Use one of those, in quotes if the name has spaces:')
        message.append(f'  reelforge auto "{nearby[0].name}" --model small --review')
    else:
        message.append("")
        message.append("There are no video files in this folder yet.")
        message.append("Open it with `explorer .` and drag a video in, or give a full path:")
        message.append(r'  reelforge auto "C:\Users\you\Videos\clip.mp4" --model small --review')
    raise FileNotFoundError("\n  ".join(message))


def _editor(args, profile: StyleProfile) -> AutoEditor:
    project = Path(args.project) if getattr(args, "project", None) else Path.cwd() / ".reelforge"
    return AutoEditor(profile, project_dir=project,
                      fonts_dir=PACKAGE_ROOT / "assets" / "fonts", on_status=_print)


# ------------------------------------------------------------------ commands

def cmd_auto(args) -> int:
    profile = _profile_from_args(args)
    editor = _editor(args, profile)
    clips = _resolve_videos(args.video, order=args.order)
    started = time.time()

    if len(clips) > 1:
        from .join import join_clips  # noqa: PLC0415
        for index, clip in enumerate(clips, start=1):
            _print(f"  {index}. {clip.name}")
        source = join_clips(clips, editor.work_dir / "joined.mp4",
                            max_height=_join_cap(profile), on_status=_print)
    else:
        source = clips[0]

    extra_vocab = None
    if getattr(args, "script", None):
        from .script import Script, caption_prior, script_vocabulary  # noqa: PLC0415
        script = Script.load(args.script)
        profile = profile.merged({"asr": {"initial_prompt": caption_prior(script)}})
        editor.profile = profile
        extra_vocab = script_vocabulary(script)
        _print(f"using script '{script.title or script.topic}' as the transcription prior "
               f"({len(extra_vocab)} terms)")

    result = editor.plan(source, refresh=args.refresh, extra_vocab=extra_vocab)
    edl = result.edl
    summary = edl.summary()

    for warning in result.warnings:
        _print(f"note: {warning}")
    _print(f"{summary['source_duration']}s -> {summary['output_duration']}s "
           f"({summary['removed']}s of dead air removed)")
    _print(f"{summary['cuts']} segments, {summary['zooms']} zoom moves, "
           f"{summary['overlays']} b-roll layers, {summary['caption_lines']} caption lines")

    naming = clips[0]
    output = Path(args.out) if args.out else naming.with_name(f"{naming.stem}-reel.mp4")
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
    clips = _resolve_videos(args.video, order=getattr(args, "order", "given"))
    if len(clips) > 1:
        from .join import join_clips  # noqa: PLC0415
        source = join_clips(clips, editor.work_dir / "joined.mp4",
                            max_height=_join_cap(profile), on_status=_print)
    else:
        source = clips[0]

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
    for _changed, message in install_fonts(None, FONTS_DIR, force=args.force):
        _print(message)
    _print(f"fonts in {FONTS_DIR}")
    _print("more Arabic fonts: reelforge fonts")
    return cmd_doctor(args)


def cmd_fonts(args) -> int:
    if args.install:
        for _changed, message in install_fonts(args.install, FONTS_DIR, force=args.force):
            _print(message)
        return 0

    have = installed_fonts(FONTS_DIR)
    print("  Arabic caption fonts (all SIL Open Font License)\n")
    for entry in CATALOG:
        mark = "*" if entry.family in have else " "
        print(f"  {mark} {entry.family:<20} {entry.note}")
    print("\n  * = installed. Add one with:  reelforge fonts --install Changa")
    print("  Then use it with:             reelforge auto clip.mp4 --font Changa")
    return 0


def _template_entries() -> list[tuple[str, StyleProfile]]:
    entries = []
    for path in sorted(TEMPLATE_DIR.glob("*.yml")) + sorted(PROFILE_DIR.glob("*.yml")):
        try:
            entries.append((path.stem, StyleProfile.load(path)))
        except Exception:
            continue
    return entries


def cmd_templates(args) -> int:
    entries = _template_entries()
    if args.preview:
        return _render_template_preview(entries, args)

    print("  Ready-made looks. Start with one, then tweak.\n")
    for stem, profile in entries:
        style = profile.get("captions.style")
        font = profile.get("captions.font")
        transition = profile.get("transitions.kind") if profile.get("transitions.enabled") else "none"
        print(f"  {stem:<15} {profile.get('description', '')}")
        print(f"  {'':<15} captions: {style} / {font}   transitions: {transition}\n")
    print("  Use one:      reelforge auto clip.mp4 -t viral")
    print("  See them:     reelforge templates --preview")
    return 0


def _render_template_preview(entries, args) -> int:
    """Render one caption sample per template so you can pick by eye."""
    from .captions import CaptionLine, build_ass  # noqa: PLC0415
    from .ffmpeg import run_ffmpeg  # noqa: PLC0415
    from .speech import Word  # noqa: PLC0415

    out = Path(args.out) if args.out else Path.cwd() / "reelforge-templates.png"
    work = Path(args.project) if args.project else Path.cwd() / ".reelforge"
    work = work / "preview"
    work.mkdir(parents=True, exist_ok=True)

    sample = ["وفرت", "90%", "من", "وقت", "المونتاج"]
    words, cursor = [], 0.0
    for text in sample:
        words.append(Word(text=text, start=cursor, end=cursor + 0.42, prob=1.0))
        cursor += 0.46

    needed = {profile.get("captions.font") for _, profile in entries}
    missing = needed - installed_fonts(FONTS_DIR)
    if missing:
        _print(f"fetching fonts for the preview: {', '.join(sorted(missing))}")
        for _changed, message in install_fonts(sorted(missing), FONTS_DIR):
            _print(message)

    strips = []
    for index, (stem, profile) in enumerate(entries):
        preview_profile = profile.apply_overrides([
            "captions.y_pct=0.5", "captions.safe_area=false", "captions.enabled=true",
        ])
        # Freeze on the second word so the active-word treatment is visible.
        line = CaptionLine(words=[Word(w.text, w.start, w.end, 1.0) for w in words],
                           start=words[0].start, end=words[-1].end)
        ass_path = work / f"{stem}.ass"
        ass_path.write_text(build_ass([line], preview_profile, width=1080, height=1920),
                            encoding="utf-8")
        strip = work / f"{stem}.png"
        label = stem.replace("_", " ")
        run_ffmpeg([
            "-f", "lavfi", "-i", "color=c=0x14141c:s=1080x1920:d=1:r=5",
            "-vf", (f"ass={_escape_for_filter(ass_path)}:fontsdir={_escape_for_filter(FONTS_DIR)},"
                    f"crop=1080:340:0:790,"
                    f"drawtext=text='{label}':x=28:y=18:fontsize=34:fontcolor=0x8a8aa0:"
                    f"fontfile={_escape_for_filter(_any_font())},"
                    f"drawbox=x=0:y=338:w=1080:h=2:color=0x2a2a3a:t=fill"),
            "-ss", f"{words[1].start + 0.2:.2f}", "-frames:v", "1", str(strip),
        ])
        strips.append(strip)

    inputs = []
    for strip in strips:
        inputs += ["-i", str(strip)]
    labels = "".join(f"[{i}:v]" for i in range(len(strips)))
    run_ffmpeg(inputs + ["-filter_complex", f"{labels}vstack=inputs={len(strips)}", str(out)])
    _print(f"preview of {len(strips)} templates -> {out}")
    return 0


def _escape_for_filter(path) -> str:
    text = str(path)
    for char in ("\\", ":", "'", "[", "]", ","):
        text = text.replace(char, "\\" + char)
    return text


def _any_font() -> Path:
    for candidate in sorted(FONTS_DIR.glob("*.ttf")):
        return candidate
    return Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _studio(args) -> Studio:
    root = Path(args.studio) if getattr(args, "studio", None) else Path.cwd() / "studio"
    return Studio(root)


def cmd_studio(args) -> int:
    studio = _studio(args)
    if args.action == "init":
        written = studio.init(force=args.force)
        if written:
            for path in written:
                _print(f"created {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
        else:
            _print("studio already set up (use --force to overwrite)")
        print()
        _print("These live on your computer, not in the repo - they are your content.")
        _print("Edit them before writing scripts; the writer reads them verbatim:")
        _print(f"  {studio.root / 'background.md'}   who you are")
        _print(f"  {studio.root / 'voice.md'}        how you sound")
        _print(f"  {studio.root / 'audience.md'}     who is watching")
        return 0

    # status
    if not studio.exists:
        _print("no studio yet. Run: reelforge studio init")
        return 0
    _print(f"studio: {studio.root}")
    _print(f"frameworks: {', '.join(f.name for f in studio.frameworks())}")
    scripts = sorted(studio.scripts_dir.glob("*.json")) if studio.scripts_dir.exists() else []
    _print(f"scripts written: {len(scripts)}")
    if studio.is_unedited():
        _print("background.md is still the placeholder - edit it before writing scripts")
    print()
    for name, ready, note in describe_providers():
        _print(f"writer {name:<8} {'ready' if ready else 'not set up':<11} {note}")
    return 0


def cmd_script(args) -> int:
    from .script import write_script  # noqa: PLC0415

    studio = _studio(args)
    if args.list_frameworks:
        for framework in studio.frameworks():
            _print(f"{framework.name:<14} ~{framework.seconds}s  {framework.description}")
            if framework.best_for:
                _print(f"{'':<14} best for: {framework.best_for}")
        return 0

    if not studio.exists:
        _print("no studio yet - creating one with the defaults")
        studio.init()
    if not args.topic:
        _print('what is it about? reelforge script "why most people never compound"')
        return 1

    facts = None
    if args.symbol:
        store = _market_store(args)
        facts = fact_sheet(store, args.symbol, start=args.since, end=args.until,
                           amount=args.amount, monthly=args.monthly)
        _print(f"using {len(facts['statements']['en'])} verified facts about {args.symbol}")

    if studio.is_unedited():
        _print("note: studio/background.md is still the placeholder, so the voice "
               "will be generic. Edit it and rerun for something that sounds like you.")

    script = write_script(
        args.topic, studio, framework=args.framework, facts=facts,
        language=args.lang or "ar", seconds=args.seconds,
        provider=args.provider or "auto", model=args.model, extra=args.direction or "",
    )

    if script.provider == "stub":
        _print("no writing model available, so this is the empty structure only.")
        _print("  set ANTHROPIC_API_KEY (pip install anthropic), or run Ollama locally.")

    studio.scripts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{len(list(studio.scripts_dir.glob('*.json'))) + 1:03d}-{_slug(args.topic)}"
    json_path = script.save(studio.scripts_dir / f"{stem}.json")
    markdown_path = studio.scripts_dir / f"{stem}.md"
    markdown_path.write_text(script.to_markdown(), encoding="utf-8")

    print()
    print(script.to_markdown())
    print()
    if script.unverified_numbers:
        _print(f"CHECK THESE - not traceable to your data: "
               f"{', '.join(script.unverified_numbers)}")
    elif facts:
        _print("every number in this script traces back to your data")
    _print(f"saved {json_path.name} and {markdown_path.name} in {studio.scripts_dir}")
    _print(f"after you shoot it:  reelforge auto clip.mp4 --script {json_path}")
    return 0


def _slug(text: str, limit: int = 40) -> str:
    import re as _re  # noqa: PLC0415
    slug = _re.sub(r"[^\w\u0600-\u06FF]+", "-", (text or "").strip().lower())
    return slug.strip("-")[:limit] or "script"


def _market_store(args) -> MarketStore:
    project = Path(args.project) if getattr(args, "project", None) else Path.cwd() / ".reelforge"
    return MarketStore(project)


def cmd_market(args) -> int:
    store = _market_store(args)

    if args.action == "add":
        # Accept both `market add a.csv` and `market add -f a.csv`.
        args.files = (args.files or []) + [s for s in (args.symbols or [])
                                           if Path(s).suffix.lower() in (".csv", ".txt", ".tsv")]
        if not args.files:
            _print("give me at least one CSV: reelforge market add prices.csv --symbol SPX")
            return 1
        for path in args.files:
            symbol = args.symbol or Path(path).stem.upper()
            result = store.add_csv(path, symbol, name=args.name or "",
                                   currency=args.currency or "")
            _print(f"{result['symbol']}: {result['rows']} rows, "
                   f"{result['start']} to {result['end']}")
        return 0

    if args.action == "remove":
        if not args.symbol:
            _print("which symbol? reelforge market remove --symbol SPX")
            return 1
        removed = store.remove(args.symbol)
        _print(f"removed {args.symbol} ({removed} rows)")
        return 0

    if args.action == "list":
        rows = store.symbols()
        if not rows:
            _print("no market data yet. Add some:")
            _print("  reelforge market add prices.csv --symbol SPX --name 'S&P 500'")
            return 0
        for row in rows:
            _print(f"{row['symbol']:<10} {str(row['name'])[:26]:<28} "
                   f"{row['rows']:>6} rows   {row['start']} to {row['end']}")
        return 0

    if args.action == "compare":
        if len(args.symbols or []) < 2:
            _print("give me two or more symbols: reelforge market compare SPX GOLD")
            return 1
        result = compare(store, args.symbols, start=args.since, end=args.until)
        _print(f"{result['start_day']} to {result['end_day']} ({result['years']} years), "
               f"same window for all")
        for row in result["rows"]:
            _print(f"{row['symbol']:<10} cagr {_fmt_pct(row['cagr']):>8}   "
                   f"total {_fmt_pct(row['total_return']):>9}   "
                   f"worst fall {_fmt_pct(row['max_drawdown']):>8}")
        return 0

    # default: facts
    if not args.symbol and not args.symbols:
        _print("which symbol? reelforge market facts SPX")
        return 1
    symbol = args.symbol or args.symbols[0]
    facts = fact_sheet(store, symbol, start=args.since, end=args.until,
                       amount=args.amount, monthly=args.monthly)
    if args.json:
        print(json.dumps(facts, indent=2, ensure_ascii=False))
        return 0

    _print(f"{facts['name']} ({facts['symbol']})  "
           f"{facts['start_day']} to {facts['end_day']}  ·  {facts['bars']} trading days")
    print()
    for line in facts["statements"]["en"]:
        _print(line)
    print()
    _print("ready to say, in Arabic:")
    for line in facts["statements"]["ar"]:
        print(f"      {line}")
    print()
    _print("every number above is computed from your data, not generated")
    return 0


def _fmt_pct(value) -> str:
    return "-" if value is None else f"{value * 100:,.1f}%"


def _checkout_revision() -> str:
    """Which commit is actually installed - the first thing to check when a fix
    'did not work'."""
    import subprocess  # noqa: PLC0415
    try:
        result = subprocess.run(
            ["git", "-C", str(PACKAGE_ROOT), "log", "-1", "--format=%h %cs %s"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return "unknown"
    return (result.stdout or "").strip()[:72] or "unknown (not a git checkout)"


def cmd_doctor(args) -> int:
    ok = True
    _print(f"reelforge {__version__}  ·  {_checkout_revision()}")
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

    # Writers are only used by `reelforge script`. Not having one does not stop
    # any editing, so say so - an unexplained "pip install ..." line reads as a
    # broken install to someone who just wants to cut a video.
    writers = describe_providers()
    if any(ready for _, ready, _ in writers):
        for name, ready, note in writers:
            if ready:
                _print(f"writer {name}: ready - {note}")
    else:
        _print("writers: none set up (optional - only needed for `reelforge script`)")
        for name, _ready, note in writers:
            _print(f"    {name}: {note}")

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
    sub = parser.add_subparsers(dest="command")

    def add_common(target, *, with_video: bool = True):
        if with_video:
            target.add_argument("video", nargs="+",
                                help="video file(s). Several takes are joined into one "
                                     "timeline, in the order given. Wildcards work: *.MP4")
            target.add_argument("--order", choices=["given", "name", "time"],
                                default="given",
                                help="order of multiple clips (default: as typed)")
        target.add_argument("-o", "--out", help="output path")
        target.add_argument("-t", "--template", help="ready-made look: viral, bold, clean, "
                            "word, elegant, news (see `reelforge templates`)")
        target.add_argument("-p", "--profile", help="alias for --template; also accepts a "
                            "path to your own .yml")
        target.add_argument("--caption-style",
                            choices=["karaoke", "box", "pop", "word", "plain"],
                            help="how the spoken word is marked")
        target.add_argument("--transitions",
                            choices=["auto", "punch", "flash", "blur", "none"],
                            help="effect on each cut")
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
        target.add_argument("--no-transitions", action="store_true")
        target.add_argument("--no-learning", action="store_true")

    auto = sub.add_parser("auto", help="analyse and edit a clip end to end")
    add_common(auto)
    auto.add_argument("--review", action="store_true",
                      help="open the local review page before exporting")
    auto.add_argument("--accept", action="store_true",
                      help="record this edit as approved without reviewing")
    auto.add_argument("--plan-only", action="store_true", help="decide but do not render")
    auto.add_argument("--srt", action="store_true", help="also write .srt and .ass files")
    auto.add_argument("--script", help="the script you shot from - used as the "
                      "transcription prior, which sharpens Arabic captions")
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

    templates = sub.add_parser("templates", help="list the ready-made looks")
    templates.add_argument("--preview", action="store_true",
                           help="render a picture of every template's captions")
    templates.add_argument("-o", "--out", help="where to write the preview image")
    templates.add_argument("--project")
    templates.set_defaults(func=cmd_templates)

    fonts = sub.add_parser("fonts", help="list or install Arabic caption fonts")
    fonts.add_argument("--install", action="append", metavar="FAMILY",
                       help="install a family by name, repeatable, or 'all'")
    fonts.add_argument("--force", action="store_true")
    fonts.set_defaults(func=cmd_fonts)

    studio = sub.add_parser("studio", help="the files that make scripts sound like you")
    studio.add_argument("action", nargs="?", default="status", choices=["init", "status"])
    studio.add_argument("--studio", help="studio folder (default ./studio)")
    studio.add_argument("--force", action="store_true", help="overwrite existing files")
    studio.set_defaults(func=cmd_studio)

    script = sub.add_parser("script", help="write a script for a video")
    script.add_argument("topic", nargs="?", help="what the video is about")
    script.add_argument("-f", "--framework", help="which structure to use")
    script.add_argument("--list-frameworks", action="store_true")
    script.add_argument("-s", "--symbol", help="cite verified facts about this symbol")
    script.add_argument("--since")
    script.add_argument("--until")
    script.add_argument("--amount", type=float, default=1000.0)
    script.add_argument("--monthly", type=float)
    script.add_argument("--seconds", type=int, default=45)
    script.add_argument("--lang", default="ar")
    script.add_argument("--provider", choices=["auto", "claude", "ollama", "stub"])
    script.add_argument("--model", help="override the writing model")
    script.add_argument("-d", "--direction", help="extra steer for this script")
    script.add_argument("--studio", help="studio folder (default ./studio)")
    script.add_argument("--project")
    script.set_defaults(func=cmd_script)

    market = sub.add_parser("market", help="your own price history, and facts from it")
    market.add_argument("action", nargs="?", default="facts",
                        choices=["add", "list", "facts", "compare", "remove"])
    market.add_argument("symbols", nargs="*", help="symbol(s) for facts/compare")
    market.add_argument("-f", "--files", action="append", help="CSV to ingest, repeatable")
    market.add_argument("-s", "--symbol", help="symbol to store the data under")
    market.add_argument("--name", help="display name, e.g. 'S&P 500'")
    market.add_argument("--currency", help="e.g. USD, EGP")
    market.add_argument("--since", help="start date, YYYY-MM-DD")
    market.add_argument("--until", help="end date, YYYY-MM-DD")
    market.add_argument("--amount", type=float, default=1000.0,
                        help="lump sum to model, default 1000")
    market.add_argument("--monthly", type=float,
                        help="also model investing this much every month")
    market.add_argument("--json", action="store_true", help="machine-readable output")
    market.add_argument("--project")
    market.set_defaults(func=cmd_market)

    doctor = sub.add_parser("doctor", help="check ffmpeg, fonts and models")
    doctor.add_argument("--project")
    doctor.set_defaults(func=cmd_doctor)

    probe_cmd = sub.add_parser("probe", help="show what ffmpeg sees in a file")
    probe_cmd.add_argument("video")
    probe_cmd.set_defaults(func=cmd_probe)

    return parser


WELCOME = """
  ReelForge - editing happens on this computer. Nothing is uploaded.

  There is no "upload" step: your video file stays where it is and you point
  the command at it.

  First time:
    reelforge doctor                    check ffmpeg, fonts and the speech model
    reelforge setup                     download Arabic caption fonts

  Edit a video (put the file in this folder, or give the full path):
    reelforge auto myvideo.mp4 --review

  Shot it in several takes? Pass them all - they are joined into one Reel:
    reelforge auto take1.mp4 take2.mp4 take3.mp4 --review
    reelforge auto *.MP4 --order name --review

    ...that opens a page in your browser showing the finished cut, every
    zoom and every caption, so you can fix anything before exporting.

  Pick a look first, if you like:
    reelforge templates --preview        writes a picture of each style
    reelforge auto myvideo.mp4 -t viral --review

  Captions only, no editing:
    reelforge captions myvideo.mp4 --srt

  Everything else:
    reelforge --help
"""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # A bare `reelforge` is someone asking what to do, not a usage error.
        print(WELCOME)
        return 0
    try:
        return args.func(args)
    except FFmpegMissing as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except (FFmpegError, MarketError, LLMError, FileNotFoundError, ValueError) as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

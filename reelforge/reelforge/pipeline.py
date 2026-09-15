"""One class that runs the whole thing, used by both the CLI and the review UI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .analysis import Analysis, analyze, content_key
from .arabic import VocabCorrector
from .brain import build_edl
from .broll import BrollLibrary
from .captions import write_captions
from .edl import EDL
from .ffmpeg import extract_wav, probe
from .learn import FeedbackStore
from .profile import StyleProfile
from .render import Renderer, check_font
from .speech import Transcript, transcribe

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PlanResult:
    edl: EDL
    run_id: int
    analysis: Analysis
    transcript: Transcript
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class AutoEditor:
    """Analyse -> decide -> render, with every decision recorded for learning."""

    def __init__(self, profile: StyleProfile, *, project_dir: str | Path | None = None,
                 fonts_dir: str | Path | None = None, on_status=None):
        self.profile = profile
        self.project_dir = Path(project_dir or Path.cwd() / ".reelforge")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.project_dir / "cache"
        self.work_dir = self.project_dir / "work"
        self.runs_dir = self.project_dir / "runs"
        for directory in (self.cache_dir, self.work_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.fonts_dir = Path(fonts_dir) if fonts_dir else PACKAGE_ROOT / "assets" / "fonts"
        self.store = FeedbackStore(self.project_dir)
        self.on_status = on_status or (lambda message: None)

    # -- helpers ---------------------------------------------------------
    def _status(self, message: str) -> None:
        self.on_status(message)

    def effective_profile(self) -> StyleProfile:
        """Base profile plus everything learned from previous runs."""
        if not self.profile.get("learning.enabled"):
            return self.profile
        return self.store.tuned_profile(self.profile)

    def library(self, profile: StyleProfile) -> BrollLibrary:
        raw = profile.get("broll.library")
        if not raw:
            return BrollLibrary()
        path = Path(raw)
        if not path.is_absolute():
            for base in (Path.cwd(), self.project_dir.parent, PACKAGE_ROOT):
                candidate = base / path
                if candidate.exists():
                    path = candidate
                    break
        return BrollLibrary.load(path)

    # -- plan ------------------------------------------------------------
    def plan(self, source: str | Path, *, refresh: bool = False,
             extra_vocab: dict[str, str] | None = None,
             record: bool = True) -> PlanResult:
        source = Path(source).expanduser().resolve()
        profile = self.effective_profile()
        warnings: list[str] = []
        timings: dict[str, float] = {}

        info = probe(source)
        if not info.has_audio:
            warnings.append("source has no audio track - captions and cuts are skipped")

        started = time.time()
        self._status("analysing audio and shots")
        analysis = analyze(source, profile, cache_dir=self.cache_dir, info=info, refresh=refresh)
        timings["analysis"] = time.time() - started

        # A script for this shoot is the strongest possible prior: we already know
        # roughly which words were said, so they stop being guesses.
        pairs = dict(self.store.vocab_pairs()) if profile.get("learning.enabled") else {}
        if extra_vocab:
            pairs.update(extra_vocab)
        corrector = VocabCorrector(pairs)

        started = time.time()
        transcript = Transcript(language=profile.get("asr.language"), backend="none", model="")
        if info.has_audio and (profile.get("captions.enabled") or profile.get("cuts.enabled")):
            wav = self.work_dir / f"{source.stem}-16k.wav"
            if not wav.exists() or refresh:
                extract_wav(source, wav)
            transcript = transcribe(
                wav, profile, analysis=analysis, corrector=corrector,
                cache_dir=self.cache_dir, cache_key=content_key(source),
                refresh=refresh, on_status=self._status,
            )
            if transcript.backend == "stub":
                warnings.append(
                    "no speech model installed - captions are placeholder text. "
                    "Install one with: pip install -r requirements-asr.txt"
                )
        timings["transcribe"] = time.time() - started

        started = time.time()
        self._status("planning the edit")
        scorer = self.store.scorer(min_samples=int(profile.get("learning.min_samples"))) \
            if profile.get("learning.enabled") else None
        edl = build_edl(
            str(source), analysis, transcript, profile,
            library=self.library(profile), scorer=scorer,
            weights=self.store.keyword_weights() if profile.get("learning.enabled") else None,
        )
        timings["plan"] = time.time() - started

        font_warning = check_font(profile, self.fonts_dir)
        if font_warning and profile.get("captions.enabled"):
            warnings.append(font_warning)

        # A plan made to answer "what would this look like" is not a run: the
        # browser asks for one every time a setting moves, and recording them
        # would bury the edits you actually kept - which is what the editor
        # learns from - under thousands of previews nobody ever saw.
        run_id = 0
        if record:
            run_id = self.store.record_run(str(source), profile, edl,
                                           video_key=content_key(source))
            edl.save(self.runs_dir / f"run-{run_id}.edl.json")
        return PlanResult(edl=edl, run_id=run_id, analysis=analysis,
                          transcript=transcript, warnings=warnings, timings=timings)

    # -- render ----------------------------------------------------------
    def render(self, edl: EDL, output: str | Path, *, preview: bool = False,
               burn_captions: bool = True) -> Path:
        profile = self.effective_profile()
        renderer = Renderer(profile, work_dir=self.work_dir, fonts_dir=self.fonts_dir)
        return renderer.render(edl, output, preview=preview, burn_captions=burn_captions,
                               on_status=self._status)

    def export_captions(self, edl: EDL, out_dir: str | Path, stem: str) -> dict[str, Path]:
        profile = self.effective_profile()
        return write_captions(edl.captions, profile, Path(out_dir), stem,
                              width=edl.output.get("width"), height=edl.output.get("height"))

    # -- feedback --------------------------------------------------------
    def accept(self, run_id: int, final: EDL) -> dict:
        """Record the edit you actually kept, then retrain."""
        profile = self.effective_profile()
        learned = self.store.record_feedback(run_id, final,
                                             ema=float(profile.get("learning.ema")))
        model = self.store.train_zoom_model(
            lr=float(profile.get("learning.learning_rate")),
            min_samples=int(profile.get("learning.min_samples")),
        )
        learned["model"] = ({"samples": model.samples, "accuracy": round(model.accuracy, 3)}
                            if model else None)
        return learned

"""Test suite. Uses only the standard library plus ffmpeg.

The media tests build their own tiny clip with ffmpeg's synthetic sources, so the
suite runs anywhere ffmpeg is installed and needs no sample files or models.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reelforge import arabic, captions as captions_mod
from reelforge.analysis import Interval, analyze, invert_intervals
from reelforge.brain import build_edl, rule_score
from reelforge.broll import BrollLibrary
from reelforge.captions import CaptionLine, apply_text_edit, build_ass, build_srt, group_words
from reelforge.edl import EDL, Cut, Timeline, Zoom, build_timeline
from reelforge.learn import FeedbackStore, ZoomModel
from reelforge.pipeline import AutoEditor
from reelforge.profile import StyleProfile
from reelforge.render import zoom_expression
from reelforge.speech import Word, snap_words_to_energy, transcribe

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is required for this test")


def make_clip(path: Path, *, duration: int = 8) -> Path:
    """A clip with two shots and speech-like bursts separated by silence."""
    half = duration // 2
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={half}",
        "-f", "lavfi", "-i", f"smptebars=size=640x360:rate=25:duration={duration - half}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-filter_complex",
        f"[0:v][1:v]concat=n=2:v=1:a=0[v];"
        f"[2:a]volume='if(between(t,0.4,{half - 0.5})+between(t,{half + 0.5},{duration - 0.4}),0.8,0.0)'"
        f":eval=frame[a]",
        "-map", "[v]", "-map", "[a]", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True, capture_output=True)
    return path


class ArabicTests(unittest.TestCase):
    def test_normalisation_folds_variants(self):
        self.assertEqual(arabic.normalize_for_match("الذَّكاء"), arabic.normalize_for_match("الذكاء"))
        self.assertEqual(arabic.normalize_for_match("أسامة"), arabic.normalize_for_match("اسامه"))
        self.assertEqual(arabic.normalize_for_match("قناتــي"), "قناتي")

    def test_display_cleanup_keeps_letters_but_fixes_punctuation(self):
        self.assertEqual(arabic.clean_for_display("مرحبا, كيف حالك?"), "مرحبا، كيف حالك؟")
        self.assertNotIn("\n", arabic.clean_for_display("سطر\nثان"))

    def test_display_does_not_reverse_text(self):
        # Pre-reversing is the classic Arabic subtitle bug; libass handles bidi itself.
        text = "مرحبا بكم"
        self.assertEqual(arabic.clean_for_display(text), text)

    def test_similarity_and_corrector(self):
        self.assertGreater(arabic.similarity("أسامة", "اسامه"), 0.9)
        self.assertLess(arabic.similarity("كتاب", "سيارة"), 0.4)
        corrector = arabic.VocabCorrector({"اسامه": "أسامة"})
        self.assertEqual(corrector.correct_text("انا اسامه"), "انا أسامة")

    def test_arabic_ratio(self):
        self.assertEqual(arabic.arabic_ratio("hello"), 0.0)
        self.assertEqual(arabic.arabic_ratio("مرحبا"), 1.0)


class ProfileTests(unittest.TestCase):
    def test_overrides_are_typed_and_non_destructive(self):
        base = StyleProfile()
        tuned = base.apply_overrides(["zoom.max_factor=1.4", "captions.karaoke=false",
                                      "captions.font=Tajawal"])
        self.assertEqual(tuned.get("zoom.max_factor"), 1.4)
        self.assertIs(tuned.get("captions.karaoke"), False)
        self.assertEqual(tuned.get("captions.font"), "Tajawal")
        self.assertEqual(base.get("zoom.max_factor"), 1.22)  # untouched

    def test_unknown_key_raises_unless_defaulted(self):
        profile = StyleProfile()
        with self.assertRaises(KeyError):
            profile.get("zoom.nonexistent")
        self.assertEqual(profile.get("zoom.nonexistent", 3), 3)

    def test_named_profiles_load(self):
        directory = Path(__file__).resolve().parent.parent / "profiles"
        for name in ("punchy", "calm", "captions_only"):
            profile = StyleProfile.resolve(name, [directory])
            self.assertEqual(profile.get("name"), name)
        self.assertFalse(StyleProfile.resolve("captions_only", [directory]).get("cuts.enabled"))


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline(build_timeline([(0.5, 3.5), (5.0, 7.5), (9.0, 11.5)]))

    def test_output_duration_excludes_cut_material(self):
        self.assertAlmostEqual(self.timeline.duration, 8.0)

    def test_source_maps_into_output_time(self):
        self.assertAlmostEqual(self.timeline.to_out(0.5), 0.0)
        self.assertAlmostEqual(self.timeline.to_out(6.0), 4.0)
        self.assertAlmostEqual(self.timeline.to_out(11.5), 8.0)

    def test_removed_time_clamps_forward_but_is_strictly_none(self):
        self.assertIsNone(self.timeline.to_out(4.2, clamp=False))
        self.assertAlmostEqual(self.timeline.to_out(4.2), 3.0)

    def test_round_trip(self):
        self.assertAlmostEqual(self.timeline.to_src(4.0), 6.0)

    def test_cut_boundaries_are_in_output_time(self):
        self.assertEqual(self.timeline.cut_boundaries(), [3.0, 5.5])

    def test_interval_inversion(self):
        speech = invert_intervals([Interval(0, 1), Interval(4, 5)], 8.0)
        self.assertEqual([(i.start, i.end) for i in speech], [(1.0, 4.0), (5.0, 8.0)])


class CaptionTests(unittest.TestCase):
    def words(self, texts, step=0.4):
        out, t = [], 0.0
        for text in texts:
            out.append(Word(text, t, t + step * 0.9, 0.9))
            t += step
        return out

    def test_grouping_respects_word_limit(self):
        profile = StyleProfile().apply_overrides(["captions.max_words=3"])
        lines = group_words(self.words(["واحد", "اثنين", "ثلاثة", "أربعة", "خمسة"]), profile)
        self.assertTrue(all(len(line.words) <= 3 for line in lines))

    def test_grouping_splits_on_a_pause(self):
        words = self.words(["مرحبا", "بكم"])
        words.append(Word("اليوم", 2.5, 2.9, 0.9))  # long gap
        lines = group_words(words, StyleProfile())
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1].text, "اليوم")

    def test_ass_is_rtl_safe_and_karaoke_tiles_without_gaps(self):
        profile = StyleProfile()
        lines = group_words(self.words(["مرحبا", "بكم", "في", "قناتي"]), profile)
        ass = build_ass(lines, profile)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("Style: Reel,Cairo", ass)

        events = [line for line in ass.splitlines() if line.startswith("Dialogue")]
        self.assertEqual(len(events), sum(len(line.words) for line in lines))
        # Text must survive in logical order, not reversed.
        self.assertIn("مرحبا", events[0])
        # Consecutive events for one line must touch, or the caption blinks.
        times = [(event.split(",")[1], event.split(",")[2]) for event in events]
        for (_, end), (start, _) in zip(times, times[1:]):
            self.assertEqual(end, start)

    def test_ass_escapes_braces(self):
        line = CaptionLine(words=[Word("{hack}", 0, 1)], start=0, end=1)
        self.assertIn("\\{hack\\}", build_ass([line], StyleProfile()))

    def test_srt_output(self):
        lines = group_words(self.words(["مرحبا", "بكم"]), StyleProfile())
        srt = build_srt(lines)
        self.assertIn("00:00:00,000 --> ", srt)

    def test_text_edit_keeps_timing_when_word_count_matches(self):
        line = CaptionLine(words=[Word("اسامه", 0, 0.5), Word("معكم", 0.5, 1.0)], start=0, end=1.0)
        apply_text_edit(line, "أسامة معكم")
        self.assertEqual([w.text for w in line.words], ["أسامة", "معكم"])
        self.assertEqual(line.words[0].end, 0.5)

    def test_text_edit_redistributes_when_word_count_changes(self):
        line = CaptionLine(words=[Word("اسامه", 0, 1.0)], start=0, end=1.0)
        apply_text_edit(line, "أسامة مغاوري معكم")
        self.assertEqual(len(line.words), 3)
        self.assertLessEqual(line.words[-1].end, line.end + 1e-6)


class ZoomExpressionTests(unittest.TestCase):
    def evaluate(self, zooms, t):
        """Mirror of the ffmpeg expression, for verifying continuity."""
        factor = 1.0
        for zoom in sorted(zooms, key=lambda z: z.out_start):
            if t < zoom.out_start:
                return zoom.start_factor
            if t < zoom.out_end:
                p = (t - zoom.out_start) / (zoom.out_end - zoom.out_start)
                shaped = p * p * (3 - 2 * p)
                return zoom.start_factor + (zoom.end_factor - zoom.start_factor) * shaped
            factor = zoom.end_factor
        return factor

    def test_expression_is_emitted_for_each_move(self):
        zooms = [Zoom("z1", 1, 2, 1.0, 1.2), Zoom("z2", 4, 5, 1.2, 1.0)]
        expression = zoom_expression(zooms)
        for boundary in ("1.000", "2.000", "4.000", "5.000"):
            self.assertIn(f"lt(in_time,{boundary})", expression)
        self.assertNotIn("HOLD", expression)   # placeholder must be fully substituted

    def test_factor_holds_between_moves(self):
        zooms = [Zoom("z1", 1, 2, 1.0, 1.2), Zoom("z2", 4, 5, 1.2, 1.0)]
        self.assertAlmostEqual(self.evaluate(zooms, 3.0), 1.2)
        self.assertAlmostEqual(self.evaluate(zooms, 6.0), 1.0)

    def test_curve_is_continuous_at_every_boundary(self):
        zooms = [Zoom("z1", 1, 2, 1.0, 1.2), Zoom("z2", 4, 5, 1.2, 1.0)]
        for boundary in (1.0, 2.0, 4.0, 5.0):
            before = self.evaluate(zooms, boundary - 1e-4)
            after = self.evaluate(zooms, boundary + 1e-4)
            self.assertLess(abs(before - after), 1e-3)

    def test_no_moves_is_identity(self):
        self.assertEqual(zoom_expression([]), "1.0")

    def test_disabled_moves_are_ignored(self):
        self.assertEqual(zoom_expression([Zoom("z1", 1, 2, 1.0, 1.2, enabled=False)]), "1.0")


class BrollTests(unittest.TestCase):
    def test_arabic_filenames_become_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "الذكاء_الاصطناعي.mp4").write_bytes(b"x")
            (root / "laptop-work.mp4").write_bytes(b"x")
            library = BrollLibrary.load(root)
            self.assertEqual(len(library), 2)
            match = library.match("سنتحدث عن الذكاء الاصطناعي")
            self.assertIsNotNone(match)
            self.assertEqual(match[0].path.name, "الذكاء_الاصطناعي.mp4")

    def test_stopwords_do_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "في.mp4").write_bytes(b"x")
            self.assertIsNone(BrollLibrary.load(root).match("نحن في البيت"))

    def test_learned_weight_can_suppress_an_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "فلوس.mp4").write_bytes(b"x")
            library = BrollLibrary.load(root)
            self.assertIsNotNone(library.match("عندي فلوس كثيرة"))
            self.assertIsNone(library.match("عندي فلوس كثيرة",
                                            weights={"فلوس.mp4|فلوس": 0.2}))


class LearningTests(unittest.TestCase):
    def test_model_learns_a_preference_and_reports_drivers(self):
        import random
        random.seed(11)
        rows = []
        for _ in range(150):
            features = {
                "rel_energy": random.uniform(-6, 6), "rel_peak": random.uniform(-2, 10),
                "duration": random.uniform(0.5, 2.5), "word_rate": random.uniform(0.8, 4),
                "after_cut": float(random.choice([0, 1])), "position": random.random(),
                "since_last_zoom": random.uniform(0, 8),
            }
            rows.append((features, 1 if features["rel_energy"] > 2.0 else 0))
        model = ZoomModel().fit(rows)
        self.assertGreater(model.accuracy, 0.9)
        self.assertEqual(model.explain()[0][0], "rel_energy")
        self.assertGreater(model.predict({**rows[0][0], "rel_energy": 6.0}), 0.5)

    def test_model_survives_a_round_trip(self):
        model = ZoomModel().fit([({"rel_energy": 1.0}, 1), ({"rel_energy": -1.0}, 0)])
        restored = ZoomModel.from_json(model.to_json())
        self.assertAlmostEqual(restored.predict({"rel_energy": 1.0}),
                               model.predict({"rel_energy": 1.0}))

    def test_feedback_records_decisions_and_tunes_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            profile = StyleProfile()
            edl = EDL(source="x.mp4", output={}, cuts=build_timeline([(0, 60)]),
                      zooms=[Zoom(f"z{i}", i, i + 1, 1.0, 1.1,
                                  features={"rel_energy": float(i)}) for i in range(6)])
            run_id = store.record_run("x.mp4", profile, edl)

            # Keep only the first two moves.
            for zoom in edl.zooms[2:]:
                zoom.enabled = False
            learned = store.record_feedback(run_id, edl)

            self.assertEqual(learned["zooms_kept"], 2)
            self.assertEqual(learned["zooms_dropped"], 4)
            # Dropping most of them should make the system more conservative.
            self.assertGreater(store.get_param("zoom.score_threshold", 0.45), 0.45)
            self.assertEqual(len(store.training_rows("zoom")), 6)

    def test_caption_edits_become_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            proposed = EDL(source="x.mp4", output={}, cuts=build_timeline([(0, 10)]),
                           captions=[CaptionLine(words=[Word("اسامه", 0, 0.5),
                                                        Word("معكم", 0.5, 1.0)],
                                                 start=0, end=1.0)])
            run_id = store.record_run("x.mp4", StyleProfile(), proposed)

            final = EDL.from_dict(json.loads(json.dumps(proposed.to_dict())))
            final.captions[0].words[0].text = "أسامة"
            learned = store.record_feedback(run_id, final)

            self.assertEqual(learned["vocab_added"], 1)
            pairs = store.vocab_pairs()
            self.assertEqual(pairs[arabic.normalize_for_match("اسامه")], "أسامة")
            # And it biases the next transcription.
            self.assertIn("أسامة", arabic.VocabCorrector(pairs).prompt_terms())

    def test_tuned_profile_applies_learned_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.set_param("zoom.rate_per_min", 5.0)
            tuned = store.tuned_profile(StyleProfile())
            self.assertEqual(tuned.get("zoom.rate_per_min"), 5.0)
            self.assertEqual(tuned.get("zoom.max_factor"), 1.22)  # others untouched

    def test_parameters_stay_inside_sane_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            self.assertEqual(store.set_param("zoom.rate_per_min", 999.0), 30.0)
            self.assertEqual(store.set_param("zoom.score_threshold", -5.0), 0.15)


class EDLTests(unittest.TestCase):
    def test_round_trip_preserves_everything(self):
        edl = EDL(source="a.mp4", output={"width": 1080, "height": 1920},
                  cuts=build_timeline([(0, 2), (3, 5)]),
                  zooms=[Zoom("z1", 0, 1, 1.0, 1.2)],
                  captions=[CaptionLine(words=[Word("مرحبا", 0, 1)], start=0, end=1)])
        restored = EDL.from_dict(json.loads(json.dumps(edl.to_dict())))
        self.assertEqual(restored.summary(), edl.summary())
        self.assertEqual(restored.captions[0].text, "مرحبا")

    def test_future_version_is_rejected(self):
        with self.assertRaises(ValueError):
            EDL.from_dict({"version": 99, "source": "a.mp4", "output": {}})


@needs_ffmpeg
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reelforge-test-")
        cls.clip = make_clip(Path(cls.tmp) / "clip.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def profile(self, *overrides):
        return StyleProfile().apply_overrides(["asr.backend=stub", *overrides])

    def test_analysis_finds_speech_and_shot_change(self):
        analysis = analyze(self.clip, self.profile())
        self.assertGreaterEqual(len(analysis.speech), 2)
        self.assertTrue(any(abs(s - 4.0) < 0.3 for s in analysis.scenes))
        self.assertTrue(analysis.energy)

    def test_cuts_shorten_the_clip_but_keep_the_speech(self):
        profile = self.profile()
        analysis = analyze(self.clip, profile)
        transcript = transcribe(self.clip, profile, analysis=analysis)
        edl = build_edl(str(self.clip), analysis, transcript, profile)
        self.assertLess(edl.duration, analysis.duration)
        self.assertGreater(edl.duration, analysis.duration * 0.5)

    def test_word_snapping_keeps_words_ordered(self):
        profile = self.profile()
        analysis = analyze(self.clip, profile)
        words = [Word("أ", 1.0, 1.4), Word("ب", 1.45, 1.9)]
        snapped = snap_words_to_energy(words, analysis)
        self.assertEqual(len(snapped), 2)
        self.assertLessEqual(snapped[0].end, snapped[1].start)
        self.assertTrue(all(w.end > w.start for w in snapped))

    def test_disabling_features_produces_an_untouched_timeline(self):
        profile = self.profile("cuts.enabled=false", "zoom.enabled=false",
                               "captions.enabled=false", "broll.enabled=false")
        analysis = analyze(self.clip, profile)
        transcript = transcribe(self.clip, profile, analysis=analysis)
        edl = build_edl(str(self.clip), analysis, transcript, profile)
        self.assertEqual(len(edl.cuts), 1)
        self.assertEqual(edl.zooms, [])
        self.assertAlmostEqual(edl.duration, analysis.duration, places=1)

    def test_rule_score_is_bounded(self):
        profile = self.profile()
        for energy in (-20.0, 0.0, 20.0):
            features = {"rel_energy": energy, "rel_peak": energy, "duration": 1.0,
                        "word_rate": 2.0, "after_cut": 1.0, "position": 0.0,
                        "since_last_zoom": 5.0}
            self.assertGreaterEqual(rule_score(features, profile), 0.0)
            self.assertLessEqual(rule_score(features, profile), 1.0)

    def test_end_to_end_render_is_vertical_and_shorter(self):
        project = Path(self.tmp) / "project"
        editor = AutoEditor(self.profile(), project_dir=project,
                            fonts_dir=Path(__file__).resolve().parent.parent / "assets" / "fonts")
        result = editor.plan(self.clip)
        output = Path(self.tmp) / "out.mp4"
        editor.render(result.edl, output, preview=True)

        self.assertTrue(output.exists())
        from reelforge.ffmpeg import probe
        info = probe(output)
        self.assertGreater(info.height, info.width)          # vertical
        self.assertLess(info.duration, 8.0)                  # dead air removed
        self.assertTrue(info.has_audio)

    def test_render_survives_awkward_paths(self):
        """Filtergraph arguments must be escaped - paths like this used to break it."""
        awkward = Path(self.tmp) / "my clip, take [2].mp4"
        shutil.copy(self.clip, awkward)
        project = Path(self.tmp) / "project-awkward"
        editor = AutoEditor(self.profile(), project_dir=project,
                            fonts_dir=Path(__file__).resolve().parent.parent / "assets" / "fonts")
        result = editor.plan(awkward)
        output = Path(self.tmp) / "out-awkward.mp4"
        editor.render(result.edl, output, preview=True)
        self.assertTrue(output.exists() and output.stat().st_size > 0)

    def test_accepting_a_run_records_training_data(self):
        project = Path(self.tmp) / "project-learn"
        editor = AutoEditor(self.profile(), project_dir=project)
        result = editor.plan(self.clip)
        learned = editor.accept(result.run_id, result.edl)
        self.assertGreaterEqual(learned["zooms_kept"] + learned["zooms_dropped"], 0)
        self.assertEqual(editor.store.stats()["reviewed"], 1)

    def test_analysis_is_cached(self):
        project = Path(self.tmp) / "project-cache"
        editor = AutoEditor(self.profile(), project_dir=project)
        editor.plan(self.clip)
        cached = list((project / "cache").glob("analysis-*.json"))
        self.assertTrue(cached)


if __name__ == "__main__":
    unittest.main(verbosity=2)


try:
    import faster_whisper  # noqa: F401
    HAVE_FASTER_WHISPER = True
except ImportError:
    HAVE_FASTER_WHISPER = False


@unittest.skipUnless(HAVE_FASTER_WHISPER, "faster-whisper is not installed")
class FasterWhisperAdapterTests(unittest.TestCase):
    """Drive the real backend with the library's own result types.

    A model download is not needed to check the part that actually breaks: whether
    we read the right fields off faster-whisper's Segment/Word objects and pass it
    kwargs it accepts. Using the genuine dataclasses means this test fails loudly
    if the library changes shape under us.
    """

    def real_segment(self, start, end, text, words):
        from faster_whisper.transcribe import Segment, Word
        return Segment(
            id=1, seek=0, start=start, end=end, text=text,
            tokens=[], avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01,
            words=[Word(start=s, end=e, word=w, probability=p) for w, s, e, p in words],
            temperature=0.0,
        )

    def patched_model(self, segments, recorder):
        """A stand-in WhisperModel that records how it was called."""
        class FakeModel:
            def __init__(self, name, **kwargs):
                recorder["init"] = {"name": name, **kwargs}

            def transcribe(self, audio, **kwargs):
                recorder["transcribe"] = {"audio": audio, **kwargs}
                return iter(segments), object()

        return FakeModel

    def run_backend(self, segments, profile=None, prompt=None):
        import faster_whisper
        from reelforge.speech import transcribe_faster_whisper
        recorder: dict = {}
        original = faster_whisper.WhisperModel
        faster_whisper.WhisperModel = self.patched_model(segments, recorder)
        try:
            transcript = transcribe_faster_whisper(
                Path("audio.wav"),
                profile or StyleProfile().apply_overrides(["asr.device=cpu"]),
                prompt=prompt,
            )
        finally:
            faster_whisper.WhisperModel = original
        return transcript, recorder

    def test_words_and_probabilities_are_read_correctly(self):
        segments = [self.real_segment(0.0, 1.2, " مرحبا بكم ",
                                      [("مرحبا", 0.0, 0.5, 0.94), ("بكم", 0.6, 1.2, 0.81)])]
        transcript, _ = self.run_backend(segments)

        self.assertEqual(transcript.backend, "faster-whisper")
        self.assertEqual([w.text for w in transcript.words], ["مرحبا", "بكم"])
        self.assertAlmostEqual(transcript.words[0].prob, 0.94)
        self.assertAlmostEqual(transcript.words[1].start, 0.6)
        self.assertEqual(transcript.segments[0].text, "مرحبا بكم")   # whitespace cleaned

    def test_words_with_missing_timestamps_are_dropped_not_crashed_on(self):
        # Real Whisper output occasionally carries a word with no timing.
        from faster_whisper.transcribe import Segment, Word
        segment = Segment(
            id=1, seek=0, start=0.0, end=1.0, text="مرحبا بكم", tokens=[],
            avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01,
            words=[Word(start=0.0, end=0.5, word="مرحبا", probability=0.9),
                   Word(start=None, end=None, word="بكم", probability=0.5),
                   Word(start=0.6, end=0.9, word="   ", probability=0.4)],
            temperature=0.0,
        )
        transcript, _ = self.run_backend([segment])
        self.assertEqual([w.text for w in transcript.words], ["مرحبا"])

    def test_profile_settings_reach_the_library(self):
        profile = StyleProfile().apply_overrides([
            "asr.device=cpu", "asr.compute_type=int8", "asr.model=large-v3",
            "asr.language=ar", "asr.beam_size=7", "asr.vad_min_silence_ms=250",
        ])
        segments = [self.real_segment(0.0, 0.4, "نعم", [("نعم", 0.0, 0.4, 0.9)])]
        _, recorder = self.run_backend(segments, profile=profile, prompt="أسامة")

        self.assertEqual(recorder["init"]["name"], "large-v3")
        self.assertEqual(recorder["init"]["compute_type"], "int8")
        call = recorder["transcribe"]
        self.assertEqual(call["language"], "ar")
        self.assertEqual(call["beam_size"], 7)
        self.assertTrue(call["word_timestamps"])
        self.assertEqual(call["initial_prompt"], "أسامة")
        self.assertEqual(call["vad_parameters"], {"min_silence_duration_ms": 250})
        # Guards against Whisper's repetition loops.
        self.assertFalse(call["condition_on_previous_text"])

    def test_every_kwarg_we_send_is_accepted_by_the_installed_library(self):
        """The failure mode that would break a real run: an unsupported kwarg."""
        import inspect
        from faster_whisper import WhisperModel
        segments = [self.real_segment(0.0, 0.4, "نعم", [("نعم", 0.0, 0.4, 0.9)])]
        _, recorder = self.run_backend(segments)

        accepted = set(inspect.signature(WhisperModel.transcribe).parameters)
        for kwarg in recorder["transcribe"]:
            if kwarg == "audio":
                continue
            self.assertIn(kwarg, accepted, f"faster-whisper does not accept '{kwarg}'")

        accepted_init = set(inspect.signature(WhisperModel.__init__).parameters)
        for kwarg in recorder["init"]:
            if kwarg == "name":
                continue
            self.assertIn(kwarg, accepted_init)

    def test_learned_vocabulary_is_applied_to_the_result(self):
        from reelforge.arabic import VocabCorrector
        from reelforge.speech import transcribe
        import faster_whisper

        segments = [self.real_segment(0.0, 1.0, "انا اسامه",
                                      [("انا", 0.0, 0.4, 0.9), ("اسامه", 0.4, 1.0, 0.6)])]
        recorder: dict = {}
        original = faster_whisper.WhisperModel
        faster_whisper.WhisperModel = self.patched_model(segments, recorder)
        try:
            transcript = transcribe(
                Path("audio.wav"),
                StyleProfile().apply_overrides(["asr.backend=faster-whisper", "asr.device=cpu"]),
                corrector=VocabCorrector({"اسامه": "أسامة"}),
            )
        finally:
            faster_whisper.WhisperModel = original

        self.assertEqual([w.text for w in transcript.words], ["انا", "أسامة"])
        # ...and the same vocabulary biased the decoder up front.
        self.assertIn("أسامة", recorder["transcribe"]["initial_prompt"])


class CaptionStyleTests(unittest.TestCase):
    def words(self, texts, step=0.4):
        out, t = [], 0.0
        for text in texts:
            out.append(Word(text, t, t + step * 0.9, 0.9))
            t += step
        return out

    def build(self, style, *overrides):
        profile = StyleProfile().apply_overrides([f"captions.style={style}", *overrides])
        lines = group_words(self.words(["وفرت", "تسعين", "من", "وقتك"]), profile)
        return profile, lines, build_ass(lines, profile)

    def events(self, ass):
        return [line for line in ass.splitlines() if line.startswith("Dialogue")]

    def body(self, event: str) -> str:
        """The Text field of an ASS Dialogue line (the first 9 fields are metadata)."""
        return event.split(",", 9)[9]

    def test_every_style_produces_a_usable_script(self):
        for style in ("karaoke", "box", "pop", "word", "plain"):
            _, lines, ass = self.build(style)
            with self.subTest(style=style):
                self.assertIn("[Events]", ass)
                self.assertTrue(self.events(ass), f"{style} produced no events")

    def test_box_style_uses_an_opaque_box_only_on_the_active_word(self):
        _, _, ass = self.build("box")
        style_line = [l for l in ass.splitlines() if l.startswith("Style:")][0]
        self.assertEqual(style_line.split(",")[15], "3")        # BorderStyle 3 = filled box
        self.assertEqual(style_line.split(",")[5], "&HFF000000")  # transparent by default
        self.assertIn("\\3a&H00&", self.events(ass)[0])         # active word opts in

    def test_pop_style_scales_the_line_not_a_single_word(self):
        # Scaling one word would grow it into its neighbour, since libass does not reflow.
        _, _, ass = self.build("pop")
        text = self.body(self.events(ass)[0])
        self.assertTrue(text.startswith("{\\fscx"), f"line-level pulse missing: {text[:40]}")
        self.assertEqual(text.count("\\fscx"), 2)   # only the opening tag and its transform

    def test_word_style_shows_one_word_per_event(self):
        _, lines, ass = self.build("word")
        events = self.events(ass)
        self.assertEqual(len(events), sum(len(line.words) for line in lines))
        for event, word in zip(events, [w for line in lines for w in line.words]):
            body = self.body(event)
            self.assertIn(word.text, body)
            for other in ("وفرت", "تسعين", "من", "وقتك"):
                if other != word.text:
                    self.assertNotIn(other, body)

    def test_plain_style_emits_one_event_per_line(self):
        _, lines, ass = self.build("plain")
        self.assertEqual(len(self.events(ass)), len(lines))

    def test_emphasis_marks_numbers_without_resizing_them_by_default(self):
        profile = StyleProfile()
        self.assertEqual(profile.get("captions.emphasis_scale"), 1.0)
        words = [Word("وفرت", 0, 0.4), Word("90%", 0.4, 0.8), Word("وقتك", 0.8, 1.2)]
        ass = build_ass(group_words(words, profile), profile)
        body = self.body(self.events(ass)[0])
        self.assertIn("90%", body)
        self.assertNotIn("\\fscx", body)   # colour only - no width change, no overlap

    def test_emphasis_can_be_disabled(self):
        profile = StyleProfile().apply_overrides(["captions.emphasis=false"])
        words = [Word("وفرت", 0, 0.4), Word("90%", 0.4, 0.8)]
        emphasis_colour = _inline = profile.get("captions.emphasis_color").lstrip("#")
        ass = build_ass(group_words(words, profile), profile)
        self.assertNotIn(emphasis_colour[4:6] + emphasis_colour[2:4] + emphasis_colour[0:2],
                         ass.upper())

    def test_custom_emphasis_words_are_honoured(self):
        self.assertFalse(arabic.is_emphatic("قناتي"))
        self.assertTrue(arabic.is_emphatic("قناتي", arabic.emphasis_set(["قناتي"])))


class TransitionTests(unittest.TestCase):
    def timeline(self):
        return Timeline(build_timeline([(0, 3), (5, 8), (10, 13)]))

    def analysis(self, scenes=()):
        from reelforge.analysis import Analysis
        return Analysis(duration=14.0, width=1280, height=720, fps=30.0,
                        has_audio=True, scenes=list(scenes))

    def test_one_transition_per_cut_boundary(self):
        from reelforge.brain import plan_transitions
        transitions = plan_transitions(self.timeline(), self.analysis(), StyleProfile())
        self.assertEqual([round(t.out_time, 2) for t in transitions], [3.0, 6.0])

    def test_disabled_or_none_yields_nothing(self):
        from reelforge.brain import plan_transitions
        for override in ("transitions.enabled=false", "transitions.kind=none"):
            profile = StyleProfile().apply_overrides([override])
            self.assertEqual(plan_transitions(self.timeline(), self.analysis(), profile), [])

    def test_auto_picks_blur_where_the_shot_actually_changes(self):
        from reelforge.brain import plan_transitions
        # Output 3.0 maps back to source 3.0; put a scene change right there.
        transitions = plan_transitions(self.timeline(), self.analysis(scenes=[3.0]),
                                       StyleProfile())
        self.assertEqual(transitions[0].kind, "blur")

    def test_min_gap_prevents_stacking(self):
        from reelforge.brain import plan_transitions
        profile = StyleProfile().apply_overrides(["transitions.min_gap=10"])
        self.assertEqual(len(plan_transitions(self.timeline(), self.analysis(), profile)), 1)

    def test_punch_composes_with_the_zoom_curve_and_returns_to_one(self):
        from reelforge.edl import Transition
        from reelforge.render import max_punch_factor, punch_expression
        punches = [Transition("t1", 2.0, "punch", 0.2, 1.0)]
        expression = punch_expression(punches)
        self.assertIn("between(in_time,2.000,2.200)", expression)
        self.assertGreater(max_punch_factor(punches), 1.0)
        # Outside the window the multiplier is exactly 1, so framing is untouched.
        self.assertTrue(expression.rstrip(")").endswith("1"))

    def test_flash_and_blur_expressions(self):
        from reelforge.edl import Transition
        from reelforge.render import blur_enable_expression, flash_expression
        items = [Transition("t1", 1.0, "flash", 0.2, 1.0), Transition("t2", 4.0, "blur", 0.2, 1.0)]
        self.assertIn("between(t,1.000,1.200)", flash_expression(items))
        self.assertEqual(blur_enable_expression(items), "between(t,4.000,4.200)")
        self.assertEqual(flash_expression([items[1]]), "")   # no flashes -> no filter

    def test_disabled_transitions_are_not_rendered(self):
        from reelforge.edl import Transition
        from reelforge.render import flash_expression
        self.assertEqual(flash_expression([Transition("t1", 1.0, "flash", enabled=False)]), "")


class FontAndTemplateTests(unittest.TestCase):
    def test_catalog_entries_are_well_formed(self):
        from reelforge.fonts import CATALOG, resolve
        self.assertGreaterEqual(len(CATALOG), 10)
        for entry in CATALOG:
            self.assertTrue(entry.url.startswith("https://"))
            self.assertTrue(entry.filename.endswith((".ttf", ".otf")))
            self.assertTrue(entry.note)
        self.assertIsNotNone(resolve("cairo"))
        self.assertIsNone(resolve("not-a-font"))

    def test_every_template_loads_and_is_valid(self):
        root = Path(__file__).resolve().parent.parent
        templates = sorted((root / "templates").glob("*.yml"))
        self.assertGreaterEqual(len(templates), 5)
        valid_styles = {"karaoke", "box", "pop", "word", "plain"}
        valid_transitions = {"auto", "punch", "flash", "blur", "none"}
        for path in templates:
            profile = StyleProfile.load(path)
            with self.subTest(template=path.stem):
                self.assertEqual(profile.get("name"), path.stem)
                self.assertTrue(profile.get("description"))
                self.assertIn(profile.get("captions.style"), valid_styles)
                self.assertIn(profile.get("transitions.kind"), valid_transitions)
                # The font must be one we can actually install.
                from reelforge.fonts import resolve
                self.assertIsNotNone(resolve(profile.get("captions.font")),
                                     f"{path.stem} uses an uninstallable font")

    def test_templates_render_valid_ass(self):
        root = Path(__file__).resolve().parent.parent
        words = [Word("وفرت", 0, 0.4), Word("90%", 0.4, 0.8), Word("وقتك", 0.8, 1.2)]
        for path in sorted((root / "templates").glob("*.yml")):
            profile = StyleProfile.load(path)
            with self.subTest(template=path.stem):
                ass = build_ass(group_words(words, profile), profile)
                self.assertIn("[V4+ Styles]", ass)
                self.assertTrue([l for l in ass.splitlines() if l.startswith("Dialogue")])


class MarketParsingTests(unittest.TestCase):
    def test_numbers_from_real_exports(self):
        from reelforge.market import parse_number
        cases = {
            "1,234.56": 1234.56,      # US thousands
            "1.234,56": 1234.56,      # European thousands
            "4,500": 4500.0,          # three digits after a comma is thousands
            "1,23": 1.23,             # two digits is a decimal comma
            "(45.2)": -45.2,          # accounting negative
            "$1,200.00": 1200.0,
            "12%": 12.0,
            "2.3M": 2_300_000.0,
            "٤٥٫٥": 45.5,   # Arabic-Indic digits and decimal mark
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertAlmostEqual(parse_number(raw), expected, places=4)
        for blank in ("", "n/a", "-", None):
            self.assertIsNone(parse_number(blank))

    def test_ambiguous_dates_are_resolved_from_the_whole_column(self):
        from reelforge.market import _day_first, parse_date
        self.assertTrue(_day_first(["05/01/2024", "15/03/2024"]))    # 15 can only be a day
        self.assertFalse(_day_first(["03/15/2024"]))
        self.assertEqual(parse_date("15/03/2024", day_first=True).isoformat(), "2024-03-15")
        self.assertEqual(parse_date("03/15/2024").isoformat(), "2024-03-15")
        self.assertEqual(parse_date("Mar 15, 2024").isoformat(), "2024-03-15")
        self.assertIsNone(parse_date("not a date"))

    def _write(self, tmp, name, text):
        path = Path(tmp) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_yahoo_investing_and_arabic_exports_the_same_way(self):
        from reelforge.market import read_csv
        with tempfile.TemporaryDirectory() as tmp:
            yahoo = self._write(tmp, "y.csv",
                "Date,Open,High,Low,Close,Adj Close,Volume\n"
                "2024-01-02,100,101,99,100.00,100.00,1000\n"
                "2024-01-03,100,103,100,102.00,102.00,1200\n")
            # newest first, d/m/Y, quoted, thousands separators
            investing = self._write(tmp, "i.csv",
                '"Date","Price","Open","High","Low","Vol.","Change %"\n'
                '"03/01/2024","1,102.00","1,100","1,103","1,100","1.2K","2%"\n'
                '"02/01/2024","1,100.00","1,100","1,101","1,099","1.0K","0%"\n')
            arabic = self._write(tmp, "a.csv",
                "التاريخ;الاغلاق\n"
                "2024/01/02;100.00\n2024/01/03;102.00\n")

            for path in (yahoo, investing, arabic):
                bars = read_csv(path)
                with self.subTest(path=path.name):
                    self.assertEqual(len(bars), 2)
                    self.assertEqual(bars[0].day.isoformat(), "2024-01-02")  # sorted oldest first
                    self.assertLess(bars[0].close, bars[1].close)

    def test_a_file_without_price_columns_says_so(self):
        from reelforge.market import MarketError, read_csv
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "bad.csv", "foo,bar\n1,2\n")
            with self.assertRaises(MarketError) as caught:
                read_csv(path)
            self.assertIn("date", str(caught.exception).lower())


class MarketAnalyticsTests(unittest.TestCase):
    def bars(self, pairs):
        from reelforge.market import Bar
        from datetime import date as _date
        return [Bar(day=_date.fromisoformat(d), close=c) for d, c in pairs]

    def test_return_and_cagr_are_exact(self):
        from reelforge.market import cagr, total_return
        # A clean doubling over exactly two years.
        bars = self.bars([("2020-01-01", 100.0), ("2022-01-01", 200.0)])
        self.assertAlmostEqual(total_return(bars), 1.0, places=6)
        self.assertAlmostEqual(cagr(bars), 2 ** 0.5 - 1, places=3)

    def test_cagr_refuses_to_annualise_a_few_days(self):
        from reelforge.market import cagr
        self.assertIsNone(cagr(self.bars([("2024-01-01", 100.0), ("2024-01-10", 130.0)])))

    def test_max_drawdown_finds_peak_and_trough(self):
        from reelforge.market import max_drawdown
        bars = self.bars([("2020-01-01", 100.0), ("2020-06-01", 150.0),
                          ("2020-09-01", 75.0), ("2021-01-01", 120.0)])
        result = max_drawdown(bars)
        self.assertAlmostEqual(result["drawdown"], -0.5, places=6)
        self.assertEqual(result["peak_day"], "2020-06-01")
        self.assertEqual(result["trough_day"], "2020-09-01")

    def test_lump_sum_and_monthly_plan(self):
        from reelforge.market import invest_lump, invest_monthly
        bars = self.bars([("2020-01-01", 100.0), ("2020-02-01", 100.0),
                          ("2020-03-01", 100.0), ("2021-01-01", 200.0)])
        lump = invest_lump(bars, 1000.0)
        self.assertAlmostEqual(lump["value"], 2000.0, places=4)
        self.assertAlmostEqual(lump["multiple"], 2.0, places=6)

        plan = invest_monthly(bars, 100.0)
        self.assertEqual(plan["months"], 4)          # one buy per calendar month
        self.assertAlmostEqual(plan["invested"], 400.0, places=4)
        # Three units bought at 100 plus one at 200, all valued at 200.
        self.assertAlmostEqual(plan["value"], (3 * 1.0 + 0.5) * 200.0, places=4)

    def test_calendar_years_chain_from_the_previous_close(self):
        from reelforge.market import calendar_years
        bars = self.bars([("2020-01-01", 100.0), ("2020-12-31", 110.0),
                          ("2021-12-31", 121.0)])
        years = calendar_years(bars)
        self.assertAlmostEqual(years[2020], 0.10, places=6)
        self.assertAlmostEqual(years[2021], 0.10, places=6)


class MarketStoreTests(unittest.TestCase):
    def store_with(self, tmp, rows, symbol="TEST"):
        from reelforge.market import MarketStore
        path = Path(tmp) / "prices.csv"
        path.write_text("Date,Close\n" + "".join(f"{d},{c}\n" for d, c in rows),
                        encoding="utf-8")
        store = MarketStore(Path(tmp) / "store")
        store.add_csv(path, symbol, name="Test Index", currency="USD")
        return store

    def rows(self):
        out = []
        for year in range(2015, 2025):
            for month in (1, 4, 7, 10):
                out.append((f"{year}-{month:02d}-01", 100.0 * (1.10 ** (year - 2015))))
        out.append(("2024-12-31", 100.0 * (1.10 ** 9) * 1.10))
        return out

    def test_ingest_list_and_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            listed = store.symbols()
            self.assertEqual(listed[0]["symbol"], "TEST")
            self.assertEqual(listed[0]["rows"], len(self.rows()))
            self.assertTrue(store.has("test"))               # case-insensitive
            self.assertEqual(len(store.bars("TEST", "2015-01-01", "2015-12-31")), 4)

    def test_price_on_falls_back_to_the_previous_trading_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            bar = store.price_on("TEST", "2015-02-15")       # no bar that day
            self.assertEqual(bar.day.isoformat(), "2015-01-01")
            self.assertIsNone(store.price_on("TEST", "2000-01-01"))

    def test_reingesting_updates_rather_than_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            before = store.symbols()[0]["rows"]
            path = Path(tmp) / "prices.csv"
            store.add_csv(path, "TEST")
            self.assertEqual(store.symbols()[0]["rows"], before)

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            store.remove("TEST")
            self.assertFalse(store.has("TEST"))

    def test_fact_sheet_produces_citable_statements(self):
        from reelforge.market import fact_sheet
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            facts = fact_sheet(store, "TEST", amount=1000, monthly=100)
            self.assertAlmostEqual(facts["cagr"], 0.10, places=2)   # built as 10% a year
            self.assertTrue(facts["statements"]["en"])
            self.assertTrue(facts["statements"]["ar"])
            self.assertIn("TEST", facts["symbol"])
            self.assertIsNotNone(facts["monthly_plan"])
            # Arabic statements must carry real Arabic, not a transliteration.
            self.assertTrue(any(arabic.is_arabic(line) for line in facts["statements"]["ar"]))

    def test_partial_years_are_never_quoted_as_best_or_worst(self):
        from reelforge.market import fact_sheet
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            facts = fact_sheet(store, "TEST", start="2016-04-01", end="2023-07-01")
            for label in ("best_year", "worst_year"):
                if facts[label]:
                    self.assertNotIn(facts[label][0], (2016, 2023),
                                     f"{label} quoted a part-year")

    def test_unknown_symbol_explains_how_to_add_one(self):
        from reelforge.market import MarketError, fact_sheet
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            with self.assertRaises(MarketError) as caught:
                fact_sheet(store, "NOPE")
            self.assertIn("market add", str(caught.exception))

    def test_compare_aligns_to_the_overlapping_window(self):
        from reelforge.market import compare
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store_with(tmp, self.rows())
            short = Path(tmp) / "short.csv"
            short.write_text("Date,Close\n2018-01-01,100\n2020-01-01,150\n2024-12-31,300\n",
                             encoding="utf-8")
            store.add_csv(short, "SHORT")
            result = compare(store, ["TEST", "SHORT"])
            # The window must start where the later series starts.
            self.assertEqual(result["start_day"], "2018-01-01")
            self.assertEqual(len(result["rows"]), 2)
            self.assertTrue(all(row["cagr"] is not None for row in result["rows"]))


class LLMProviderTests(unittest.TestCase):
    def test_json_is_recovered_from_fences_and_prose(self):
        from reelforge.llm import _maybe_json
        self.assertEqual(_maybe_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_maybe_json('sure thing: {"b": 2} done'), {"b": 2})
        self.assertIsNone(_maybe_json("no json here"))
        self.assertIsNone(_maybe_json(""))
        self.assertIsNone(_maybe_json("[1,2,3]"))      # array is not a script payload

    def test_explicit_provider_is_honoured(self):
        from reelforge.llm import available_provider
        self.assertEqual(available_provider("ollama"), "ollama")
        self.assertEqual(available_provider("stub"), "stub")

    def test_stub_returns_no_payload_rather_than_fake_prose(self):
        from reelforge.llm import generate
        response = generate("write me a script", system="x", schema={}, provider="stub")
        self.assertEqual(response.provider, "stub")
        self.assertIsNone(response.data)
        self.assertEqual(response.text, "")

    def _fake_anthropic(self, recorder, *, beta_raises=None, stop_reason="end_turn"):
        """A stand-in anthropic module that records the request we build."""
        import types

        class Block:
            def __init__(self, text):
                self.type, self.text = "text", text

        class Usage:
            input_tokens, output_tokens = 10, 20

        class Response:
            def __init__(self):
                self.content = [Block('{"title": "t", "caption_hook": "h", "beats": []}')]
                self.usage = Usage()
                self.stop_reason = stop_reason

        class Messages:
            def create(self, **kwargs):
                recorder.setdefault("calls", []).append(("stable", kwargs))
                return Response()

        class BetaMessages:
            def create(self, **kwargs):
                recorder.setdefault("calls", []).append(("beta", kwargs))
                if beta_raises:
                    raise beta_raises
                return Response()

        class Client:
            def __init__(self, *a, **k):
                self.messages = Messages()
                self.beta = types.SimpleNamespace(messages=BetaMessages())

        module = types.ModuleType("anthropic")
        module.Anthropic = Client
        return module

    def _run_claude(self, module):
        import sys as _sys
        from reelforge.llm import generate_claude
        original = _sys.modules.get("anthropic")
        _sys.modules["anthropic"] = module
        try:
            return generate_claude("prompt", system="system", schema={"type": "object"})
        finally:
            if original is None:
                _sys.modules.pop("anthropic", None)
            else:
                _sys.modules["anthropic"] = original

    def test_claude_request_shape(self):
        recorder: dict = {}
        self._run_claude(self._fake_anthropic(recorder))
        kind, kwargs = recorder["calls"][0]
        self.assertEqual(kind, "beta")
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
        self.assertEqual(kwargs["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kwargs["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", kwargs["betas"])
        # No assistant prefill - removed on current models.
        self.assertEqual([m["role"] for m in kwargs["messages"]], ["user"])

    def test_claude_falls_back_when_the_sdk_predates_fallbacks(self):
        recorder: dict = {}
        module = self._fake_anthropic(recorder, beta_raises=TypeError("unexpected kwarg"))
        response = self._run_claude(module)
        kinds = [kind for kind, _ in recorder["calls"]]
        self.assertEqual(kinds, ["beta", "stable"])
        self.assertIsNotNone(response.data)          # still produced a script payload
        self.assertNotIn("fallbacks", recorder["calls"][1][1])

    def test_claude_surfaces_a_refusal_clearly(self):
        from reelforge.llm import LLMError
        recorder: dict = {}
        module = self._fake_anthropic(recorder, stop_reason="refusal")
        with self.assertRaises(LLMError):
            self._run_claude(module)


class StudioTests(unittest.TestCase):
    def test_init_creates_files_and_copies_frameworks(self):
        from reelforge.knowledge import Studio
        with tempfile.TemporaryDirectory() as tmp:
            studio = Studio(Path(tmp) / "studio")
            written = studio.init()
            self.assertTrue(studio.exists)
            self.assertTrue(any(p.name == "background.md" for p in written))
            self.assertGreaterEqual(len(studio.frameworks()), 5)

    def test_placeholder_studio_is_detected(self):
        from reelforge.knowledge import Studio
        with tempfile.TemporaryDirectory() as tmp:
            studio = Studio(Path(tmp) / "studio")
            studio.init()
            self.assertTrue(studio.is_unedited())
            (studio.root / "background.md").write_text("I am a real person.", encoding="utf-8")
            self.assertFalse(studio.is_unedited())

    def test_defaults_contain_no_invented_biography(self):
        from reelforge.knowledge import BACKGROUND
        # A plausible-sounding fake credential would get shipped by accident.
        for invented in ("years of experience", "CFA", "portfolio manager", "I have"):
            self.assertNotIn(invented, BACKGROUND)

    def test_studio_framework_overrides_the_builtin(self):
        from reelforge.knowledge import Studio
        with tempfile.TemporaryDirectory() as tmp:
            studio = Studio(Path(tmp) / "studio")
            studio.init()
            (studio.frameworks_dir / "number-story.yml").write_text(
                "name: number-story\ndescription: mine\nbeats:\n  - role: hook\n"
                "    purpose: p\n    seconds: 3\n", encoding="utf-8")
            self.assertEqual(studio.framework("number-story").description, "mine")

    def test_unknown_framework_lists_the_options(self):
        from reelforge.knowledge import Studio
        with tempfile.TemporaryDirectory() as tmp:
            studio = Studio(Path(tmp) / "studio")
            studio.init()
            with self.assertRaises(FileNotFoundError) as caught:
                studio.framework("nope")
            self.assertIn("number-story", str(caught.exception))


class ScriptTests(unittest.TestCase):
    def facts(self):
        return {
            "symbol": "TEST", "name": "Test", "start_day": "2015-01-01",
            "end_day": "2024-12-31", "years": 10.0, "start_price": 100.0,
            "end_price": 173.5, "total_return": 0.735, "cagr": 0.0566,
            "volatility": 0.16, "calendar_years": {"2015": 0.58},
            "max_drawdown": {"drawdown": -0.339, "peak_day": "2020-02-19",
                             "trough_day": "2020-03-23", "peak": 150.0, "trough": 99.0},
            "lump_sum": {"invested": 1000.0, "value": 1735.0, "multiple": 1.735},
            "monthly_plan": None, "best_year": (2015, 0.58), "worst_year": (2019, -0.142),
            "statements": {"en": ["Test returned 73.5% over 10.0 years."],
                           "ar": ["Test حقق 73.5% خلال 10.0 سنة."]},
        }

    def test_quoted_numbers_pass_the_audit(self):
        from reelforge.script import audit_numbers
        self.assertEqual(audit_numbers("حقق 73.5% على مدى 10 سنوات", self.facts()), [])
        self.assertEqual(audit_numbers("worth 1,735 after 1,000 invested", self.facts()), [])

    def test_invented_numbers_are_caught(self):
        from reelforge.script import audit_numbers
        flagged = audit_numbers("it returned 91.4% and will hit 5000 next year", self.facts())
        self.assertIn("91.4", flagged)
        self.assertIn("5000", flagged)

    def test_arabic_indic_digits_are_audited_too(self):
        from reelforge.script import audit_numbers
        self.assertTrue(audit_numbers("٩١٫٤", self.facts()))

    def test_small_counts_are_not_treated_as_claims(self):
        from reelforge.script import audit_numbers
        self.assertEqual(audit_numbers("3 steps, 5 minutes, 2 rules", self.facts()), [])

    def test_rounding_is_tolerated(self):
        from reelforge.script import audit_numbers
        # 5.66% quoted as 5.7% is a rounding of a real number, not an invention.
        self.assertEqual(audit_numbers("about 5.7% a year", self.facts()), [])

    def test_audit_with_no_facts_flags_every_statistic(self):
        from reelforge.script import audit_numbers
        self.assertIn("42.5", audit_numbers("markets rose 42.5%", None))

    def test_generation_parses_a_model_payload_and_audits_it(self):
        from reelforge.knowledge import Studio
        from reelforge.llm import PROVIDERS, LLMResponse
        from reelforge.script import write_script

        payload = {
            "title": "T", "caption_hook": "hook",
            "beats": [{"role": "hook", "text": "حقق 73.5% في 10 سنين", "seconds": 4,
                       "onscreen": "73.5%", "broll": ["فلوس"]},
                      {"role": "lesson", "text": "بس هيعمل 99.9% السنة الجاية",
                       "seconds": 6, "onscreen": "", "broll": []}],
        }

        def fake(prompt, *, system, schema=None, model=None, max_tokens=16000):
            fake.system = system
            fake.prompt = prompt
            return LLMResponse(text="", provider="claude", model="claude-opus-5",
                               data=payload)

        original = PROVIDERS["claude"]
        PROVIDERS["claude"] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                studio = Studio(Path(tmp) / "studio")
                studio.init()
                script = write_script("topic", studio, framework="number-story",
                                      facts=self.facts(), provider="claude")
        finally:
            PROVIDERS["claude"] = original

        self.assertEqual(len(script.beats), 2)
        self.assertEqual(script.beats[0].broll, ["فلوس"])
        self.assertIn("99.9", script.unverified_numbers)   # the invented one
        self.assertNotIn("73.5", script.unverified_numbers)
        # The facts must be handed to the model, and the no-invention rule stated.
        self.assertIn("73.5%", fake.prompt)
        self.assertIn("never compute", fake.system.lower())

    def test_stub_yields_an_obviously_empty_script(self):
        from reelforge.knowledge import Studio
        from reelforge.script import write_script
        with tempfile.TemporaryDirectory() as tmp:
            studio = Studio(Path(tmp) / "studio")
            studio.init()
            script = write_script("topic", studio, facts=self.facts(), provider="stub")
            self.assertEqual(script.provider, "stub")
            self.assertTrue(script.beats)
            # Some beats carry facts; none carry invented prose.
            self.assertEqual(script.unverified_numbers, [])
            self.assertTrue(any(not b.text for b in script.beats))

    def test_script_round_trip_and_editor_handoff(self):
        from reelforge.script import Beat, Script, caption_prior, script_vocabulary
        script = Script(topic="t", framework="number-story", language="ar",
                        beats=[Beat("hook", "الذكاء الاصطناعي بيوفر وقتك", 4.0),
                               Beat("cta", "تابعني", 2.0)])
        restored = Script.from_dict(json.loads(json.dumps(script.to_dict())))
        self.assertEqual(restored.spoken_text, script.spoken_text)
        self.assertEqual(restored.seconds, 6.0)

        self.assertIn("الذكاء", caption_prior(script))
        vocabulary = script_vocabulary(script)
        self.assertTrue(vocabulary)
        self.assertTrue(all(len(word) >= 4 for word in vocabulary.values()))

    def test_markdown_flags_unverified_numbers_for_the_reader(self):
        from reelforge.script import Beat, Script
        script = Script(topic="t", framework="f", language="ar",
                        beats=[Beat("hook", "x", 3.0)], unverified_numbers=["91.4"])
        self.assertIn("91.4", script.to_markdown())
        self.assertIn("not in your data", script.to_markdown())


class JoinTests(unittest.TestCase):
    """Several takes shot on a phone, treated as one video."""

    def info(self, width, height, fps=30.0, audio=True, rate=48000, duration=3.0):
        from reelforge.ffmpeg import MediaInfo
        return MediaInfo(path=Path("x.mp4"), duration=duration, width=width, height=height,
                         fps=fps, has_audio=audio, audio_rate=rate, rotation=0, size_bytes=1)

    def test_canvas_follows_the_dominant_orientation(self):
        from reelforge.join import target_shape
        # Six portrait phone takes plus one landscape screen recording must not
        # produce a square canvas that pads every single clip.
        infos = [self.info(1080, 1920) for _ in range(6)] + [self.info(1920, 1080)]
        width, height, _fps, _rate = target_shape(infos)
        self.assertEqual((width, height), (1080, 1920))

    def test_landscape_majority_keeps_landscape(self):
        from reelforge.join import target_shape
        infos = [self.info(1920, 1080), self.info(1280, 720), self.info(1080, 1920)]
        width, height, _f, _r = target_shape(infos)
        self.assertEqual((width, height), (1920, 1080))

    def test_shape_takes_the_largest_of_the_dominant_group(self):
        from reelforge.join import target_shape
        infos = [self.info(720, 1280, fps=24), self.info(1080, 1920, fps=30)]
        width, height, fps, _r = target_shape(infos)
        self.assertEqual((width, height, fps), (1080, 1920, 30))

    def test_frame_rate_is_capped(self):
        from reelforge.join import target_shape
        _w, _h, fps, _r = target_shape([self.info(1080, 1920, fps=240)])
        self.assertLessEqual(fps, 60)

    def test_audio_rate_defaults_when_every_clip_is_silent(self):
        from reelforge.join import target_shape
        _w, _h, _f, rate = target_shape([self.info(1080, 1920, audio=False, rate=0)])
        self.assertEqual(rate, 48000)

    def test_single_clip_is_passed_through_untouched(self):
        from reelforge.join import join_clips
        with tempfile.TemporaryDirectory() as tmp:
            only = Path(tmp) / "a.mp4"
            only.write_bytes(b"x")
            self.assertEqual(join_clips([only], Path(tmp) / "out.mp4"), only)


@needs_ffmpeg
class JoinRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reelforge-join-")
        root = Path(cls.tmp)

        def make(name, size, rate, duration, audio=True):
            path = root / name
            args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={duration}"]
            if audio:
                args += ["-f", "lavfi", "-i", f"sine=f=300:d={duration}", "-c:a", "aac"]
            else:
                args += ["-an"]
            args += ["-pix_fmt", "yuv420p", str(path)]
            subprocess.run(args, check=True, capture_output=True)
            return path

        cls.portrait = make("a.mp4", "480x854", 30, 2)
        cls.small = make("b.mp4", "360x640", 24, 2)
        cls.silent = make("c.mp4", "640x360", 25, 2, audio=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_mismatched_clips_join_into_one_continuous_file(self):
        from reelforge.ffmpeg import probe
        from reelforge.join import join_clips
        out = Path(self.tmp) / "joined.mp4"
        join_clips([self.portrait, self.small, self.silent], out)

        info = probe(out)
        self.assertAlmostEqual(info.duration, 6.0, delta=0.5)     # 2 + 2 + 2
        self.assertEqual((info.width, info.height), (480, 854))   # portrait majority
        # The silent clip must not knock out the audio track for the others.
        self.assertTrue(info.has_audio)

    def test_clip_boundaries_mark_the_joins(self):
        from reelforge.join import clip_boundaries
        boundaries = clip_boundaries([self.portrait, self.small, self.silent])
        self.assertEqual(len(boundaries), 2)
        self.assertAlmostEqual(boundaries[0], 2.0, delta=0.3)

    def test_a_silent_first_clip_still_yields_audio(self):
        from reelforge.ffmpeg import probe
        from reelforge.join import join_clips
        out = Path(self.tmp) / "joined-silent-first.mp4"
        join_clips([self.silent, self.portrait], out)
        info = probe(out)
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 4.0, delta=0.5)


class VideoResolutionTests(unittest.TestCase):
    def test_wildcards_are_expanded_since_powershell_does_not(self):
        from reelforge.cli import _resolve_videos
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("IMG_7413.MP4", "IMG_7414.MP4", "notes.txt"):
                (root / name).write_bytes(b"x")
            found = _resolve_videos([str(root / "IMG_*.MP4")])
            self.assertEqual([p.name for p in found], ["IMG_7413.MP4", "IMG_7414.MP4"])

    def test_ordering_options(self):
        import os
        from reelforge.cli import _resolve_videos
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "b.mp4", root / "a.mp4"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            os.utime(first, (1_000_000, 1_000_000))       # b is older
            os.utime(second, (2_000_000, 2_000_000))

            given = _resolve_videos([str(first), str(second)], order="given")
            self.assertEqual([p.name for p in given], ["b.mp4", "a.mp4"])
            by_name = _resolve_videos([str(first), str(second)], order="name")
            self.assertEqual([p.name for p in by_name], ["a.mp4", "b.mp4"])
            by_time = _resolve_videos([str(first), str(second)], order="time")
            self.assertEqual([p.name for p in by_time], ["b.mp4", "a.mp4"])

    def test_duplicates_are_dropped(self):
        from reelforge.cli import _resolve_videos
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "a.mp4"
            clip.write_bytes(b"x")
            self.assertEqual(len(_resolve_videos([str(clip), str(clip)])), 1)

    def test_a_pattern_matching_nothing_says_where_it_looked(self):
        from reelforge.cli import _resolve_videos
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                _resolve_videos([str(Path(tmp) / "*.mp4")])
            self.assertIn(tmp, str(caught.exception))


class FiltergraphMechanismTests(unittest.TestCase):
    """`-filter_complex_script` was removed in ffmpeg 8; `-/filter_complex` arrived in 7."""

    def test_mode_order_follows_the_installed_version(self):
        import reelforge.ffmpeg as ffmpeg_module
        original = ffmpeg_module.version_tuple
        try:
            ffmpeg_module.version_tuple = lambda: (9, 0)
            self.assertEqual(ffmpeg_module._filter_mode_order()[0], "new")
            ffmpeg_module.version_tuple = lambda: (6, 1)
            self.assertEqual(ffmpeg_module._filter_mode_order()[0], "old")
            ffmpeg_module.version_tuple = lambda: None       # git build
            self.assertEqual(ffmpeg_module._filter_mode_order()[0], "new")
        finally:
            ffmpeg_module.version_tuple = original

    def test_inline_is_always_the_last_resort(self):
        import reelforge.ffmpeg as ffmpeg_module
        self.assertEqual(ffmpeg_module._filter_mode_order()[-1], "inline")

    def test_each_mode_builds_the_right_arguments(self):
        import reelforge.ffmpeg as ffmpeg_module
        script = Path("g.txt")
        self.assertEqual(ffmpeg_module._filter_args("new", script, "G"),
                         ["-/filter_complex", "g.txt"])
        self.assertEqual(ffmpeg_module._filter_args("old", script, "G"),
                         ["-filter_complex_script", "g.txt"])
        self.assertEqual(ffmpeg_module._filter_args("inline", script, "G"),
                         ["-filter_complex", "G"])

    @needs_ffmpeg
    def test_a_rejected_flag_is_retried_with_the_other_one(self):
        """The failure the user hit: the preferred flag does not exist on their build."""
        import reelforge.ffmpeg as ffmpeg_module
        original_order = ffmpeg_module._filter_mode_order
        original_mode = ffmpeg_module._FILTER_MODE
        with tempfile.TemporaryDirectory() as tmp:
            try:
                # Force the wrong flag first, whichever this ffmpeg dislikes.
                rejected = "new" if (ffmpeg_module.version_tuple() or (6,))[0] < 7 else "old"
                keeper = "old" if rejected == "new" else "new"
                ffmpeg_module._filter_mode_order = lambda: [rejected, keeper, "inline"]
                ffmpeg_module._FILTER_MODE = None

                out = Path(tmp) / "out.png"
                ffmpeg_module.run_filtergraph(
                    ["-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.2"],
                    "[0:v]scale=32:32[outv]", Path(tmp) / "g.txt",
                    ["-map", "[outv]", "-frames:v", "1", str(out)],
                )
                self.assertTrue(out.exists() and out.stat().st_size > 0)
                # And it remembers, so later renders do not pay for the retry.
                self.assertEqual(ffmpeg_module._FILTER_MODE, keeper)
            finally:
                ffmpeg_module._filter_mode_order = original_order
                ffmpeg_module._FILTER_MODE = original_mode


class JoinCapTests(unittest.TestCase):
    def info(self, width, height):
        from reelforge.ffmpeg import MediaInfo
        return MediaInfo(path=Path("x.mp4"), duration=5.0, width=width, height=height,
                         fps=30.0, has_audio=True, audio_rate=48000, rotation=0, size_bytes=1)

    def test_4k_phone_clips_are_capped_to_what_the_render_uses(self):
        from reelforge.join import target_shape
        width, height, _f, _r = target_shape([self.info(2160, 3840)] * 3, max_height=2592)
        self.assertEqual(height, 2592)
        self.assertAlmostEqual(width / height, 2160 / 3840, places=2)   # aspect kept

    def test_smaller_clips_are_never_upscaled(self):
        from reelforge.join import target_shape
        width, height, _f, _r = target_shape([self.info(1080, 1920)], max_height=2592)
        self.assertEqual((width, height), (1080, 1920))

    def test_cap_comes_from_output_size_and_headroom(self):
        from reelforge.cli import _join_cap
        profile = StyleProfile().apply_overrides(["output.height=1920",
                                                  "output.zoom_headroom=1.35"])
        self.assertEqual(_join_cap(profile), 2592)


@needs_ffmpeg
class FiltergraphPathTests(unittest.TestCase):
    """Windows paths carry a drive colon and backslashes - the filtergraph's own
    separator and escape characters. The renderer keeps paths out of the graph
    entirely rather than trying to escape them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reelforge-paths-")
        cls.clip = make_clip(Path(cls.tmp) / "clip.mp4", duration=4)
        # A directory carrying every character that broke the filtergraph parser.
        cls.hostile = Path(cls.tmp) / "C: drive\\path, weird [dir]"
        cls.hostile.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def editor(self):
        profile = StyleProfile().apply_overrides(["asr.backend=stub"])
        return AutoEditor(profile, project_dir=self.hostile / ".reelforge",
                          fonts_dir=Path(__file__).resolve().parent.parent / "assets" / "fonts")

    def test_renders_from_a_path_full_of_filtergraph_metacharacters(self):
        editor = self.editor()
        result = editor.plan(self.clip)
        out = editor.render(result.edl, self.hostile / "out reel.mp4", preview=True)
        self.assertTrue(out.exists() and out.stat().st_size > 0)

    def test_the_subtitle_filter_uses_bare_names_not_paths(self):
        editor = self.editor()
        result = editor.plan(self.clip)
        editor.render(result.edl, self.hostile / "out2.mp4", preview=True)

        graph = (editor.work_dir / "filtergraph_preview.txt").read_text("utf-8")
        ass_segment = next(seg for seg in graph.split(";") if "ass=" in seg)
        self.assertIn("ass=captions.ass", ass_segment)
        self.assertNotIn(str(self.hostile), graph)
        self.assertNotIn("\\", graph)          # no escaped path fragments anywhere

    def test_fonts_are_copied_next_to_the_subtitles(self):
        editor = self.editor()
        result = editor.plan(self.clip)
        editor.render(result.edl, self.hostile / "out3.mp4", preview=True)
        fonts = editor.work_dir / "fonts"
        self.assertTrue(fonts.exists())
        self.assertTrue(any(f.suffix.lower() == ".ttf" for f in fonts.iterdir()))

    def test_escape_helper_converts_windows_separators(self):
        from reelforge.render import _escape_path
        escaped = _escape_path(r"C:\Users\osama\captions.ass")
        self.assertEqual(escaped, "C\\:/Users/osama/captions.ass")
        self.assertNotIn("\\U", escaped)       # no stray backslash-letter sequences


class VisualOrderTests(unittest.TestCase):
    """libass loses bidi across override tags, so we order words for display."""

    def order(self, words):
        return [words[i] for i in arabic.visual_order(words)]

    def test_arabic_words_are_laid_out_right_to_left(self):
        words = ["واحد", "اثنين", "ثلاثة", "اربعة"]
        self.assertEqual(self.order(words), ["اربعة", "ثلاثة", "اثنين", "واحد"])

    def test_a_latin_only_line_is_left_alone(self):
        words = ["hello", "world", "now"]
        self.assertEqual(self.order(words), words)

    def test_latin_runs_keep_their_own_order_inside_an_arabic_line(self):
        # The bidi algorithm reverses the Arabic but not the Latin phrase.
        words = ["استخدم", "Claude", "Code", "اليوم"]
        self.assertEqual(self.order(words), ["اليوم", "Claude", "Code", "استخدم"])

    def test_numbers_follow_the_right_to_left_flow(self):
        words = ["وفرت", "90%", "من", "وقتك"]
        self.assertEqual(self.order(words), ["وقتك", "من", "90%", "وفرت"])

    def test_single_word_and_empty(self):
        self.assertEqual(self.order(["انت"]), ["انت"])
        self.assertEqual(arabic.visual_order([]), [])


class CaptionOrderingTests(unittest.TestCase):
    def line(self, texts):
        words, t = [], 0.0
        for text in texts:
            words.append(Word(text, t, t + 0.4, 1.0))
            t += 0.5
        return CaptionLine(words=words, start=0.0, end=t)

    def body(self, event):
        return event.split(",", 9)[9]

    def events(self, ass):
        return [l for l in ass.splitlines() if l.startswith("Dialogue")]

    def test_tagged_arabic_lines_are_emitted_in_visual_order(self):
        profile = StyleProfile().apply_overrides(["captions.style=karaoke"])
        line = self.line(["واحد", "اثنين", "ثلاثة", "اربعة"])
        first = self.body(self.events(build_ass([line], profile))[0])
        # The line reads right-to-left, so the last word is written first.
        self.assertTrue(first.startswith("اربعة"), first)
        self.assertTrue(first.rstrip().endswith("}"), first)   # active word is last
        self.assertIn("واحد", first.split("}")[-2] if "}" in first else first)

    def test_the_spoken_word_is_the_one_marked(self):
        profile = StyleProfile().apply_overrides(["captions.style=karaoke"])
        line = self.line(["واحد", "اثنين", "ثلاثة", "اربعة"])
        events = self.events(build_ass([line], profile))
        self.assertEqual(len(events), 4)
        for index, expected in enumerate(["واحد", "اثنين", "ثلاثة", "اربعة"]):
            body = self.body(events[index])
            marked = body.split("{\\c&H00D7FF&}")[1].split("{")[0]
            self.assertEqual(marked.strip(), expected)

    def test_an_untagged_line_is_left_in_logical_order_for_libass(self):
        # With no override tags libass does the bidi correctly itself.
        profile = StyleProfile().apply_overrides(["captions.style=plain",
                                                  "captions.emphasis=false"])
        line = self.line(["واحد", "اثنين", "ثلاثة", "اربعة"])
        body = self.body(self.events(build_ass([line], profile))[0])
        self.assertTrue(body.startswith("واحد"), body)

    def test_emphasis_alone_still_triggers_visual_order(self):
        # An emphasised number inserts a tag, which costs us bidi just the same.
        profile = StyleProfile().apply_overrides(["captions.style=plain"])
        line = self.line(["وفرت", "90%", "وقتك"])
        body = self.body(self.events(build_ass([line], profile))[0])
        self.assertTrue(body.startswith("وقتك"), body)

    def test_latin_captions_are_unaffected(self):
        profile = StyleProfile().apply_overrides(["captions.style=karaoke"])
        line = self.line(["saved", "most", "time"])
        body = self.body(self.events(build_ass([line], profile))[0])
        self.assertIn("saved", body.split("}")[1] if "}" in body else body)


@needs_ffmpeg
class CaptionPixelOrderTests(unittest.TestCase):
    """Measured, not eyeballed - reading Arabic glyph order from a picture is
    exactly how this bug survived review in the first place."""

    def render_positions(self, profile, words):
        from reelforge.captions import CaptionLine as Line
        colours = ["&H0000FF&", "&H00FF00&", "&HFF0000&", "&H00FFFF&"]
        targets = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        with tempfile.TemporaryDirectory() as tmp:
            timed, t = [], 0.0
            for text in words:
                timed.append(Word(text, t, t + 0.4, 1.0))
                t += 0.5
            ass = build_ass([Line(words=timed, start=0.0, end=t)], profile,
                            width=900, height=200)
            # Recolour each word so its position can be measured unambiguously.
            lines, done = [], False
            for row in ass.splitlines():
                if row.startswith("Dialogue"):
                    if done:
                        continue
                    head = row.split(",", 9)
                    body = re.sub(r"\{[^}]*\}", "", head[9])
                    for word, colour in zip(words, colours):
                        body = body.replace(word, "{\\c%s}%s" % (colour, word))
                    lines.append(",".join(head[:9]) + "," + body)
                    done = True
                else:
                    lines.append(row)
            path = Path(tmp) / "m.ass"
            path.write_text("\n".join(lines), encoding="utf-8")

            fonts = Path(__file__).resolve().parent.parent / "assets" / "fonts"
            raw = Path(tmp) / "m.raw"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", "color=c=black:s=900x200:d=1:r=5",
                            "-vf", f"ass={path.name}:fontsdir={fonts}",
                            "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo",
                            str(raw)], check=True, capture_output=True, cwd=tmp)
            data = raw.read_bytes()

        width, height = 900, 200
        found: dict[int, list[int]] = {}
        for y in range(height):
            for x in range(width):
                i = (y * width + x) * 3
                r, g, b = data[i], data[i + 1], data[i + 2]
                if r + g + b < 90:
                    continue
                for index, (tr, tg, tb) in enumerate(targets[:len(words)]):
                    if abs(r - tr) < 70 and abs(g - tg) < 70 and abs(b - tb) < 70:
                        found.setdefault(index, []).append(x)
                        break
        centres = {i: sum(xs) / len(xs) for i, xs in found.items() if xs}
        return [i for i, _ in sorted(centres.items(), key=lambda kv: kv[1])]

    def test_arabic_karaoke_line_reads_right_to_left_on_screen(self):
        profile = StyleProfile().apply_overrides([
            "captions.style=karaoke", "captions.font_size=64", "captions.outline=0",
            "captions.shadow=0", "captions.y_pct=0.5", "captions.safe_area=false",
            "captions.max_words=4"])
        order = self.render_positions(profile, ["واحد", "اثنين", "ثلاثة", "اربعة"])
        # Left to right on screen must be the last word first.
        self.assertEqual(order, [3, 2, 1, 0])


class CaptionTimingTests(unittest.TestCase):
    """Two lines on screen at once reads as a repeated word; lines that touch
    never blink off, so one statement runs into the next."""

    def overlapping_words(self):
        # Whisper emits segments independently, so words straddle the boundary.
        return [Word("انت", 0.00, 0.60), Word("و", 0.55, 0.80),
                Word("صاحبك", 0.75, 1.30), Word("كل", 1.20, 1.60),
                Word("واحد", 1.55, 2.10), Word("فيكوا", 2.05, 2.60)]

    def test_enforce_order_removes_overlap_but_keeps_the_words(self):
        from reelforge.speech import enforce_order
        fixed = enforce_order(self.overlapping_words())
        self.assertEqual([w.text for w in fixed],
                         [w.text for w in self.overlapping_words()])
        for earlier, later in zip(fixed, fixed[1:]):
            self.assertLessEqual(earlier.end, later.start)
            self.assertGreater(earlier.end, earlier.start)

    def test_enforce_order_sorts_out_of_order_input(self):
        from reelforge.speech import enforce_order
        jumbled = [Word("b", 1.0, 1.5), Word("a", 0.0, 0.5)]
        self.assertEqual([w.text for w in enforce_order(jumbled)], ["a", "b"])

    def test_lines_never_overlap_even_from_overlapping_words(self):
        from reelforge.speech import enforce_order
        profile = StyleProfile().apply_overrides(["captions.max_words=3"])
        lines = group_words(enforce_order(self.overlapping_words()), profile)
        self.assertGreaterEqual(len(lines), 2)
        for earlier, later in zip(lines, lines[1:]):
            self.assertLessEqual(earlier.end, later.start,
                                 "two caption lines would be on screen at once")

    def test_there_is_a_visible_break_between_lines(self):
        from reelforge.speech import enforce_order
        profile = StyleProfile().apply_overrides(["captions.max_words=3",
                                                  "captions.line_gap=0.08"])
        lines = group_words(enforce_order(self.overlapping_words()), profile)
        for earlier, later in zip(lines, lines[1:]):
            self.assertGreaterEqual(round(later.start - earlier.end, 3), 0.079)

    def test_short_line_padding_does_not_eat_the_next_line(self):
        profile = StyleProfile().apply_overrides(["captions.max_words=1",
                                                  "captions.min_duration=2.0",
                                                  "captions.line_gap=0.08"])
        words = [Word("انت", 0.0, 0.2), Word("كل", 0.5, 0.7), Word("واحد", 1.0, 1.2)]
        lines = group_words(words, profile)
        for earlier, later in zip(lines, lines[1:]):
            self.assertLessEqual(earlier.end, later.start)

    def test_a_line_is_never_collapsed_to_nothing(self):
        profile = StyleProfile().apply_overrides(["captions.max_words=1",
                                                  "captions.line_gap=0.5"])
        words = [Word("انت", 0.0, 0.2), Word("كل", 0.25, 0.45)]
        for line in group_words(words, profile):
            self.assertGreater(line.end, line.start)

    def test_karaoke_events_still_tile_a_line_without_gaps(self):
        # Inside a line the marker must move with no blink; the gap is only
        # between lines.
        profile = StyleProfile().apply_overrides(["captions.style=karaoke",
                                                  "captions.max_words=4"])
        words = [Word("انت", 0.0, 0.4), Word("و", 0.4, 0.6),
                 Word("صاحبك", 0.6, 1.1), Word("كل", 1.1, 1.5)]
        ass = build_ass(group_words(words, profile), profile)
        events = [l for l in ass.splitlines() if l.startswith("Dialogue")]
        stamps = [(e.split(",")[1], e.split(",")[2]) for e in events]
        for (_, end), (start, _) in zip(stamps, stamps[1:]):
            self.assertEqual(end, start)

    def test_end_to_end_edl_has_no_overlapping_captions(self):
        from reelforge.analysis import Analysis
        from reelforge.speech import Segment, Transcript
        profile = StyleProfile().apply_overrides(["cuts.enabled=false"])
        analysis = Analysis(duration=5.0, width=1080, height=1920, fps=30.0,
                            has_audio=True)
        # Two segments whose words overlap at the join, as real output does.
        transcript = Transcript(language="ar", backend="test", model="t", segments=[
            Segment(text="انت و صاحبك", start=0.0, end=1.3, words=[
                Word("انت", 0.0, 0.6), Word("و", 0.55, 0.8), Word("صاحبك", 0.75, 1.3)]),
            Segment(text="كل واحد", start=1.2, end=2.1, words=[
                Word("كل", 1.2, 1.6), Word("واحد", 1.55, 2.1)]),
        ])
        edl = build_edl("x.mp4", analysis, transcript, profile)
        for earlier, later in zip(edl.captions, edl.captions[1:]):
            self.assertLessEqual(earlier.end, later.start)


@needs_ffmpeg
class JoinCacheTests(unittest.TestCase):
    """Joining re-encodes every frame; repeating it on unchanged clips is waste."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reelforge-joincache-")
        root = Path(cls.tmp)

        def make(name, duration=1):
            path = root / name
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", f"testsrc2=s=240x426:r=25:d={duration}",
                            "-f", "lavfi", "-i", f"sine=f=300:d={duration}",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                           check=True, capture_output=True)
            return path

        cls.a, cls.b = make("a.mp4"), make("b.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_an_unchanged_set_of_clips_is_not_rejoined(self):
        from reelforge.join import join_clips
        dest = Path(self.tmp) / "j.mp4"
        join_clips([self.a, self.b], dest)
        first = dest.stat().st_mtime_ns

        messages = []
        join_clips([self.a, self.b], dest, on_status=messages.append)
        self.assertEqual(dest.stat().st_mtime_ns, first, "the file was re-encoded")
        self.assertTrue(any("reusing" in m for m in messages), messages)

    def test_a_changed_clip_forces_a_rejoin(self):
        from reelforge.join import join_clips
        dest = Path(self.tmp) / "j2.mp4"
        join_clips([self.a, self.b], dest)
        os.utime(self.b, (1_000_000, 1_000_000))       # the clip was re-recorded
        messages = []
        join_clips([self.a, self.b], dest, on_status=messages.append)
        self.assertTrue(any("joining" in m for m in messages), messages)

    def test_a_different_order_is_a_different_join(self):
        from reelforge.join import join_clips
        dest = Path(self.tmp) / "j3.mp4"
        join_clips([self.a, self.b], dest)
        messages = []
        join_clips([self.b, self.a], dest, on_status=messages.append)
        self.assertTrue(any("joining" in m for m in messages), messages)


try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False


@unittest.skipUnless(HAVE_WEB and HAVE_FFMPEG, "the web app needs fastapi and ffmpeg")
class WebAppTests(unittest.TestCase):
    """The server version: uploads, a queue, and a password."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="reelforge-web-")
        cls.clip_a = make_clip(Path(cls.tmp) / "a.mp4", duration=4)
        cls.clip_b = make_clip(Path(cls.tmp) / "b.mp4", duration=4)
        os.environ["REELFORGE_ASR_BACKEND"] = "stub"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("REELFORGE_ASR_BACKEND", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def client(self, name="c"):
        from reelforge.web import create_app
        app = create_app(data_dir=Path(self.tmp) / name, password="letmein",
                         secret="fixed-secret")
        return TestClient(app)

    def login(self, client):
        self.assertEqual(client.post("/api/login", json={"password": "letmein"}).status_code, 200)

    # -- security --------------------------------------------------------
    def test_everything_needs_a_password(self):
        client = self.client("sec")
        for path in ("/api/jobs", "/api/templates"):
            self.assertEqual(client.get(path).status_code, 401, path)
        self.assertEqual(client.post("/api/jobs").status_code, 401)

    def test_a_wrong_password_is_refused(self):
        client = self.client("sec2")
        self.assertEqual(client.post("/api/login", json={"password": "guess"}).status_code, 401)
        self.assertEqual(client.get("/api/jobs").status_code, 401)

    def test_logging_out_revokes_access(self):
        client = self.client("sec3")
        self.login(client)
        self.assertEqual(client.get("/api/jobs").status_code, 200)
        client.post("/api/logout")
        self.assertEqual(client.get("/api/jobs").status_code, 401)

    def test_session_tokens_are_signed(self):
        from reelforge.web import make_token, valid_token
        token = make_token("secret")
        self.assertTrue(valid_token("secret", token))
        self.assertFalse(valid_token("secret", token[:-1] + "0"))    # tampered
        self.assertFalse(valid_token("other", token))                # forged
        self.assertFalse(valid_token("secret", None))

    # -- the job flow ----------------------------------------------------
    def upload(self, client, paths, **data):
        files = [("files", (Path(p).name, open(p, "rb"), "video/mp4")) for p in paths]
        return client.post("/api/jobs", files=files,
                           data={"template": "", "model": "small", **data})

    def wait(self, client, job_id, limit=240):
        for _ in range(limit):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("ready", "error"):
                return job
            time.sleep(1)
        self.fail("job never finished")

    def test_upload_two_clips_and_export(self):
        client = self.client("flow")
        self.login(client)

        response = self.upload(client, [self.clip_a, self.clip_b])
        self.assertEqual(response.status_code, 200)
        job = self.wait(client, response.json()["id"])
        self.assertEqual(job["status"], "ready", job.get("error"))
        # Two 4s clips joined, then trimmed.
        self.assertGreater(job["summary"]["source_duration"], 7.0)

        edl = client.get(f"/api/jobs/{job['id']}/edl").json()
        self.assertTrue(edl["captions"])

        # Range requests, so the preview can be scrubbed on a phone.
        preview = client.get(f"/api/jobs/{job['id']}/preview.mp4",
                             headers={"Range": "bytes=0-1023"})
        self.assertEqual(preview.status_code, 206)
        self.assertIn("bytes 0-1023/", preview.headers.get("content-range", ""))

        edit = client.post(f"/api/jobs/{job['id']}/edl", json={
            "rerender": False, "captions": [{"text": "نص جديد"}],
            "zooms": [{"enabled": False}], "overlays": [], "transitions": []})
        self.assertEqual(edit.status_code, 200)

        exported = client.post(f"/api/jobs/{job['id']}/export")
        self.assertEqual(exported.status_code, 200)
        download = client.get(f"/api/jobs/{job['id']}/download")
        self.assertEqual(download.status_code, 200)
        self.assertGreater(len(download.content), 10_000)

    def test_a_non_video_upload_is_rejected(self):
        client = self.client("bad")
        self.login(client)
        notes = Path(self.tmp) / "notes.txt"
        notes.write_text("not a video", encoding="utf-8")
        response = client.post("/api/jobs",
                               files=[("files", ("notes.txt", open(notes, "rb"), "text/plain"))],
                               data={"template": "", "model": "small"})
        self.assertEqual(response.status_code, 400)

    def test_unknown_job_is_a_404(self):
        client = self.client("missing")
        self.login(client)
        self.assertEqual(client.get("/api/jobs/nope").status_code, 404)
        self.assertEqual(client.get("/api/jobs/nope/edl").status_code, 404)

    def test_jobs_survive_a_restart_and_are_marked_interrupted(self):
        from reelforge.web import Job, JobStore
        root = Path(self.tmp) / "restart"
        store = JobStore(root)
        job = store.create("clip", "", "small")
        store.update(job, status="working", stage="rendering")
        # A new store is what a process restart looks like.
        reopened = JobStore(root)
        recovered = reopened.get(job.id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, "error")
        self.assertIn("restart", recovered.error)

"""Test suite. Uses only the standard library plus ffmpeg.

The media tests build their own tiny clip with ffmpeg's synthetic sources, so the
suite runs anywhere ffmpeg is installed and needs no sample files or models.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
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

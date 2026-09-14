"""The part that compounds.

Nothing here fine-tunes a video model - that is not how this gets fast. What
actually improves, run after run, is:

1. **Vocabulary.** Every caption word you fix is stored as a correction and fed
   back as both an ASR bias prompt and a post-pass replacement. Names, brands and
   dialect words stop being wrong after you fix them once.
2. **Pacing parameters.** Disable half the punch-ins and the threshold rises; keep
   them all and it falls. A handful of scalars converge on your taste in a few edits.
3. **A zoom model.** Once there are enough labelled examples, a small logistic
   regression trained on your own accept/reject decisions replaces the hand-written
   emphasis score. Seven features, pure Python, trains in milliseconds.
4. **B-roll preferences.** Assets you keep for a keyword get promoted; ones you
   delete sink below the match threshold.

All of it is local SQLite you can inspect, export or delete.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .arabic import normalize_for_match
from .edl import EDL

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    source TEXT NOT NULL,
    video_key TEXT,
    profile TEXT,
    proposed_edl TEXT,
    final_edl TEXT,
    output TEXT,
    reviewed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    features TEXT,
    kept INTEGER NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vocab (
    wrong TEXT PRIMARY KEY,
    right TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    updated TEXT
);
CREATE TABLE IF NOT EXISTS keyword_weights (
    key TEXT PRIMARY KEY,
    weight REAL DEFAULT 1.0,
    count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS params (
    name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS models (
    kind TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    trained TEXT,
    samples INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decisions_kind ON decisions(kind);
"""

# Profile scalars the system is allowed to tune from your behaviour.
TUNABLE = {
    "zoom.score_threshold": (0.15, 0.9),
    "zoom.rate_per_min": (2.0, 30.0),
    "broll.max_per_min": (0.0, 15.0),
    "broll.min_score": (0.3, 0.95),
    "captions.max_words": (2.0, 8.0),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- zoom model

@dataclass
class ZoomModel:
    """Logistic regression over the emphasis features, trained on your choices."""

    FEATURES = ("rel_energy", "rel_peak", "duration", "word_rate",
                "after_cut", "position", "since_last_zoom")

    weights: list[float] = field(default_factory=lambda: [0.0] * 7)
    bias: float = 0.0
    mean: list[float] = field(default_factory=lambda: [0.0] * 7)
    std: list[float] = field(default_factory=lambda: [1.0] * 7)
    samples: int = 0
    accuracy: float = 0.0

    def vector(self, features: dict) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.FEATURES]

    def _standardize(self, row: list[float]) -> list[float]:
        return [(value - self.mean[i]) / (self.std[i] or 1.0) for i, value in enumerate(row)]

    def predict(self, features: dict) -> float:
        row = self._standardize(self.vector(features))
        z = self.bias + sum(w * x for w, x in zip(self.weights, row))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def fit(self, rows: list[tuple[dict, int]], *, lr: float = 0.08,
            epochs: int = 400, l2: float = 0.01) -> "ZoomModel":
        if not rows:
            return self
        matrix = [self.vector(f) for f, _ in rows]
        labels = [float(label) for _, label in rows]
        count = len(matrix)
        width = len(self.FEATURES)

        self.mean = [sum(r[i] for r in matrix) / count for i in range(width)]
        self.std = []
        for i in range(width):
            variance = sum((r[i] - self.mean[i]) ** 2 for r in matrix) / count
            self.std.append(math.sqrt(variance) or 1.0)
        scaled = [self._standardize(r) for r in matrix]

        self.weights = [0.0] * width
        self.bias = 0.0
        for _ in range(epochs):
            grad_w = [0.0] * width
            grad_b = 0.0
            for row, label in zip(scaled, labels):
                z = self.bias + sum(w * x for w, x in zip(self.weights, row))
                pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                error = pred - label
                for i, value in enumerate(row):
                    grad_w[i] += error * value
                grad_b += error
            for i in range(width):
                self.weights[i] -= lr * (grad_w[i] / count + l2 * self.weights[i])
            self.bias -= lr * (grad_b / count)

        correct = sum(1 for row, label in zip(scaled, labels)
                      if ((self.bias + sum(w * x for w, x in zip(self.weights, row))) > 0)
                      == (label > 0.5))
        self.samples = count
        self.accuracy = correct / count
        return self

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "bias": self.bias, "mean": self.mean,
                           "std": self.std, "samples": self.samples, "accuracy": self.accuracy})

    @classmethod
    def from_json(cls, payload: str) -> "ZoomModel":
        data = json.loads(payload)
        return cls(weights=data["weights"], bias=data["bias"], mean=data["mean"],
                   std=data["std"], samples=data.get("samples", 0),
                   accuracy=data.get("accuracy", 0.0))

    def explain(self) -> list[tuple[str, float]]:
        """Which signals drive your zoom choices, strongest first."""
        return sorted(zip(self.FEATURES, self.weights), key=lambda p: abs(p[1]), reverse=True)


# ------------------------------------------------------------------- store

class FeedbackStore:
    """Local SQLite record of what was proposed and what you kept."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "history.db"
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- runs ------------------------------------------------------------
    def record_run(self, source: str, profile, edl: EDL, *, video_key: str = "",
                   output: str = "") -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO runs (created, source, video_key, profile, proposed_edl, output) "
                "VALUES (?,?,?,?,?,?)",
                (_now(), str(source), video_key,
                 json.dumps(profile.to_dict(), ensure_ascii=False),
                 json.dumps(edl.to_dict(), ensure_ascii=False), str(output)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_run(self, run_id: int) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def latest_run(self) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, created, source, output, reviewed FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- feedback --------------------------------------------------------
    def record_feedback(self, run_id: int, final: EDL, *, ema: float = 0.25) -> dict:
        """Diff the edit you kept against the edit that was proposed, and learn."""
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"no run {run_id}")
        proposed = EDL.from_dict(json.loads(run["proposed_edl"]))

        learned = {"zooms_kept": 0, "zooms_dropped": 0, "overlays_kept": 0,
                   "overlays_dropped": 0, "vocab_added": 0, "params": {}}

        final_zooms = {z.id: z for z in final.zooms}
        final_overlays = {o.id: o for o in final.overlays}

        with closing(self._connect()) as conn:
            for zoom in proposed.zooms:
                match = final_zooms.get(zoom.id)
                kept = 1 if (match is not None and match.enabled) else 0
                learned["zooms_kept" if kept else "zooms_dropped"] += 1
                conn.execute(
                    "INSERT INTO decisions (run_id, kind, ref_id, features, kept, created) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, "zoom", zoom.id, json.dumps(zoom.features), kept, _now()),
                )

            for overlay in proposed.overlays:
                match = final_overlays.get(overlay.id)
                kept = 1 if (match is not None and match.enabled) else 0
                learned["overlays_kept" if kept else "overlays_dropped"] += 1
                conn.execute(
                    "INSERT INTO decisions (run_id, kind, ref_id, features, kept, created) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, "overlay", overlay.id,
                     json.dumps({"keyword": overlay.keyword,
                                 "asset": Path(overlay.asset).name,
                                 "score": overlay.score}), kept, _now()),
                )
                key = f"{Path(overlay.asset).name}|{overlay.keyword}"
                delta = 0.08 if kept else -0.12
                conn.execute(
                    "INSERT INTO keyword_weights (key, weight, count) VALUES (?,?,1) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "weight=max(0.2, min(1.6, weight + ?)), count=count+1",
                    (key, max(0.2, min(1.6, 1.0 + delta)), delta),
                )

            # Caption edits are the highest-value signal: they teach vocabulary.
            for pair in _diff_caption_words(proposed, final):
                wrong, right = pair
                conn.execute(
                    "INSERT INTO vocab (wrong, right, count, updated) VALUES (?,?,1,?) "
                    "ON CONFLICT(wrong) DO UPDATE SET right=excluded.right, "
                    "count=count+1, updated=excluded.updated",
                    (wrong, right, _now()),
                )
                learned["vocab_added"] += 1

            conn.execute("UPDATE runs SET final_edl=?, reviewed=1 WHERE id=?",
                         (json.dumps(final.to_dict(), ensure_ascii=False), run_id))
            conn.commit()

        learned["params"] = self._tune_from_rates(proposed, learned, ema=ema)
        return learned

    def _tune_from_rates(self, proposed: EDL, learned: dict, *, ema: float) -> dict:
        """Nudge pacing scalars toward the rate you actually accept."""
        updates: dict[str, float] = {}
        total_zooms = learned["zooms_kept"] + learned["zooms_dropped"]
        if total_zooms >= 3:
            keep_rate = learned["zooms_kept"] / total_zooms
            current = self.get_param("zoom.score_threshold", 0.45)
            # Keeping everything means we were too shy; dropping a lot means too eager.
            target = current + (0.5 - keep_rate) * 0.35
            updates["zoom.score_threshold"] = self.set_param(
                "zoom.score_threshold", _blend(current, target, ema))

            out_minutes = max(0.25, proposed.duration / 60.0)
            accepted_rate = learned["zooms_kept"] / out_minutes
            current_rate = self.get_param("zoom.rate_per_min", 14.0)
            updates["zoom.rate_per_min"] = self.set_param(
                "zoom.rate_per_min", _blend(current_rate, accepted_rate, ema))

        total_overlays = learned["overlays_kept"] + learned["overlays_dropped"]
        if total_overlays >= 2:
            out_minutes = max(0.25, proposed.duration / 60.0)
            accepted_rate = learned["overlays_kept"] / out_minutes
            current_rate = self.get_param("broll.max_per_min", 6.0)
            updates["broll.max_per_min"] = self.set_param(
                "broll.max_per_min", _blend(current_rate, accepted_rate, ema))
        return updates

    # -- params ----------------------------------------------------------
    def get_param(self, name: str, default: float) -> float:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM params WHERE name=?", (name,)).fetchone()
            return float(row["value"]) if row else float(default)

    def set_param(self, name: str, value: float) -> float:
        low, high = TUNABLE.get(name, (-1e9, 1e9))
        value = max(low, min(high, float(value)))
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO params (name, value, count) VALUES (?,?,1) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, count=count+1",
                (name, value),
            )
            conn.commit()
        return value

    def params(self) -> dict[str, float]:
        with closing(self._connect()) as conn:
            return {r["name"]: r["value"] for r in conn.execute("SELECT name, value FROM params")}

    def tuned_profile(self, profile):
        """Overlay everything learned so far on top of a base profile."""
        tuned = profile
        for name, value in self.params().items():
            if name in TUNABLE:
                if name == "captions.max_words":
                    value = int(round(value))
                tuned = tuned.merged(_nest(name, value))
        return tuned

    # -- vocabulary / weights --------------------------------------------
    def vocab_pairs(self) -> dict[str, str]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT wrong, right FROM vocab ORDER BY count DESC").fetchall()
            return {r["wrong"]: r["right"] for r in rows}

    def add_vocab(self, wrong: str, right: str) -> None:
        key = normalize_for_match(wrong)
        if not key or not right.strip():
            return
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO vocab (wrong, right, count, updated) VALUES (?,?,1,?) "
                "ON CONFLICT(wrong) DO UPDATE SET right=excluded.right, count=count+1, "
                "updated=excluded.updated",
                (key, right.strip(), _now()),
            )
            conn.commit()

    def keyword_weights(self) -> dict[str, float]:
        with closing(self._connect()) as conn:
            return {r["key"]: r["weight"]
                    for r in conn.execute("SELECT key, weight FROM keyword_weights")}

    # -- model -----------------------------------------------------------
    def training_rows(self, kind: str = "zoom") -> list[tuple[dict, int]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT features, kept FROM decisions WHERE kind=?", (kind,)
            ).fetchall()
        out: list[tuple[dict, int]] = []
        for row in rows:
            try:
                features = json.loads(row["features"] or "{}")
            except json.JSONDecodeError:
                continue
            if features:
                out.append((features, int(row["kept"])))
        return out

    def train_zoom_model(self, *, lr: float = 0.08, min_samples: int = 40) -> ZoomModel | None:
        rows = self.training_rows("zoom")
        labels = {label for _, label in rows}
        # A model needs both accepted and rejected examples to learn anything.
        if len(rows) < min_samples or len(labels) < 2:
            return None
        model = ZoomModel().fit(rows, lr=lr)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO models (kind, payload, trained, samples) VALUES ('zoom',?,?,?) "
                "ON CONFLICT(kind) DO UPDATE SET payload=excluded.payload, "
                "trained=excluded.trained, samples=excluded.samples",
                (model.to_json(), _now(), model.samples),
            )
            conn.commit()
        return model

    def load_zoom_model(self, *, min_samples: int = 40) -> ZoomModel | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload, samples FROM models WHERE kind='zoom'").fetchone()
        if not row or int(row["samples"] or 0) < min_samples:
            return None
        try:
            return ZoomModel.from_json(row["payload"])
        except (json.JSONDecodeError, KeyError):
            return None

    def scorer(self, *, min_samples: int = 40):
        """A callable emphasis scorer, or None to keep using the rules."""
        model = self.load_zoom_model(min_samples=min_samples)
        return model.predict if model else None

    # -- reporting -------------------------------------------------------
    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            runs = conn.execute("SELECT COUNT(*) c, SUM(reviewed) r FROM runs").fetchone()
            decisions = conn.execute(
                "SELECT kind, COUNT(*) total, SUM(kept) kept FROM decisions GROUP BY kind"
            ).fetchall()
            vocab = conn.execute("SELECT COUNT(*) c FROM vocab").fetchone()
            model = conn.execute("SELECT samples, trained, payload FROM models "
                                 "WHERE kind='zoom'").fetchone()
        accuracy = None
        if model:
            try:
                accuracy = ZoomModel.from_json(model["payload"]).accuracy
            except (json.JSONDecodeError, KeyError):
                accuracy = None
        return {
            "runs": int(runs["c"] or 0),
            "reviewed": int(runs["r"] or 0),
            "decisions": {r["kind"]: {"total": r["total"], "kept": r["kept"] or 0}
                          for r in decisions},
            "vocab_terms": int(vocab["c"] or 0),
            "zoom_model": ({"samples": model["samples"], "trained": model["trained"],
                            "accuracy": accuracy} if model else None),
            "params": self.params(),
        }


# ---------------------------------------------------------------- helpers

def _blend(current: float, target: float, ema: float) -> float:
    return (1.0 - ema) * current + ema * target


def _nest(dotted: str, value) -> dict:
    parts = dotted.split(".")
    node: dict = {}
    cursor = node
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node


def _diff_caption_words(proposed: EDL, final: EDL) -> list[tuple[str, str]]:
    """Word-level corrections between the proposed captions and the ones you kept."""
    pairs: list[tuple[str, str]] = []
    for before, after in zip(proposed.captions, final.captions):
        old_words = [w.text for w in before.words]
        new_words = [w.text for w in after.words]
        if old_words == new_words:
            continue
        if len(old_words) == len(new_words):
            for old, new in zip(old_words, new_words):
                # Any change to the displayed word is worth learning, including a
                # pure spelling fix like اسامه -> أسامة: the map is keyed on the
                # normalised form, so it teaches the right *display* spelling.
                if old.strip() != new.strip():
                    key = normalize_for_match(old)
                    if key and new.strip():
                        pairs.append((key, new.strip()))
        else:
            # Length changed - only learn from words that vanished one-for-one.
            missing = [w for w in old_words if w not in new_words]
            added = [w for w in new_words if w not in old_words]
            if len(missing) == len(added):
                for old, new in zip(missing, added):
                    key = normalize_for_match(old)
                    if key and new.strip():
                        pairs.append((key, new.strip()))
    return pairs

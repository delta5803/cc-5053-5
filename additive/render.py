"""Offline rendering: schedule notes on a timeline and bounce to audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import TimbreConfig
from .engine import AdditiveEngine


@dataclass
class NoteEvent:
    start: float                 # seconds from render start
    duration: float              # seconds held before note-off
    freq_multiplier: float = 1.0
    note_gain: float = 1.0


def render_timeline(
    config: TimbreConfig,
    events: List[NoteEvent],
    total_duration: Optional[float] = None,
    block_size: int = 256,
) -> np.ndarray:
    """Render a monophonic/multitrack timeline to a mono float64 array.

    Every event starts its own note instance, so overlapping events produce
    genuinely independent voices, each with its own phase and envelope state.
    """
    if not events:
        raise ValueError("at least one note event is required")
    if total_duration is None:
        total_duration = max(e.start + e.duration for e in events) + float(
            max(p.release for p in config.partials)
        ) + 0.05
    total_samples = int(round(total_duration * config.sample_rate))
    engine = AdditiveEngine(config)
    sr = config.sample_rate

    order = sorted(range(len(events)), key=lambda i: events[i].start)
    cursor = 0
    chunks: List[np.ndarray] = []

    def advance(target_sample: int) -> None:
        nonlocal cursor
        while cursor < target_sample:
            size = min(block_size, target_sample - cursor)
            chunks.append(engine.render_block(size))
            cursor += size

    for idx in order:
        ev = events[idx]
        start_sample = int(round(ev.start * sr))
        advance(start_sample)
        note_id = engine.note_on(ev.freq_multiplier, ev.note_gain)
        off_sample = int(round((ev.start + ev.duration) * sr))
        advance(off_sample)
        engine.note_off(note_id)

    advance(total_samples)
    return np.concatenate(chunks)[:total_samples]


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Peak normalize; returns a copy. Silences pass through untouched."""
    peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_abs <= 1e-12:
        return audio.copy()
    return audio * (peak / peak_abs)

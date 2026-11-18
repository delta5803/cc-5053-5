"""Additive harmonic timbre reconstruction instrument."""

from .config import PartialSpec, TimbreConfig
from .engine import AdditiveEngine, BandlimitReport, NoteVoice
from .render import NoteEvent, normalize, render_timeline
from .audio_io import export_wav

__all__ = [
    "PartialSpec",
    "TimbreConfig",
    "AdditiveEngine",
    "NoteVoice",
    "BandlimitReport",
    "NoteEvent",
    "render_timeline",
    "normalize",
    "export_wav",
]

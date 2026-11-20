"""One performance track with two interchangeable synthesis backends.

A :class:`DualBackendEngine` plays the same note through both:

* the time-domain oscillator bank (:class:`~additive.engine.NoteVoice`,
  ``"td"``);
* the chunked frequency-domain bank
  (:class:`~additive.fd_engine.FDNoteVoice`, ``"fd"``).

Both backends advance their own oscillator state sample by sample from the
same :class:`~additive.config.TimbreConfig`. The FD analysis frame is the
current oscillator hop plus the next one (a 2*hop, 50 %-overlap Hann frame),
so FD output hop b is built from oscillator hops b and b+1. To align the two
paths, the dual voice delays the TD output by one hop the same way: both
backends emit output hop b while rendering oscillator hop b+1.

Phase handoff timing. Because output hop b reflects oscillator state up to
hop b+1, changing the target backend's oscillator phase *before* it renders
oscillator hop s (a hop boundary) makes every target output sample from s on
use the new phase while earlier samples keep the old one. No retroactive
buffer editing or Hilbert transform is needed. At the switch the source and
target per-partial phases are identical, so the waveform value and slope are
continuous across the cut. Switches are quantized to hop boundaries, which is
also where a block-based backend can hand over cleanly.

Three handoff strategies:

``hard``
    Instant cut; the target keeps its own evolved phase. A phase discontinuity
    (click) is expected -- the deliberately naive baseline.

``handoff``
    The target's per-partial phase is aligned to the source's at the switch
    (a two-way state transfer). One backend owns every partial at every
    sample, so nothing is double counted or dropped.

``parallel``
    Bounded parallel transition over a fixed crossfade window with per-partial
    weights ``w_td + w_fd = 1`` at every sample, so each partial is counted
    exactly once. The window is bounded; outside it one backend alone owns
    the output.

Every switch records a :class:`SwitchAudit` of per-sample ownership weights and
the phases before/after, used to prove no partial is duplicated or lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

from .config import TimbreConfig
from .engine import NoteVoice
from .fd_engine import FDNoteVoice

Strategy = Literal["hard", "handoff", "parallel"]


@dataclass
class SwitchEvent:
    """A backend switch scheduled on one note's timeline.

    ``sample_index`` is rounded to the FD hop boundary; clean block handover
    is only defined there.
    """

    sample_index: int
    strategy: Strategy = "handoff"
    crossfade_samples: int = 0
    # Per-partial phase offset (radians) applied to the target at the switch
    # sample (the handoff's crossing-point phase edit).
    phase_perturbation: Optional[np.ndarray] = None
    # Per-partial phase offset (radians) applied to BOTH backends at the
    # switch sample: a musical phase event independent of which backend owns
    # the output. Present in every condition including the no-switch reference.
    common_phase_perturbation: Optional[np.ndarray] = None
    # When False the event carries parameter changes (e.g. the common phase
    # edit) but performs no backend handoff -- the no-switch reference path.
    active: bool = True


@dataclass
class SwitchAudit:
    """What each backend owned around a switch."""

    switch_sample: int
    strategy: Strategy
    td_weight: np.ndarray
    fd_weight: np.ndarray
    phase_before: np.ndarray
    phase_after: np.ndarray
    crossfade_samples: int = 0


class DualVoice:
    """One note rendered by two backends with a switchable output path."""

    def __init__(self, note_id: int, config: TimbreConfig, freq_multiplier: float,
                 note_gain: float, start_backend: str, fd_phase_offset: float = 0.0) -> None:
        if start_backend not in ("td", "fd"):
            raise ValueError("start_backend must be 'td' or 'fd'")
        self.note_id = note_id
        self.config = config
        self.hop = int(config.fd_hop)
        self.td = NoteVoice(note_id, config, freq_multiplier)
        self.fd = FDNoteVoice(note_id, config, freq_multiplier, phase_offset=fd_phase_offset)
        for voice in (self.td, self.fd):
            voice.gain = voice.target_gain.copy()
            voice.set_note_gain(note_gain)
            voice.note_gain = note_gain
        self.n_partials = len(config.partials)
        self.active = start_backend
        self.start_backend = start_backend
        self.osc_index = 0             # oscillator samples rendered so far
        self.switches: List[SwitchEvent] = []
        self.audits: List[SwitchAudit] = []
        self.alive = True
        # TD output is delayed one hop to match the FD look-ahead latency.
        self._td_buf: Optional[np.ndarray] = None
        self._td_note: Optional[np.ndarray] = None
        # Parallel crossfade runtime, indexed on output samples.
        self._xfade_remaining = 0
        self._xfade_total = 0
        self._xfade_source = start_backend

    # -- scheduling -----------------------------------------------------------

    def schedule_switch(self, event: SwitchEvent) -> None:
        if event.strategy not in ("hard", "handoff", "parallel"):
            raise ValueError(f"unknown switch strategy {event.strategy!r}")
        if event.strategy == "parallel" and event.crossfade_samples <= 0:
            raise ValueError("parallel strategy needs crossfade_samples > 0")
        # Quantize to a hop boundary; remember the requested point for the UI.
        event.sample_index = int(round(event.sample_index / self.hop)) * self.hop
        if event.crossfade_samples:
            event.crossfade_samples = int(round(event.crossfade_samples / self.hop)) * self.hop
        self.switches.append(event)
        self.switches.sort(key=lambda e: e.sample_index)

    def set_solo_mask(self, solo_scale: np.ndarray) -> None:
        self.td.set_solo_mask(solo_scale)
        self.fd.set_solo_mask(solo_scale)

    def note_off(self) -> None:
        self.td.note_off()
        self.fd.note_off()

    # -- phase handoff --------------------------------------------------------

    def _align_target_phase(self, perturbation: Optional[np.ndarray]):
        """Per-partial offset aligning the target's phase to the source's."""
        if self.active == "td":
            target, source = self.fd, self.td
        else:
            target, source = self.td, self.fd
        before = target.phase.copy()
        delta = (source.phase - target.phase + math.pi) % (2 * math.pi) - math.pi
        if perturbation is not None:
            delta = delta + np.asarray(perturbation, dtype=np.float64)
        target.phase = (target.phase + delta) % (2 * math.pi)
        return before, target.phase.copy()

    # -- rendering ------------------------------------------------------------

    def render_block(self, block_size: int) -> np.ndarray:
        if block_size != self.hop:
            raise ValueError(f"dual voice renders hop-sized blocks ({self.hop}), got {block_size}")
        hop = self.hop

        # Output hop b (samples b..b+hop-1) is produced from oscillator hop
        # b (FD look-ahead) / the TD hop buffered one block earlier. A switch
        # at output sample s therefore needs the target's phase changed before
        # oscillator hop s-hop is rendered. Apply those handoffs now.
        trigger_osc = self.osc_index
        for ev in self.switches:
            if ev.sample_index - hop != trigger_osc:
                continue
            if ev.common_phase_perturbation is not None:
                common = np.asarray(ev.common_phase_perturbation, dtype=np.float64)
                self.td.phase = (self.td.phase + common) % (2 * math.pi)
                self.fd.phase = (self.fd.phase + common) % (2 * math.pi)
            if ev.active and ev.strategy in ("handoff", "parallel"):
                self._align_target_phase(ev.phase_perturbation)

        td_parts, td_note = self.td.render_block_parts(hop)
        fd_parts, fd_note = self.fd.render_block_parts(hop)
        self.osc_index += hop

        # FD emits the hop directly (its internal look-ahead); TD is delayed.
        if self._td_buf is None:
            self._td_buf, self._td_note = td_parts, td_note
            return np.zeros(hop, dtype=np.float64)

        out_start = self.osc_index - 2 * hop   # output sample index of this hop
        n = self.n_partials
        td_aligned, fd_aligned = self._td_buf, fd_parts
        td_n_aligned, fd_n_aligned = self._td_note, fd_note
        td_w = np.zeros((hop, n), dtype=np.float64)
        fd_w = np.zeros((hop, n), dtype=np.float64)
        phase_before = phase_after = np.zeros(n, dtype=np.float64)
        pending = [e for e in self.switches if e.sample_index == out_start]

        for ev in pending:
            if not ev.active:
                continue
            if ev.strategy == "hard":
                tgt = self.fd if self.active == "td" else self.td
                src = self.td if self.active == "td" else self.fd
                phase_before, phase_after = tgt.phase.copy(), src.phase.copy()
                self.active = "fd" if self.active == "td" else "td"
            else:
                # Phase was already aligned at the oscillator trigger above;
                # the audit phases were captured there. Re-capture current.
                phase_before, phase_after = self._audit_phases(ev)
                if ev.strategy == "handoff":
                    self.active = "fd" if self.active == "td" else "td"
                else:
                    self._xfade_source = self.active
                    self._xfade_total = ev.crossfade_samples
                    self._xfade_remaining = ev.crossfade_samples

        for k in range(hop):
            cur_out = out_start + k
            if self._xfade_remaining > 0 and cur_out >= out_start:
                done = self._xfade_total - self._xfade_remaining
                frac = (done + 1) / self._xfade_total
                if self._xfade_source == "td":
                    w_td, w_fd = 1.0 - frac, frac
                else:
                    w_td, w_fd = frac, 1.0 - frac
                self._xfade_remaining -= 1
                if self._xfade_remaining == 0:
                    self.active = "fd" if self._xfade_source == "td" else "td"
            else:
                w_td, w_fd = (1.0, 0.0) if self.active == "td" else (0.0, 1.0)
            td_w[k], fd_w[k] = w_td, w_fd

        for ev in pending:
            if not ev.active:
                continue
            self.audits.append(
                SwitchAudit(
                    switch_sample=ev.sample_index,
                    strategy=ev.strategy,
                    td_weight=td_w.copy(),
                    fd_weight=fd_w.copy(),
                    phase_before=phase_before,
                    phase_after=phase_after,
                    crossfade_samples=ev.crossfade_samples,
                )
            )

        out = (
            td_n_aligned[:, None] * td_w * td_aligned
            + fd_n_aligned[:, None] * fd_w * fd_aligned
        ).sum(axis=1)

        self._td_buf, self._td_note = td_parts, td_note
        self.alive = self.td.alive and self.fd.alive
        return out

    def _audit_phases(self, ev: SwitchEvent):
        tgt = self.fd if self.active == "td" else self.td
        src = self.td if self.active == "td" else self.fd
        # After alignment target.phase == source.phase (+ perturbation); the
        # "before" value is reconstructed by subtracting the perturbation.
        after = tgt.phase.copy()
        if ev.phase_perturbation is not None:
            before = (after - np.asarray(ev.phase_perturbation, dtype=np.float64)) % (2 * math.pi)
        else:
            before = src.phase.copy()
        return before, after

    def flush(self) -> np.ndarray:
        """Emit the final TD-delayed hop; FD has no extra tail after alignment."""
        if self._td_buf is None:
            return np.zeros(0, dtype=np.float64)
        hop, n = self.hop, self.n_partials
        td_w = np.zeros((hop, n), dtype=np.float64)
        fd_w = np.zeros((hop, n), dtype=np.float64)
        for k in range(hop):
            if self._xfade_remaining > 0:
                done = self._xfade_total - self._xfade_remaining
                frac = (done + 1) / self._xfade_total
                if self._xfade_source == "td":
                    w_td, w_fd = 1.0 - frac, frac
                else:
                    w_td, w_fd = frac, 1.0 - frac
                self._xfade_remaining -= 1
                if self._xfade_remaining == 0:
                    self.active = "fd" if self._xfade_source == "td" else "td"
            else:
                w_td, w_fd = (1.0, 0.0) if self.active == "td" else (0.0, 1.0)
            td_w[k], fd_w[k] = w_td, w_fd
        # FD already emitted its last aligned hop; only the TD tail remains.
        out = (self._td_note[:, None] * td_w * self._td_buf).sum(axis=1)
        self._td_buf = None
        return out


class DualBackendEngine:
    """Polyphonic instrument routing every note through both backends."""

    def __init__(self, config: TimbreConfig, start_backend: str = "td") -> None:
        self.config = config
        self.start_backend = start_backend
        self.voices: Dict[int, DualVoice] = {}
        self._next_id = 0
        self.master_gain = float(config.master_amplitude)
        self.target_master_gain = self.master_gain
        self.solo_scale = np.ones(len(config.partials), dtype=np.float64)
        self.block_counter = 0

    def note_on(self, freq_multiplier: float = 1.0, note_gain: float = 1.0,
                fd_phase_offset: float = 0.0) -> int:
        note_id = self._next_id
        self._next_id += 1
        voice = DualVoice(note_id, self.config, freq_multiplier, note_gain,
                          self.start_backend, fd_phase_offset=fd_phase_offset)
        voice.set_solo_mask(self.solo_scale)
        self.voices[note_id] = voice
        return note_id

    def note_off(self, note_id: int) -> None:
        voice = self.voices.get(note_id)
        if voice is not None:
            voice.note_off()

    def all_notes_off(self) -> None:
        for voice in self.voices.values():
            voice.note_off()

    def schedule_switch(self, note_id: int, event: SwitchEvent) -> None:
        voice = self.voices.get(note_id)
        if voice is None:
            raise KeyError(f"no note {note_id}")
        voice.schedule_switch(event)

    def set_master_gain(self, value: float) -> None:
        self.target_master_gain = max(0.0, float(value))

    def set_partial_amplitudes_on_voice(self, note_id: int, amplitudes: np.ndarray) -> None:
        """Update one note's per-partial gains on both backends in lockstep."""
        voice = self.voices.get(note_id)
        if voice is None:
            raise KeyError(f"no note {note_id}")
        voice.td.set_partial_amplitudes(amplitudes)
        voice.fd.set_partial_amplitudes(amplitudes)

    def solo_partial(self, index: Optional[int]) -> None:
        n = len(self.config.partials)
        self.solo_scale = np.zeros(n, dtype=np.float64)
        if index is None:
            self.solo_scale[:] = 1.0
        else:
            self.solo_scale[index] = 1.0
        for voice in self.voices.values():
            voice.set_solo_mask(self.solo_scale)

    def render_block(self, block_size: int) -> np.ndarray:
        out = np.zeros(block_size, dtype=np.float64)
        dead: List[int] = []
        for note_id, voice in self.voices.items():
            out += voice.render_block(block_size)
            if not voice.alive:
                dead.append(note_id)
        for note_id in dead:
            del self.voices[note_id]
        self.master_gain = self.target_master_gain
        self.block_counter += 1
        return out * self.master_gain

    def render_samples(self, sample_count: int, block_size: int = 256) -> np.ndarray:
        """Render an exact count; the leading silent alignment hop is dropped."""
        hop = int(self.config.fd_hop)
        n_blocks = int(math.ceil(sample_count / hop)) + 1
        chunks: List[np.ndarray] = []
        for _ in range(n_blocks):
            chunks.append(self.render_block(hop))
        for voice in list(self.voices.values()):
            tail = voice.flush()
            if tail.size:
                chunks.append(tail * self.master_gain)
        out = np.concatenate(chunks)
        return out[hop : hop + sample_count]

    def audits(self) -> List[SwitchAudit]:
        out: List[SwitchAudit] = []
        for voice in self.voices.values():
            out.extend(voice.audits)
        return out

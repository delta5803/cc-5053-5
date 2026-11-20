"""Chunked frequency-domain synthesis backend.

The same instrument can be rendered two ways. The time-domain backend
(:mod:`additive.engine`) sums ``gain * envelope * sin(phase)`` directly, one
oscillator per partial. This backend keeps *exactly* the same per-partial
oscillator state -- phase accumulation, envelopes, gain ramps all live in the
shared :class:`~additive.engine.NoteVoice` -- but replaces the final summation
with a genuinely frequency-domain path:

1. rendered hops are buffered; an analysis frame is the *current* hop and the
   *next* hop (a 2*hop frame, 50 % overlap), Hann windowed;
2. each partial's frame is transformed with an RFFT; its bins are accumulated,
   so the bookkeeping of *which backend owns which partial* is observable in
   the frequency domain (see :class:`FDBinContribution`);
3. each partial's spectrum is inverted (IRFFT) and overlap-added; periodic
   Hann at 50 % overlap satisfies the COLA condition
   ``w(n) + w(n - hop) = 1``, so the windowed FFT path reconstructs the same
   continuous-time signal as the direct time-domain sum (interior error is
   floating-point round-off).

``render_block_parts`` returns the per-partial matrix after the STFT/OLA path,
shaped exactly like the time-domain matrix, so the dual-backend engine can
weight and sum the two backends' partials interchangeably. The look-ahead frame
adds one hop of latency; :meth:`FDEngine.render_samples` renders an extra hop,
flushes the tail and removes the leading silent hop to align with the TD path.
This path never touches ``NoteVoice.phase``; it only consumes the per-partial
sample matrix the shared oscillator machinery produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .config import TimbreConfig
from .engine import AdditiveEngine, NoteVoice


@dataclass
class FDBinContribution:
    """Per-partial spectral bookkeeping for one rendered frame.

    ``bin_index``/``bin_freq`` locate where one partial's energy landed in the
    frame spectrum; this is what lets an audit decide whether a partial was
    counted twice (both backends emitting bins at the same frequency) or
    dropped (neither backend contributing bins).
    """

    frame_index: int
    partial_index: int
    bin_index: int
    bin_freq: float
    magnitude: float
    phase: float


class FDNoteVoice(NoteVoice):
    """A note voice whose per-partial signals go through windowed FFT/IRFFT."""

    def __init__(self, note_id: int, config: TimbreConfig, freq_multiplier: float = 1.0,
                 phase_offset: float = 0.0) -> None:
        super().__init__(note_id, config, freq_multiplier)
        # An independent per-voice phase origin. It defaults to zero (the FD
        # bank then matches the TD bank exactly), but the dual-backend
        # experiment can give the two backends different origins to emulate
        # two independently evolved phase states and stress handoff.
        if phase_offset:
            self.phase = (self.phase + phase_offset) % (2.0 * math.pi)
            self.initial_phases = self.phase.copy()
        self.hop = int(config.fd_hop)
        self.frame_size = 2 * self.hop
        n = self.frame_size
        # Periodic Hann window; COLA at 50 % overlap: w(n) + w(n-hop) = 1.
        self.window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n) / n)
        # Oscillator hop waiting for the next block to complete its frame.
        self._pending: np.ndarray | None = None
        # Per-partial second half of the previous frame (overlap-add carry).
        self._carry = np.zeros((self.hop, self.n), dtype=np.float64)
        self.frame_counter = 0
        # Optional per-partial spectral audit trail (off by default for speed).
        self.record_bins = False
        self.bin_log: List[FDBinContribution] = []

    def _frame_parts(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Window, RFFT, IRFFT one 2*hop frame for every partial separately."""
        n = self.frame_size
        windowed = np.concatenate([first, second], axis=0) * self.window[:, None]
        out = np.zeros((n, self.n), dtype=np.float64)
        for j in range(self.n):
            cols = np.fft.rfft(windowed[:, j])
            out[:, j] = np.fft.irfft(cols, n=n)
            if self.record_bins:
                mags = np.abs(cols)
                k = int(np.argmax(mags))
                if mags[k] > 1e-12:
                    self.bin_log.append(
                        FDBinContribution(
                            frame_index=self.frame_counter,
                            partial_index=j,
                            bin_index=k,
                            bin_freq=k * self.sample_rate / n,
                            magnitude=float(mags[k]),
                            phase=float(np.angle(cols[k])),
                        )
                    )
        self.frame_counter += 1
        return out

    def render_block_parts(self, block_size: int):
        if block_size != self.hop:
            raise ValueError(
                f"FD backend requires fixed hop blocks of {self.hop}; got {block_size}"
            )
        raw, note_k = super().render_block_parts(block_size)
        if self._pending is None:
            # First hop: one hop of look-ahead latency, emit silence.
            self._pending = raw.copy()
            return np.zeros_like(raw), note_k

        frame_out = self._frame_parts(self._pending, raw)
        self._pending = raw.copy()
        head = frame_out[: self.hop] + self._carry
        self._carry = frame_out[self.hop:]
        return head, note_k

    def flush_parts(self) -> np.ndarray:
        """Emit the buffered hop (zero-padded look-ahead), per partial."""
        hop = self.hop
        if self._pending is None:
            return np.zeros((0, self.n), dtype=np.float64)
        frame_out = self._frame_parts(
            self._pending, np.zeros((hop, self.n), dtype=np.float64)
        )
        self._pending = None
        return frame_out[:hop] + self._carry


class FDEngine(AdditiveEngine):
    """Polyphonic instrument whose every voice uses the FD backend."""

    def note_on(self, freq_multiplier: float = 1.0, note_gain: float = 1.0) -> int:
        note_id = self._next_id
        self._next_id += 1
        voice = FDNoteVoice(note_id, self.config, freq_multiplier)
        voice.set_solo_mask(self.solo_scale)
        voice.gain = voice.target_gain.copy()
        voice.set_note_gain(note_gain)
        voice.note_gain = note_gain
        self.voices[note_id] = voice
        return note_id

    def render_block(self, block_size: int) -> np.ndarray:
        if block_size != int(self.config.fd_hop):
            raise ValueError(
                f"FDEngine renders fixed hop blocks ({self.config.fd_hop}); "
                "use render_samples for arbitrary lengths"
            )
        return super().render_block(block_size)

    def render_samples(self, sample_count: int, block_size: int = 0) -> np.ndarray:
        """Render exactly ``sample_count`` samples aligned to the TD backend.

        One extra hop covers the look-ahead latency; voice tails are flushed
        through their note gain and the master gain, and the leading silent
        hop is removed.
        """
        hop = int(self.config.fd_hop)
        n_blocks = int(math.ceil(sample_count / hop)) + 1
        chunks: List[np.ndarray] = []
        render_block = super().render_block
        for _ in range(n_blocks):
            chunks.append(render_block(hop))
        tail = np.zeros(hop, dtype=np.float64)
        for voice in list(self.voices.values()):
            if isinstance(voice, FDNoteVoice):
                tail_parts = voice.flush_parts()
                if tail_parts.size:
                    tail += voice.note_gain * tail_parts.sum(axis=1)
        chunks.append(tail * self.master_gain)
        out = np.concatenate(chunks)
        return out[hop : hop + sample_count]

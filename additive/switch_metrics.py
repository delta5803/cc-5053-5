"""Measurements on rendered audio and on the dual-backend handoff itself.

Everything here is evidence taken from the actual output samples (or from the
engine's per-partial ownership bookkeeping), never from the configuration
alone. It answers the four operational questions about a backend switch:

* **first onset** -- when energy first emerges and how long the attack takes;
* **beat frequency** -- slow amplitude modulation between neighboring
  partials, estimated from the analytic envelope;
* **switch noise** -- how much extra energy a switch injects, measured against
  a no-switch reference rendered with the same code path;
* **compute burden** -- wall-clock cost of each backend and of the parallel
  crossfade window where both run at once.

Plus a spectral-window audit that locates phase discontinuities and checks
whether any partial was double counted or dropped during the handoff.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis import _partial_envelope
from .dual_engine import SwitchAudit


# --------------------------------------------------------------------------
# First onset
# --------------------------------------------------------------------------

@dataclass
class OnsetMeasurement:
    onset_sample: int
    onset_time: float
    threshold: float
    peak_time: float
    attack_time: float          # 10 % -> 90 % of the local peak


def measure_first_onset(audio: np.ndarray, sample_rate: int,
                        threshold_db: float = -60.0) -> OnsetMeasurement:
    """Locate the first sample whose magnitude crosses an absolute threshold.

    Uses an RMS envelope over 1 ms windows so a single noisy sample cannot
    trigger a false onset.
    """
    win = max(1, sample_rate // 1000)
    rms = np.sqrt(np.convolve(audio**2, np.ones(win) / win, mode="same"))
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 0.0:
        return OnsetMeasurement(-1, math.nan, 0.0, math.nan, math.nan)
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return OnsetMeasurement(-1, math.nan, threshold, math.nan, math.nan)
    onset = int(above[0])
    peak_idx = int(np.argmax(rms))
    lo = np.flatnonzero(rms[: peak_idx + 1] >= 0.1 * peak)
    hi = np.flatnonzero(rms[: peak_idx + 1] >= 0.9 * peak)
    t10 = lo[0] / sample_rate if lo.size else math.nan
    t90 = hi[0] / sample_rate if hi.size else math.nan
    return OnsetMeasurement(
        onset_sample=onset,
        onset_time=onset / sample_rate,
        threshold=threshold,
        peak_time=peak_idx / sample_rate,
        attack_time=(t90 - t10) if not math.isnan(t10) and not math.isnan(t90) else math.nan,
    )


# --------------------------------------------------------------------------
# Beat frequency
# --------------------------------------------------------------------------

def estimate_beat_frequency(audio: np.ndarray, sample_rate: int,
                            partial_frequency: float,
                            known_frequencies: Optional[np.ndarray] = None,
                            fit_start: float = 0.05,
                            fit_end: Optional[float] = None) -> Tuple[float, float]:
    """Estimate the amplitude-modulation (beat) rate around one partial.

    The partial's analytic envelope is isolated and its dominant modulation
    frequency is taken from the envelope spectrum. Returns ``(beat_hz,
    strength)`` where strength is the spectral peak's normalized magnitude.
    A flat envelope yields beat_hz 0.
    """
    env = _partial_envelope(audio, sample_rate, partial_frequency, known_frequencies)
    n = env.size
    t = np.arange(n) / sample_rate
    if fit_end is None:
        fit_end = t[-1]
    mask = (t >= fit_start) & (t <= fit_end)
    seg = env[mask]
    if seg.size < 16:
        return 0.0, 0.0
    seg = seg - float(np.mean(seg))
    if float(np.max(np.abs(seg))) < 1e-9:
        return 0.0, 0.0
    # Detrend and window before the envelope FFT.
    window = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / sample_rate)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    strength = float(spec[k] / (np.sum(spec) + 1e-30))
    return float(freqs[k]), strength


# --------------------------------------------------------------------------
# Switch noise
# --------------------------------------------------------------------------

@dataclass
class SwitchNoise:
    switch_sample: int
    strategy: str
    excess_peak: float           # max |switched - reference| near the switch
    excess_rms: float            # RMS of the difference over the audit window
    click_energy: float          # excess squared first-difference energy
    window_seconds: float
    signal_rms: float


def measure_switch_noise(switched: np.ndarray, reference: np.ndarray,
                         sample_rate: int, switch_sample: int, strategy: str,
                         window_seconds: float = 0.05) -> SwitchNoise:
    """Extra energy introduced at a switch, against a no-switch reference."""
    half = int(window_seconds * sample_rate)
    lo = max(0, switch_sample - half)
    hi = min(switched.size, switch_sample + half)
    diff = switched[lo:hi] - reference[lo:hi]
    excess_peak = float(np.max(np.abs(diff)))
    excess_rms = float(np.sqrt(np.mean(diff**2)))
    d_sw = np.diff(switched[lo:hi])
    d_ref = np.diff(reference[lo:hi])
    click_energy = float(np.sum((d_sw - d_ref) ** 2))
    signal_rms = float(np.sqrt(np.mean(reference[lo:hi] ** 2)))
    return SwitchNoise(
        switch_sample=switch_sample,
        strategy=strategy,
        excess_peak=excess_peak,
        excess_rms=excess_rms,
        click_energy=click_energy,
        window_seconds=window_seconds,
        signal_rms=signal_rms,
    )


# --------------------------------------------------------------------------
# Phase discontinuity via short spectral windows
# --------------------------------------------------------------------------

@dataclass
class PhaseBreak:
    partial_index: int
    frequency: float
    window_center: float
    phase_jump_radians: float
    magnitude_before: float
    magnitude_after: float


def _window_phase_magnitude(seg: np.ndarray, window: np.ndarray,
                            sample_rate: int, frequency: float) -> Tuple[float, float]:
    """Phase/magnitude of one partial at its *exact* frequency in a window.

    A complex dot product against a pure tone measures phase directly at the
    partial's true frequency instead of snapping to the nearest FFT bin, so
    there is no phase drift from the coarse FFT frequency grid to mask a real
    discontinuity. The window normalisation cancels in the phase, and the
    magnitude is scaled to an equivalent sinusoid amplitude.
    """
    n = seg.size
    t = np.arange(n, dtype=np.float64) / sample_rate
    basis = np.exp(-1j * 2.0 * math.pi * frequency * t) * window
    z = complex(np.dot(seg, basis))
    coherent = float(np.sum(window))
    phase = math.atan2(z.imag, z.real)
    magnitude = 2.0 * abs(z) / coherent
    return phase, magnitude


def _window_partial_coeffs(audio: np.ndarray, lo: int, window_samples: int,
                           window: np.ndarray, sample_rate: int,
                           frequencies: np.ndarray) -> np.ndarray:
    """Joint least-squares complex coefficient of every partial in a window.

    All partials are fit simultaneously to their exact frequencies, so each
    coefficient is isolated from the others' leakage and from amplitude ramps
    (the fit uses instantaneous window samples, not a magnitude-only FFT bin).
    Returns one complex coefficient per partial; its angle is the partial's
    phase and ``2*|c|/coherent_gain`` its sinusoid amplitude.
    """
    seg = audio[lo : lo + window_samples] * window
    t = np.arange(window_samples, dtype=np.float64) / sample_rate
    cols = []
    for f in np.atleast_1d(frequencies):
        ang = 2.0 * math.pi * float(f) * t
        cols.append(np.cos(ang) * window)
        cols.append(np.sin(ang) * window)
    basis = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(basis, seg, rcond=None)
    coef = coef.reshape(-1, 2)
    return coef[:, 0] - 1j * coef[:, 1]


def spectral_phase_windows(audio: np.ndarray, sample_rate: int,
                           frequencies: np.ndarray, window_samples: int,
                           hop_samples: int,
                           jump_threshold: float = 0.5,
                           local_zscore: float = 6.0,
                           reference: Optional[np.ndarray] = None,
                           start_skip_samples: Optional[int] = None) -> List[PhaseBreak]:
    """Locate per-partial phase discontinuities introduced by a backend cut.

    Two adjacent, non-overlapping Hann windows are placed back to back; each
    partial's complex coefficient is fit jointly at its exact frequency in
    both windows. For a continuous tone the inter-window phase advance is
    exactly ``2*pi*f*dt`` even while amplitudes ramp, because the joint fit
    removes neighbour leakage and the coefficient tracks instantaneous phase.
    A cut inside the pair makes the observed advance deviate by the phase
    break, which stands out as a robust median/MAD outlier.

    Two kinds of non-switch events are removed so only backend-handover
    discontinuities are reported:

    * ``start_skip_samples`` -- windows overlapping the note-start transient
      are ignored (oscillators ramp from zero there in every render);
    * ``reference`` -- when a no-switch render is supplied, a break that also
      appears there at the same time is a *musical* phase event (e.g. the
      deliberate crossing-point edit present in every condition) and is not a
      handoff artifact.
    """
    window = np.hanning(window_samples)
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    skip = window_samples if start_skip_samples is None else int(start_skip_samples)
    ref_breaks = (
        _detect_phase_breaks(reference, sample_rate, freqs,
                             window_samples, hop_samples, window, jump_threshold,
                             local_zscore, skip)
        if reference is not None else []
    )
    ref_keys = {(b.partial_index, round(b.window_center, 5)) for b in ref_breaks}
    breaks = _detect_phase_breaks(audio, sample_rate, freqs, window_samples,
                                  hop_samples, window, jump_threshold,
                                  local_zscore, skip)
    if reference is not None:
        breaks = [b for b in breaks
                  if (b.partial_index, round(b.window_center, 5)) not in ref_keys]
    return breaks


def _detect_phase_breaks(audio: np.ndarray, sample_rate: int,
                         freqs: np.ndarray, window_samples: int,
                         hop_samples: int, window: np.ndarray,
                         jump_threshold: float, local_zscore: float,
                         start_skip_samples: int) -> List[PhaseBreak]:
    n = audio.size
    breaks: List[PhaseBreak] = []
    first_boundary = max(window_samples, start_skip_samples)
    boundaries = list(range(first_boundary, n - window_samples, hop_samples))
    dt = window_samples / sample_rate
    for j, f in enumerate(freqs):
        f = float(f)
        residuals: List[float] = []
        boundary_list: List[int] = []
        mags_list: List[float] = []
        for b in boundaries:
            ca = _window_partial_coeffs(audio, b - window_samples, window_samples,
                                        window, sample_rate, freqs)
            cb = _window_partial_coeffs(audio, b, window_samples, window,
                                        sample_rate, freqs)
            mag_a, mag_b = 2.0 * abs(ca[j]) / float(np.sum(window)), \
                           2.0 * abs(cb[j]) / float(np.sum(window))
            if mag_a <= 1e-6 or mag_b <= 1e-6:
                continue
            expected = 2 * math.pi * f * dt
            jump = (np.angle(cb[j]) - np.angle(ca[j]) - expected + math.pi) % (2 * math.pi) - math.pi
            residuals.append(float(jump))
            boundary_list.append(b)
            mags_list.append((mag_a, mag_b))

        if not residuals:
            continue
        r = np.asarray(residuals)
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) or 1e-12
        for idx, jump in enumerate(r):
            z = abs(jump - med) / (1.4826 * mad)
            if abs(jump) > jump_threshold and z > local_zscore:
                mag_a, mag_b = mags_list[idx]
                breaks.append(
                    PhaseBreak(j, f, boundary_list[idx] / sample_rate,
                               float(jump), mag_a, mag_b)
                )
    return breaks


# --------------------------------------------------------------------------
# Ownership audit: double counting / dropping
# --------------------------------------------------------------------------

@dataclass
class OwnershipAudit:
    switch_sample: int
    strategy: str
    max_weight_deviation: float       # max |w_td + w_fd - 1|
    min_total_weight: float
    max_total_weight: float
    double_counted_samples: int       # samples where total weight > 1 + tol
    dropped_samples: int              # samples where total weight < 1 - tol
    per_partial: List[Dict[str, float]] = field(default_factory=list)


def audit_ownership(audit: SwitchAudit, tolerance: float = 1e-12) -> OwnershipAudit:
    """Prove every partial is counted exactly once across the switch."""
    total = audit.td_weight + audit.fd_weight
    deviation = float(np.max(np.abs(total - 1.0))) if total.size else 0.0
    double = int(np.count_nonzero(total > 1.0 + tolerance))
    dropped = int(np.count_nonzero(total < 1.0 - tolerance))
    per_partial = []
    if total.size:
        for j in range(total.shape[1]):
            per_partial.append({
                "partial": j,
                "min_weight": float(np.min(total[:, j])),
                "max_weight": float(np.max(total[:, j])),
                "td_share": float(np.mean(audit.td_weight[:, j])),
                "fd_share": float(np.mean(audit.fd_weight[:, j])),
            })
    return OwnershipAudit(
        switch_sample=audit.switch_sample,
        strategy=audit.strategy,
        max_weight_deviation=deviation,
        min_total_weight=float(np.min(total)) if total.size else 1.0,
        max_total_weight=float(np.max(total)) if total.size else 1.0,
        double_counted_samples=double,
        dropped_samples=dropped,
        per_partial=per_partial,
    )


def per_partial_magnitude_track(audio: np.ndarray, sample_rate: int,
                                frequencies: np.ndarray, window_samples: int,
                                hop_samples: int) -> np.ndarray:
    """Per-partial magnitude in each short window (shape windows x partials).

    Used to see a partial vanish exactly at the switch sample (or survive
    twice as loud if both backends counted it).
    """
    n = audio.size
    window = np.hanning(window_samples)
    centers = list(range(window_samples // 2, n - window_samples // 2, hop_samples))
    mags = np.zeros((len(centers), frequencies.size), dtype=np.float64)
    for wi, c in enumerate(centers):
        lo = c - window_samples // 2
        seg = audio[lo : lo + window_samples] * window
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate)
        for j, f in enumerate(frequencies):
            k = int(np.argmin(np.abs(freqs - f)))
            mags[wi, j] = spec[k]
    return mags


def per_partial_amplitude_track(audio: np.ndarray, sample_rate: int,
                                frequencies: np.ndarray, window_samples: int,
                                hop_samples: int) -> np.ndarray:
    """Leakage-robust per-partial amplitude per window (shape windows x partials).

    A raw FFT bin mixes a partial with its neighbours' Hann-window sidelobes,
    so a silenced partial can still show a large bin magnitude from leakage.
    Here all partials are fit *jointly* by least squares to the windowed
    segment as a sum of cosine/sine pairs at their exact frequencies; the
    fitted coefficient magnitude isolates each partial's true amplitude
    regardless of neighbour leakage. Magnitudes are normalised by the window
    coherent gain so an amplitude-a sine reads ~a.
    """
    n = audio.size
    window = np.hanning(window_samples)
    coherent_gain = float(np.mean(window))
    t = np.arange(window_samples, dtype=np.float64) / sample_rate
    cols = []
    for f in np.atleast_1d(frequencies):
        ang = 2.0 * math.pi * float(f) * t
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    basis = np.stack(cols, axis=1) * window[:, None]
    centers = list(range(window_samples // 2, n - window_samples // 2, hop_samples))
    amps = np.zeros((len(centers), len(np.atleast_1d(frequencies))), dtype=np.float64)
    for wi, c in enumerate(centers):
        lo = c - window_samples // 2
        y = audio[lo : lo + window_samples] * window
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        coef = coef.reshape(-1, 2)
        amps[wi, :] = np.hypot(coef[:, 0], coef[:, 1]) / coherent_gain
    return amps


# --------------------------------------------------------------------------
# Compute burden
# --------------------------------------------------------------------------

@dataclass
class ComputeBurden:
    backend: str
    audio_seconds: float
    elapsed_seconds: float
    realtime_factor: float
    samples_per_second: float


def time_render(render_fn: Callable[[], np.ndarray], sample_rate: int,
                backend: str, repeats: int = 3) -> ComputeBurden:
    """Time a render callback (best of ``repeats``) and report realtime factor."""
    best = math.inf
    audio = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        audio = render_fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    audio_seconds = audio.size / sample_rate
    return ComputeBurden(
        backend=backend,
        audio_seconds=audio_seconds,
        elapsed_seconds=best,
        realtime_factor=audio_seconds / best if best > 0 else math.inf,
        samples_per_second=audio.size / best if best > 0 else math.inf,
    )

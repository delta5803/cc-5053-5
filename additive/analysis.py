"""Measurement utilities used to *prove* the synthesis is really additive.

These functions analyze rendered audio rather than trusting the
configuration. A partial line that exists only in a diagram but contributes
no energy will not show up as a spectral peak here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpectralPeak:
    frequency: float
    amplitude: float
    phase_radians: float


def spectrum(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided amplitude/phase spectrum of a (windowed) signal."""
    n = audio.size
    window = np.hanning(n)
    coherent_gain = float(np.sum(window))
    spectrum_complex = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    amplitudes = 2.0 * np.abs(spectrum_complex) / coherent_gain
    phases = np.angle(spectrum_complex)
    return freqs, amplitudes, phases


def find_peaks(
    audio: np.ndarray,
    sample_rate: int,
    expected_frequencies: Optional[np.ndarray] = None,
    tolerance_hz: float = 3.0,
    min_amplitude: float = 1e-5,
) -> List[SpectralPeak]:
    """Locate spectral peaks.

    If ``expected_frequencies`` is given, report the strongest bin within
    ``tolerance_hz`` of each expected frequency (a missing partial yields an
    explicit zero-amplitude entry instead of being silently omitted).
    Otherwise return all local maxima above ``min_amplitude``.
    """
    freqs, amps, phases = spectrum(audio, sample_rate)
    peaks: List[SpectralPeak] = []
    if expected_frequencies is not None:
        for f0 in np.atleast_1d(expected_frequencies):
            mask = np.abs(freqs - f0) <= tolerance_hz
            if not np.any(mask):
                peaks.append(SpectralPeak(float(f0), 0.0, 0.0))
                continue
            local = amps * mask
            bin_index = int(np.argmax(local))
            peaks.append(SpectralPeak(float(freqs[bin_index]), float(amps[bin_index]), float(phases[bin_index])))
        return peaks

    for i in range(1, amps.size - 1):
        if amps[i] > amps[i - 1] and amps[i] >= amps[i + 1] and amps[i] >= min_amplitude:
            peaks.append(SpectralPeak(float(freqs[i]), float(amps[i]), float(phases[i])))
    return peaks


def initial_samples(audio: np.ndarray, sample_rate: int, milliseconds: float = 2.0) -> np.ndarray:
    """Return the very first samples -- where initial phase is observable."""
    count = max(2, int(round(milliseconds / 1000.0 * sample_rate)))
    return audio[:count].copy()


def _partial_envelope(
    audio: np.ndarray,
    sample_rate: int,
    frequency: float,
    known_frequencies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract one partial's amplitude envelope without cross-partial bias.

    The analytic signal (Hilbert transform) gives the true instantaneous
    complex amplitude, so taking its magnitude does not fold neighboring
    partials into a DC beat the way demodulating the raw real signal would.
    Complex demodulation then moves the target partial to DC, and a
    zero-phase low-pass removes every other partial without smearing the
    envelope's timing. When the declared frequencies are known, the cutoff
    adapts to the nearest neighbor so dense/inharmonic partials stay isolated.
    """
    from scipy.signal import butter, filtfilt, hilbert

    n = audio.size
    if known_frequencies is not None:
        others = np.abs(np.asarray(known_frequencies, dtype=np.float64) - frequency)
        others = others[others > 1e-9]
        nearest_gap = float(np.min(others)) if others.size else frequency
        cutoff = min(nearest_gap * 0.35, frequency * 0.2, sample_rate * 0.4)
    else:
        cutoff = min(frequency * 0.2, sample_rate * 0.4)

    analytic = hilbert(audio)
    t = np.arange(n) / sample_rate
    demod = analytic * np.exp(-1j * 2.0 * math.pi * frequency * t)
    b, a = butter(4, cutoff / (sample_rate / 2.0), btype="low")
    pad = min(3 * max(len(a), len(b)), n - 1)
    filtered = filtfilt(b, a, demod, padlen=pad)
    return np.abs(filtered)


def fit_decay_tau(
    audio: np.ndarray,
    sample_rate: int,
    frequency: float,
    fit_start: float = 0.05,
    fit_end: Optional[float] = None,
    known_frequencies: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Estimate the exponential amplitude time-constant of one partial.

    The partial's analytic envelope is isolated and fit in log space.
    Returns (tau_seconds, r_squared); a constant amplitude gives tau = +inf.
    """
    n = audio.size
    t = np.arange(n) / sample_rate
    if fit_end is None:
        fit_end = t[-1]
    env = _partial_envelope(audio, sample_rate, frequency, known_frequencies)
    mask = (t >= fit_start) & (t <= fit_end) & (env > 1e-8)
    if np.count_nonzero(mask) < 4:
        return math.inf, 0.0
    log_env = np.log(env[mask])
    tt = t[mask]
    slope, intercept = np.polyfit(tt, log_env, 1)
    # A slope this shallow means no measurable decay over the fit window;
    # numerical noise in a constant envelope otherwise yields a huge fake tau.
    if slope >= -2e-4:
        return math.inf, 1.0
    tau = -1.0 / slope
    predicted = slope * tt + intercept
    ss_res = float(np.sum((log_env - predicted) ** 2))
    ss_tot = float(np.sum((log_env - log_env.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(tau), float(r2)


def thd_ratio(audio: np.ndarray, sample_rate: int, fundamental: float, n_harmonics: int = 10) -> float:
    """Energy above the declared partials vs. energy in them (distortion proxy)."""
    freqs, amps, _ = spectrum(audio, sample_rate)
    declared = fundamental * np.arange(1, n_harmonics + 1)
    signal_energy = 0.0
    for f0 in declared:
        mask = np.abs(freqs - f0) <= 2.0
        signal_energy += float(np.sum(amps[mask] ** 2))
    total_energy = float(np.sum(amps**2))
    noise_energy = max(total_energy - signal_energy, 0.0)
    return math.sqrt(noise_energy / signal_energy) if signal_energy > 0 else math.inf

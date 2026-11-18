"""End-to-end verification report.

Renders the small, fully-known timbre and produces quantitative evidence that
the additive reconstruction is real:

1. spectrum  -- measured spectral peaks sit exactly on the declared partials,
   with no energy at undeclared or folded frequencies;
2. phase     -- the first samples match the waveform implied by the declared
   initial phases (each partial starts at its own phase, not zero);
3. decay     -- exponential time constants measured from the rendered audio
   match the per-partial tau values in the configuration.

Run:  python -m additive.demo.verify [config.json] [outdir]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis import _partial_envelope, find_peaks, fit_decay_tau, spectrum
from ..config import TimbreConfig
from ..engine import AdditiveEngine


def main(config_path: str, outdir: str) -> int:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TimbreConfig.load(config_path)
    engine = AdditiveEngine(cfg)
    engine.note_on()
    duration = 1.5
    audio = engine.render_samples(int(duration * cfg.sample_rate), block_size=256)
    sr = cfg.sample_rate
    expected = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])

    # -- 1. spectrum -----------------------------------------------------------
    freqs, amps, _ = spectrum(audio, sr)
    peaks = find_peaks(audio, sr, expected_frequencies=expected)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(freqs, amps, lw=0.8, color="#3a6ea5")
    for target, p in zip(expected, peaks):
        ax.axvline(target, color="#cc5555", ls="--", lw=0.7, alpha=0.7)
        ax.plot(p.frequency, p.amplitude, "o", color="#228b22", ms=5)
    ax.set_xlim(0, expected[-1] * 1.4)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title("Measured spectrum (blue) vs declared partial frequencies (red dashed)")
    fig.tight_layout(); fig.savefig(out / "01_spectrum.png", dpi=130); plt.close(fig)

    # -- 2. initial phase waveform --------------------------------------------
    head_n = int(0.02 * sr)
    t_head = np.arange(head_n) / sr
    expected_wave = np.zeros(head_n)
    for p in cfg.partials:
        if p.attack <= 0:
            ramp = np.ones(head_n)
        else:
            ramp = np.minimum(t_head / p.attack, 1.0)
        decay = 1.0 if math.isinf(p.decay) else np.exp(-np.maximum(t_head - p.attack, 0.0) / p.decay)
        expected_wave += (
            cfg.master_amplitude * p.amplitude * ramp * decay
            * np.sin(2 * math.pi * p.ratio * cfg.base_frequency * t_head + p.phase)
        )
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t_head * 1000, audio[:head_n], label="engine output", lw=1.2)
    ax.plot(t_head * 1000, expected_wave, "--", label="sum from declared phases", lw=1.0)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("amplitude")
    ax.set_title("Initial-phase waveform: rendered output vs analytical sum")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "02_phase.png", dpi=130); plt.close(fig)

    # -- 3. decay envelopes ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t = np.arange(audio.size) / sr
    fit_end = min(1.2, duration - 0.1)
    rows = []
    for i, p in enumerate(cfg.partials):
        f = p.ratio * cfg.base_frequency
        env = _partial_envelope(audio, sr, f, expected)
        tau, r2 = fit_decay_tau(audio, sr, f, fit_start=max(0.05, 2 * p.attack),
                                fit_end=fit_end, known_frequencies=expected)
        ax.plot(t[t <= fit_end], env[t <= fit_end], label=f"partial {i} f={f:.0f} Hz")
        cfg_tau = p.decay
        rows.append((i, f, cfg_tau, tau, r2))
    ax.set_yscale("log"); ax.set_xlabel("time (s)"); ax.set_ylabel("envelope amplitude (log)")
    ax.set_title("Per-partial decay envelopes (straight lines = pure exponential)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "03_decay.png", dpi=130); plt.close(fig)

    # -- textual report --------------------------------------------------------
    lines = [f"verification report for {cfg.name} (f0={cfg.base_frequency} Hz, sr={sr})", "=" * 68]
    lines.append("spectrum placement")
    for target, p in zip(expected, peaks):
        lines.append(f"  target={target:9.3f} Hz  measured={p.frequency:9.3f} Hz  "
                     f"delta={p.frequency - target:+.3f}  amp={p.amplitude:.5f}")
    lines.append("")
    lines.append("decay tau (config vs measured from rendered audio)")
    for i, f, cfg_tau, tau, r2 in rows:
        a = "inf" if math.isinf(cfg_tau) else f"{cfg_tau:.3f}s"
        b = "inf" if math.isinf(tau) else f"{tau:.3f}s"
        lines.append(f"  partial {i}: config={a:>7}  measured={b:>7}  R^2={r2:.4f}")
    phase_mismatch = float(np.max(np.abs(audio[:head_n] - expected_wave)))
    lines.append("")
    lines.append(f"initial-phase max waveform error: {phase_mismatch:.2e}")
    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nplots written to {out}")
    return 0


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parents[1] / "configs" / "few.json")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parents[1] / "renders" / "plots")
    raise SystemExit(main(cfg_path, out_dir))

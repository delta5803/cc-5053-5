"""The backend-switch experiment: score, strategies, measurements and verdict.

The score is built so loudness alone cannot judge a switch. It contains:

* a **weak partial that attacks abruptly in the middle of a processing block**
  (its onset is offset from every block boundary, the hardest placement for
  onset timing);
* two **neighboring partials that cross slowly** in amplitude around the
  switch, close enough in frequency to produce an audible beat pattern;
* an optional **phase edit applied at the crossing point** to the two
  neighboring partials;
* a **negative control**: a sustained segment with no parameter change, where
  a correct switch must be inaudible and measure zero;
* a **disappearance case**: a partial is silenced on the exact switch sample,
  so short spectral windows can confirm it vanishes once -- not twice and not
  from both backends.

For each strategy (hard / handoff / parallel) the score is rendered together
with a no-switch reference, then first onset, beat frequency, switch noise,
phase breaks, ownership (double count/drop) and compute burden are measured.
A rule is chosen from measured thresholds and persisted for the next
automatic performance.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PartialSpec, TimbreConfig
from .dual_engine import SwitchAudit, SwitchEvent, DualBackendEngine
from .switch_metrics import (
    ComputeBurden,
    OnsetMeasurement,
    OwnershipAudit,
    PhaseBreak,
    SwitchNoise,
    audit_ownership,
    estimate_beat_frequency,
    measure_first_onset,
    measure_switch_noise,
    per_partial_amplitude_track,
    per_partial_magnitude_track,
    spectral_phase_windows,
    time_render,
)

STRATEGIES = ("hard", "handoff", "parallel")

# The rule chosen from measured evidence is persisted here so the next
# automatic performance loads the same backend-switch strategy.
RULE_PATH = Path(__file__).parent / "configs" / "selected_switch_rule.json"


@dataclass
class ScoreParams:
    """Knobs of the synthetic switch-test score (times in seconds)."""

    duration: float = 1.6
    switch_time: float = 0.8
    fd_hop: int = 256
    crossfade_seconds: float = 0.04
    weak_ratio: float = 5.1
    weak_amplitude: float = 0.08
    weak_attack: float = 0.002
    weak_onset_time: float = 0.55      # before the switch, mid-block placement
    weak_onset_offset: int = 97        # samples inside a hop (not a boundary)
    neighbor_ratio_a: float = 2.0
    neighbor_ratio_b: float = 2.03
    crossing_seconds: float = 0.5
    phase_edit_radians: float = math.pi / 2
    fd_phase_offset: float = 1.3
    disappear_ratio: float = 3.0       # steady partial removed on switch sample
    base_frequency: float = 220.0
    sample_rate: int = 48000


def build_score_config(params: ScoreParams) -> TimbreConfig:
    """Steady partials (no decay) so loudness stays ~constant across the switch."""
    partials = [
        PartialSpec(ratio=1.0, amplitude=0.5, phase=0.0, attack=0.0,
                    decay=math.inf, release=0.02, label="fundamental"),
        PartialSpec(ratio=params.neighbor_ratio_a, amplitude=0.4, phase=0.6,
                    attack=0.0, decay=math.inf, release=0.02, label="neighbor A"),
        PartialSpec(ratio=params.neighbor_ratio_b, amplitude=0.15, phase=2.1,
                    attack=0.0, decay=math.inf, release=0.02, label="neighbor B"),
        PartialSpec(ratio=params.disappear_ratio, amplitude=0.3, phase=1.7,
                    attack=0.0, decay=math.inf, release=0.02, label="disappear"),
        PartialSpec(ratio=params.weak_ratio, amplitude=0.0, phase=1.1,
                    attack=params.weak_attack, decay=math.inf, release=0.02,
                    label="weak"),
    ]
    return TimbreConfig(
        name="switch-experiment",
        base_frequency=params.base_frequency,
        partials=partials,
        master_amplitude=0.6,
        transition_time=0.0,
        sample_rate=params.sample_rate,
        fd_hop=params.fd_hop,
    )


def neighbor_amplitude(t: float, params: ScoreParams, which: str) -> float:
    """Slow deterministic linear cross shared by every rendered condition."""
    sw, half = params.switch_time, params.crossing_seconds / 2.0
    frac = np.clip((t - (sw - half)) / (2 * half), 0.0, 1.0)
    return float((0.4 - 0.25 * frac) if which == "a" else (0.15 + 0.25 * frac))


@dataclass
class ScoreResult:
    audio: np.ndarray
    audits: List[SwitchAudit]
    weak_onset_sample: int
    switch_sample: int


def render_score(params: ScoreParams, strategy: str, start_backend: str = "td",
                 apply_phase_edit: bool = True, disappear: bool = False,
                 no_switch: bool = False) -> ScoreResult:
    """Render one condition. All non-switch parameter events are identical
    across conditions so the no-switch render is a valid reference."""
    cfg = build_score_config(params)
    sr, hop = params.sample_rate, params.fd_hop
    n_total = int(round(params.duration * sr))
    sw = int(round(params.switch_time * sr / hop)) * hop
    weak_idx, dis_idx = 4, 3
    weak_onset = (
        int(round(params.weak_onset_time * sr / hop)) * hop + params.weak_onset_offset
    )

    engine = DualBackendEngine(cfg, start_backend=start_backend)
    note_id = engine.note_on(fd_phase_offset=params.fd_phase_offset)
    # The crossing-point phase edit is a musical event applied to both
    # backends, so it appears identically in the no-switch reference.
    common = None
    if apply_phase_edit:
        common = np.zeros(len(cfg.partials))
        common[1] = params.phase_edit_radians
        common[2] = params.phase_edit_radians
    if no_switch:
        # A reference event carries the common phase edit but no handoff.
        engine.schedule_switch(note_id, SwitchEvent(
            sw, "handoff", 0, None, common, active=False))
    else:
        xf = int(round(params.crossfade_seconds * sr)) if strategy == "parallel" else 0
        engine.schedule_switch(note_id, SwitchEvent(sw, strategy, xf, None, common))

    audio = np.zeros(n_total, dtype=np.float64)
    weak_fired = False
    cursor = 0
    while cursor < n_total:
        size = min(hop, n_total - cursor)
        t_mid = (cursor + hop / 2) / sr
        amps = np.array([
            0.5,
            neighbor_amplitude(t_mid, params, "a"),
            neighbor_amplitude(t_mid, params, "b"),
            cfg.partials[dis_idx].amplitude,
            cfg.partials[weak_idx].amplitude,
        ])
        if not weak_fired and cursor <= weak_onset < cursor + hop:
            amps[weak_idx] = params.weak_amplitude
            weak_fired = True
        # Latch: once the switch hop is reached the partial stays silenced.
        # sw is hop-quantized, so this is the exact switch sample.
        if disappear and cursor >= sw:
            amps[dis_idx] = 0.0
        engine.set_partial_amplitudes_on_voice(note_id, amps)
        audio[cursor : cursor + size] = engine.render_block(hop)[:size]
        cursor += size

    return ScoreResult(audio, engine.audits(), weak_onset, sw)


def render_sustained_control(params: ScoreParams, strategy: str = "handoff",
                             no_switch: bool = False) -> np.ndarray:
    """A sustained tone with no parameter changes anywhere in the timeline.

    All amplitudes stay frozen at their spec values (the weak partial stays
    silent); only a backend switch -- or an inactive reference event --
    distinguishes the two renders. A correct handoff must therefore be
    inaudible and measure zero against the no-switch render.
    """
    cfg = build_score_config(params)
    sr, hop = params.sample_rate, params.fd_hop
    n_total = int(round(params.duration * sr))
    sw = int(round(params.switch_time * sr / hop)) * hop
    amps = np.array([p.amplitude for p in cfg.partials])

    engine = DualBackendEngine(cfg, start_backend="td")
    note_id = engine.note_on(fd_phase_offset=params.fd_phase_offset)
    if no_switch:
        engine.schedule_switch(note_id, SwitchEvent(
            sw, "handoff", 0, None, None, active=False))
    else:
        engine.schedule_switch(note_id, SwitchEvent(sw, strategy, 0))
    engine.set_partial_amplitudes_on_voice(note_id, amps)

    audio = np.zeros(n_total, dtype=np.float64)
    cursor = 0
    while cursor < n_total:
        size = min(hop, n_total - cursor)
        audio[cursor : cursor + size] = engine.render_block(hop)[:size]
        cursor += size
    return audio


@dataclass
class StrategyResult:
    strategy: str
    audio: np.ndarray
    reference: np.ndarray
    onset: OnsetMeasurement
    onset_error_samples: int
    beat_hz: float
    beat_strength: float
    beat_error_hz: float
    switch_noise: SwitchNoise
    phase_breaks: List[PhaseBreak]
    ownership: Optional[OwnershipAudit]
    burden: ComputeBurden
    crossfade_samples: int = 0


def run_strategy(params: ScoreParams, strategy: str, reference: np.ndarray,
                 apply_phase_edit: bool = True) -> StrategyResult:
    score = render_score(params, strategy, apply_phase_edit=apply_phase_edit)
    audio, audits, sw = score.audio, score.audits, score.switch_sample
    sr = params.sample_rate
    cfg = build_score_config(params)
    freqs = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])

    onset = measure_first_onset(audio, sr)
    ref_onset = measure_first_onset(reference, sr)
    onset_error = onset.onset_sample - ref_onset.onset_sample

    beat_hz, beat_strength = estimate_beat_frequency(
        audio, sr, freqs[1], known_frequencies=freqs,
        fit_start=params.switch_time - 0.3, fit_end=params.switch_time + 0.3,
    )
    ref_beat_hz, _ = estimate_beat_frequency(
        reference, sr, freqs[1], known_frequencies=freqs,
        fit_start=params.switch_time - 0.3, fit_end=params.switch_time + 0.3,
    )
    xf = int(round(params.crossfade_seconds * sr)) if strategy == "parallel" else 0
    noise = measure_switch_noise(audio, reference, sr, sw, strategy)
    # Ignore the note-start region (two windows) and the musical phase edit,
    # which is also present in the no-switch reference.
    breaks = spectral_phase_windows(
        audio, sr, freqs, window_samples=max(512, params.fd_hop * 2),
        hop_samples=params.fd_hop // 2, jump_threshold=0.5,
        reference=reference, start_skip_samples=max(512, params.fd_hop * 2),
    )
    ownership = audit_ownership(audits[0]) if audits else None

    def render_fn() -> np.ndarray:
        return render_score(params, strategy, apply_phase_edit=apply_phase_edit).audio

    burden = time_render(render_fn, sr, backend=strategy)
    return StrategyResult(
        strategy=strategy, audio=audio, reference=reference, onset=onset,
        onset_error_samples=onset_error, beat_hz=beat_hz, beat_strength=beat_strength,
        beat_error_hz=beat_hz - ref_beat_hz, switch_noise=noise, phase_breaks=breaks,
        ownership=ownership, burden=burden, crossfade_samples=xf,
    )


def _result_to_dict(r: StrategyResult) -> Dict[str, Any]:
    return {
        "strategy": r.strategy,
        "onset_sample": r.onset.onset_sample,
        "onset_time": r.onset.onset_time,
        "onset_error_samples": r.onset_error_samples,
        "beat_hz": r.beat_hz,
        "beat_strength": r.beat_strength,
        "beat_error_hz": r.beat_error_hz,
        "switch_excess_peak": r.switch_noise.excess_peak,
        "switch_excess_rms": r.switch_noise.excess_rms,
        "switch_click_energy": r.switch_noise.click_energy,
        "phase_break_count": len(r.phase_breaks),
        "phase_breaks": [
            {"partial": b.partial_index, "freq": b.frequency,
             "window_center": b.window_center, "jump_rad": b.phase_jump_radians}
            for b in r.phase_breaks
        ],
        "ownership": (asdict(r.ownership) if r.ownership else None),
        "compute": asdict(r.burden),
        "crossfade_samples": r.crossfade_samples,
    }


def choose_rule(results: List[StrategyResult], params: ScoreParams) -> Dict[str, Any]:
    """Pick a strategy from measured thresholds and state its usable range."""
    safe = [r for r in results if r.ownership is not None
            and r.ownership.double_counted_samples == 0
            and r.ownership.dropped_samples == 0]
    candidates = [
        r for r in safe
        if abs(r.onset_error_samples) <= params.fd_hop and abs(r.beat_error_hz) < 1.0
    ] or safe
    chosen = min(candidates, key=lambda r: r.switch_noise.click_energy)
    ratio = chosen.switch_noise.excess_rms / max(chosen.switch_noise.signal_rms, 1e-12)
    return {
        "strategy": chosen.strategy,
        "reason": "lowest measured switch click energy among ownership-safe strategies",
        "switch_noise_db_below_signal": 20 * math.log10(ratio + 1e-12),
        "onset_error_samples": chosen.onset_error_samples,
        "beat_error_hz": chosen.beat_error_hz,
        "crossfade_samples": chosen.crossfade_samples,
    }


def applicable_range(results: List[StrategyResult], params: ScoreParams) -> Dict[str, float]:
    return {
        "min_click_energy": float(min(r.switch_noise.click_energy for r in results)),
        "max_click_energy": float(max(r.switch_noise.click_energy for r in results)),
        "min_render_seconds": float(min(r.burden.elapsed_seconds for r in results)),
        "max_render_seconds": float(max(r.burden.elapsed_seconds for r in results)),
        "realtime_factor_min": float(min(r.burden.realtime_factor for r in results)),
        "hop_seconds": params.fd_hop / params.sample_rate,
        "crossfade_seconds": params.crossfade_seconds,
    }


@dataclass
class ExperimentReport:
    params: Dict[str, Any]
    results: List[Dict[str, Any]]
    negative_control: Dict[str, Any]
    disappearance: Dict[str, Any]
    chosen_rule: Dict[str, Any]
    applicable_range: Dict[str, float]


def run_experiment(params: Optional[ScoreParams] = None,
                   out_path: Optional[str] = None,
                   persist_rule: bool = True) -> ExperimentReport:
    params = params or ScoreParams()
    sr = params.sample_rate
    # Reference shares every musical event (including the phase edit) but
    # performs no backend switch, so difference energy is handoff noise only.
    reference = render_score(params, "handoff", no_switch=True).audio
    results = [run_strategy(params, s, reference, apply_phase_edit=True)
               for s in STRATEGIES]

    # Negative control: a sustained tone with no parameter change at the
    # switch. The switched render must coincide with the no-switch render
    # around the cut (an inaudible, measure-zero handoff).
    ctrl = render_sustained_control(params, "handoff", no_switch=False)
    ctrl_ref = render_sustained_control(params, "handoff", no_switch=True)
    sw = int(round(params.switch_time * sr / params.fd_hop)) * params.fd_hop
    ctrl_noise = measure_switch_noise(ctrl, ctrl_ref, sr, sw, "control")
    negative_control = {
        "description": "sustained tone, no parameter change at the switch point",
        "excess_peak": ctrl_noise.excess_peak,
        "excess_rms": ctrl_noise.excess_rms,
        "click_energy": ctrl_noise.click_energy,
        "pass": bool(ctrl_noise.excess_rms < 1e-9 and ctrl_noise.excess_peak < 1e-9),
    }

    # Disappearance: the dedicated steady partial (index 3) is silenced on the
    # exact switch sample and stays silent (latched). Its magnitude must drop
    # to the noise floor on post-switch windows while the same partial keeps
    # sounding in a no-switch reference, proving it vanished exactly once
    # rather than being double counted or dropped by the handoff machinery.
    dis = render_score(params, "handoff", apply_phase_edit=False, disappear=True)
    dis_ref = render_score(params, "handoff", apply_phase_edit=False, no_switch=True)
    cfg = build_score_config(params)
    freqs = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])
    win = max(512, params.fd_hop * 2)
    # Leakage-robust fit isolates the silenced partial's true amplitude;
    # raw bin magnitudes are kept to show the spectral-window leakage floor.
    amp_track = per_partial_amplitude_track(dis.audio, sr, freqs, win, params.fd_hop // 2)
    ref_amp = per_partial_amplitude_track(dis_ref.audio, sr, freqs, win, params.fd_hop // 2)
    mag_track = per_partial_magnitude_track(dis.audio, sr, freqs, win, params.fd_hop // 2)
    centers = np.arange(win // 2, dis.audio.size - win // 2, params.fd_hop // 2)
    # "Before" windows end before the switch; "after" windows start at least one
    # FD hop past it, excluding the look-ahead tail that still carries
    # pre-switch energy. Windows straddling the cut are excluded.
    clean_after = (centers - win // 2) >= sw + params.fd_hop
    clean_before = (centers + win // 2) < sw
    dis_col = 3
    before_idx = np.where(clean_before)[0]
    after_idx = np.where(clean_after)[0]
    amp_before = float(np.mean(amp_track[before_idx[-3:], dis_col])) if before_idx.size >= 3 else 0.0
    amp_after = float(np.mean(amp_track[after_idx[:3], dis_col])) if after_idx.size >= 3 else 0.0
    ref_after = float(np.mean(ref_amp[after_idx[:3], dis_col])) if after_idx.size >= 3 else 0.0
    bin_before = float(np.mean(mag_track[before_idx[-3:], dis_col])) if before_idx.size >= 3 else 0.0
    bin_after = float(np.mean(mag_track[after_idx[:3], dis_col])) if after_idx.size >= 3 else 0.0
    ownership_dis = audit_ownership(dis.audits[0]) if dis.audits else None
    disappearance = {
        "description": "steady partial silenced on the exact switch sample",
        "partial_index": dis_col,
        "amplitude_before": amp_before,
        "amplitude_after": amp_after,
        "reference_amplitude_after": ref_after,
        "bin_magnitude_before": bin_before,
        "bin_magnitude_after": bin_after,
        "vanishes_once": bool(amp_after < amp_before * 0.05 and ref_after > amp_before * 0.5),
        "ownership": asdict(ownership_dis) if ownership_dis else None,
    }

    rule = choose_rule(results, params)
    report = ExperimentReport(
        params={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in asdict(params).items()},
        results=[_result_to_dict(r) for r in results],
        negative_control=negative_control,
        disappearance=disappearance,
        chosen_rule=rule,
        applicable_range=applicable_range(results, params),
    )
    if out_path:
        Path(out_path).write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if persist_rule:
        save_switch_rule(rule, report.applicable_range, params)
    return report


def save_switch_rule(rule: Dict[str, Any], applicable: Dict[str, float],
                     params: ScoreParams, path: Optional[Path] = None) -> Path:
    """Persist the chosen strategy so the next automatic performance uses it."""
    path = path or RULE_PATH
    payload = {
        "strategy": rule["strategy"],
        "crossfade_samples": rule["crossfade_samples"],
        "switch_noise_db_below_signal": rule["switch_noise_db_below_signal"],
        "onset_error_samples": rule["onset_error_samples"],
        "beat_error_hz": rule["beat_error_hz"],
        "fd_hop": params.fd_hop,
        "sample_rate": params.sample_rate,
        "applicable_range": applicable,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_switch_rule(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the persisted rule, or ``None`` if no rule has been chosen yet."""
    path = path or RULE_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def switch_event_from_rule(sample_index: int,
                           path: Optional[Path] = None,
                           fallback: str = "handoff",
                           sample_rate: int = 48000) -> SwitchEvent:
    """Build the next automatic performance's switch event from the saved rule.

    If no rule has been persisted yet, fall back to a safe phase ``handoff``.
    The loaded crossfade length (in samples) is reused for the parallel
    strategy so the automatic run reproduces the measured, chosen behaviour.
    """
    rule = load_switch_rule(path)
    if rule is None:
        return SwitchEvent(sample_index, fallback, 0)
    strategy = rule.get("strategy", fallback)
    crossfade = int(rule.get("crossfade_samples", 0)) if strategy == "parallel" else 0
    return SwitchEvent(sample_index, strategy, crossfade)

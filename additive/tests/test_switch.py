"""Tests for the FD backend, dual-backend handoff and the switch experiment."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from additive.config import PartialSpec, TimbreConfig
from additive.engine import AdditiveEngine
from additive.fd_engine import FDEngine, FDNoteVoice
from additive.dual_engine import DualBackendEngine, SwitchEvent
from additive.switch_metrics import (
    audit_ownership,
    estimate_beat_frequency,
    measure_first_onset,
    measure_switch_noise,
    per_partial_amplitude_track,
    spectral_phase_windows,
)
from additive import experiment
from additive.experiment import (
    ScoreParams,
    load_switch_rule,
    render_score,
    render_sustained_control,
    run_experiment,
    switch_event_from_rule,
)

SR = 48000
HOP = 256


def make_config(**kwargs):
    partials = [
        PartialSpec(ratio=1.0, amplitude=0.5, phase=0.0, decay=math.inf, release=0.02),
        PartialSpec(ratio=2.0, amplitude=0.4, phase=0.6, decay=math.inf, release=0.02),
        PartialSpec(ratio=2.03, amplitude=0.2, phase=2.1, decay=math.inf, release=0.02),
    ]
    defaults = dict(name="switch", base_frequency=220.0, partials=partials,
                    master_amplitude=0.6, transition_time=0.0,
                    sample_rate=SR, fd_hop=HOP)
    defaults.update(kwargs)
    return TimbreConfig(**defaults)


# --------------------------------------------------------------------------
# FD backend: interior reconstruction matches the time-domain sum
# --------------------------------------------------------------------------

def test_fd_matches_time_domain_in_interior():
    cfg = make_config()
    n = 16 * HOP
    td = AdditiveEngine(cfg)
    fd = FDEngine(cfg)
    td.note_on()
    fd.note_on()
    td_out = td.render_samples(n, HOP)
    fd_out = fd.render_samples(n, HOP)
    assert td_out.shape == fd_out.shape
    # Drop the first and last hop (latency / flush boundaries).
    inner = slice(2 * HOP, n - HOP)
    assert np.max(np.abs(td_out[inner] - fd_out[inner])) < 1e-12


def test_fd_requires_hop_sized_blocks():
    voice = FDNoteVoice(0, make_config(), 1.0)
    with pytest.raises(ValueError):
        voice.render_block_parts(HOP + 1)


# --------------------------------------------------------------------------
# Dual backend: no-switch output equals a single backend, and handoff is clean
# --------------------------------------------------------------------------

def test_dual_no_switch_equals_single_backend():
    cfg = make_config()
    n = 12 * HOP
    dual = DualBackendEngine(cfg, start_backend="td")
    single = AdditiveEngine(cfg)
    dual.note_on(fd_phase_offset=1.3)
    single.note_on()
    # An inactive event must not change the output path.
    dual.schedule_switch(0, SwitchEvent(6 * HOP, "handoff", 0, None, None, active=False))
    d = dual.render_samples(n, HOP)
    s = single.render_samples(n, HOP)
    assert np.max(np.abs(d[HOP:] - s[HOP:])) < 1e-10


def test_handoff_is_continuous_but_hard_cut_clicks():
    params = ScoreParams()
    # All renders share the same musical events (phase edit included); only
    # the backend strategy differs, so difference energy is handoff noise.
    reference = render_score(params, "handoff", no_switch=True).audio
    hard = render_score(params, "hard", apply_phase_edit=True).audio
    handoff = render_score(params, "handoff", apply_phase_edit=True).audio
    sw = int(round(params.switch_time * SR / HOP)) * HOP
    hard_noise = measure_switch_noise(hard, reference, SR, sw, "hard")
    hand_noise = measure_switch_noise(handoff, reference, SR, sw, "handoff")
    # The naive hard cut injects a clearly audible discontinuity.
    assert hard_noise.excess_rms > 1e-3
    assert hard_noise.click_energy > 1e-4
    # Phase handoff is at floating-point round-off.
    assert hand_noise.excess_rms < 1e-12
    assert hand_noise.click_energy < 1e-20


def test_parallel_crossfade_keeps_exact_ownership():
    cfg = make_config()
    n = 16 * HOP
    dual = DualBackendEngine(cfg, start_backend="td")
    dual.note_on(fd_phase_offset=1.3)
    dual.schedule_switch(0, SwitchEvent(8 * HOP, "parallel", 4 * HOP))
    dual.render_samples(n, HOP)
    audits = dual.audits()
    assert len(audits) == 1
    ownership = audit_ownership(audits[0])
    assert ownership.double_counted_samples == 0
    assert ownership.dropped_samples == 0
    assert ownership.max_weight_deviation < 1e-12
    # Every partial stays exactly once across the whole crossfade window.
    total = audits[0].td_weight + audits[0].fd_weight
    assert np.allclose(total, 1.0, atol=1e-12)


def test_switch_is_quantized_to_hop_boundary():
    cfg = make_config()
    dual = DualBackendEngine(cfg)
    dual.note_on()
    requested = 6 * HOP + 37
    dual.schedule_switch(0, SwitchEvent(requested, "handoff", 0))
    event = dual.voices[0].switches[0]
    assert event.sample_index % HOP == 0
    assert event.sample_index == 6 * HOP


def test_parallel_requires_positive_crossfade():
    cfg = make_config()
    dual = DualBackendEngine(cfg)
    dual.note_on()
    with pytest.raises(ValueError):
        dual.schedule_switch(0, SwitchEvent(4 * HOP, "parallel", 0))


# --------------------------------------------------------------------------
# Measurements taken from the actual output
# --------------------------------------------------------------------------

def test_first_onset_detects_weak_partial_placement():
    score = render_score(ScoreParams(), "handoff")
    onset = measure_first_onset(score.audio, SR)
    assert onset.onset_sample > 0
    # The onset must be stable against the no-switch reference.
    ref = render_score(ScoreParams(), "handoff", no_switch=True)
    ref_onset = measure_first_onset(ref.audio, SR)
    assert abs(onset.onset_sample - ref_onset.onset_sample) == 0


def test_beat_frequency_recovers_neighbor_difference():
    # Neighbours at 2.00 and 2.03 times 220 Hz differ by 6.6 Hz; the measured
    # beat must be a low, stable rate and match the no-switch reference.
    params = ScoreParams()
    score = render_score(params, "handoff")
    ref = render_score(params, "handoff", no_switch=True)
    cfg = experiment.build_score_config(params)
    f = cfg.partials[1].ratio * cfg.base_frequency
    freqs = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])
    beat, strength = estimate_beat_frequency(
        score.audio, SR, f, known_frequencies=freqs,
        fit_start=params.switch_time - 0.3, fit_end=params.switch_time + 0.3)
    ref_beat, _ = estimate_beat_frequency(
        ref.audio, SR, f, known_frequencies=freqs,
        fit_start=params.switch_time - 0.3, fit_end=params.switch_time + 0.3)
    assert 0.0 < beat < 20.0
    assert strength > 0.0
    assert abs(beat - ref_beat) < 1.0


def test_phase_windows_flag_hard_cut_only():
    params = ScoreParams()
    cfg = experiment.build_score_config(params)
    freqs = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])
    win = max(512, HOP * 2)
    reference = render_score(params, "handoff", no_switch=True).audio
    hard = render_score(params, "hard").audio
    handoff = render_score(params, "handoff").audio
    hard_breaks = spectral_phase_windows(
        hard, SR, freqs, win, HOP // 2, reference=reference,
        start_skip_samples=win)
    clean_breaks = spectral_phase_windows(
        handoff, SR, freqs, win, HOP // 2, reference=reference,
        start_skip_samples=win)
    assert len(hard_breaks) > 0
    # Every reported break is localised at the switch, not the note start.
    sw = params.switch_time
    assert all(abs(b.window_center - sw) < 0.05 for b in hard_breaks)
    assert clean_breaks == []


def test_amplitude_track_isolates_silenced_partial():
    params = ScoreParams()
    cfg = experiment.build_score_config(params)
    freqs = np.array([p.ratio * cfg.base_frequency for p in cfg.partials])
    dis = render_score(params, "handoff", apply_phase_edit=False, disappear=True)
    win = max(512, HOP * 2)
    track = per_partial_amplitude_track(dis.audio, SR, freqs, win, HOP // 2)
    centers = np.arange(win // 2, dis.audio.size - win // 2, HOP // 2)
    sw = dis.switch_sample
    before = np.where((centers + win // 2) < sw)[0]
    after = np.where((centers - win // 2) >= sw + HOP)[0]
    amp_before = np.mean(track[before[-3:], 3])
    amp_after = np.mean(track[after[:3], 3])
    assert amp_before > 0.2
    assert amp_after < amp_before * 0.05


# --------------------------------------------------------------------------
# Whole experiment: controls, rule choice and persistence
# --------------------------------------------------------------------------

def test_run_experiment_controls_and_rule(tmp_path):
    rule_path = tmp_path / "rule.json"
    old_path = experiment.RULE_PATH
    experiment.RULE_PATH = rule_path
    try:
        report = run_experiment(ScoreParams(), out_path=str(tmp_path / "r.json"))
        # Negative control: a parameter-free switch is measure-zero.
        assert report.negative_control["pass"] is True
        assert report.negative_control["excess_rms"] < 1e-9
        # The dedicated partial vanishes exactly once, ownership stays exact.
        assert report.disappearance["vanishes_once"] is True
        assert report.disappearance["ownership"]["double_counted_samples"] == 0
        assert report.disappearance["ownership"]["dropped_samples"] == 0
        # Hard cut is dirty; handoff/parallel are machine-clean.
        by = {r["strategy"]: r for r in report.results}
        assert by["hard"]["phase_break_count"] > 0
        assert by["handoff"]["phase_break_count"] == 0
        assert by["parallel"]["switch_excess_rms"] < 1e-12
        # The rule was persisted and reloads.
        loaded = load_switch_rule(rule_path)
        assert loaded is not None
        assert loaded["strategy"] == report.chosen_rule["strategy"]
        assert loaded["strategy"] in ("handoff", "parallel")
    finally:
        experiment.RULE_PATH = old_path


def test_sustained_negative_control_is_silent_difference():
    params = ScoreParams()
    switched = render_sustained_control(params, "handoff", no_switch=False)
    reference = render_sustained_control(params, "handoff", no_switch=True)
    sw = int(round(params.switch_time * SR / HOP)) * HOP
    half = int(0.05 * SR)
    diff = switched[sw - half : sw + half] - reference[sw - half : sw + half]
    assert np.max(np.abs(diff)) < 1e-12


def test_next_automatic_performance_uses_persisted_rule(tmp_path):
    rule_path = tmp_path / "rule.json"
    old_path = experiment.RULE_PATH
    experiment.RULE_PATH = rule_path
    try:
        # No rule yet: safe handoff fallback with no crossfade.
        event = switch_event_from_rule(8 * HOP, rule_path)
        assert event.strategy == "handoff"
        assert event.crossfade_samples == 0

        # Persist a parallel rule through a real experiment run.
        run_experiment(ScoreParams())
        event = switch_event_from_rule(8 * HOP, rule_path)
        loaded = load_switch_rule(rule_path)
        assert event.strategy == loaded["strategy"] == "parallel"
        assert event.crossfade_samples == loaded["crossfade_samples"] > 0

        # The event drives a real dual-backend performance cleanly.
        cfg = experiment.build_score_config(ScoreParams())
        dual = DualBackendEngine(cfg, start_backend="td")
        note = dual.note_on(fd_phase_offset=ScoreParams().fd_phase_offset)
        dual.schedule_switch(note, event)
        out = dual.render_samples(12 * HOP, HOP)
        assert out.shape == (12 * HOP,)
        assert len(dual.audits()) == 1
        assert dual.audits()[0].strategy == "parallel"
    finally:
        experiment.RULE_PATH = old_path

import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from additive.analysis import find_peaks, fit_decay_tau, spectrum
from additive.config import PartialSpec, TimbreConfig
from additive.engine import AdditiveEngine, NoteVoice
from additive.render import NoteEvent, render_timeline
from additive.audio_io import export_wav


SR = 48000


def make_config(partials, **kwargs):
    defaults = dict(
        name="test",
        base_frequency=200.0,
        partials=partials,
        master_amplitude=1.0,
        transition_time=0.01,
        bandlimit_mode="drop",
        sample_rate=SR,
    )
    defaults.update(kwargs)
    return TimbreConfig(**defaults)


def sine_partial(ratio, amp=1.0, phase=0.0, decay=math.inf, attack=0.0, release=0.02):
    return PartialSpec(ratio=ratio, amplitude=amp, phase=phase,
                       attack=attack, decay=decay, release=release)


# --------------------------------------------------------------------------
# 1. Spectrum positions: declared partials really appear at their frequencies
# --------------------------------------------------------------------------

def test_spectrum_peaks_at_declared_frequencies():
    cfg = make_config([
        sine_partial(1.0, 1.0),
        sine_partial(2.0, 0.7),
        sine_partial(3.0, 0.4),
        sine_partial(4.2, 0.25),  # inharmonic
    ])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    audio = engine.render_samples(SR)  # 1 s
    expected = cfg.base_frequency * np.array([1.0, 2.0, 3.0, 4.2])
    peaks = find_peaks(audio, SR, expected_frequencies=expected, tolerance_hz=2.5)
    assert len(peaks) == 4
    for target, peak, amp in zip(expected, peaks, [1.0, 0.7, 0.4, 0.25]):
        assert abs(peak.frequency - target) <= 2.5
        assert peak.amplitude == pytest.approx(amp, rel=0.03)
    # No significant energy anywhere else -> clean additive sum, no aliases.
    # Energy is summed at *all declared* frequencies (including inharmonic).
    freqs, amps, _ = spectrum(audio, SR)
    declared_energy = 0.0
    for target in expected:
        declared_energy += float(np.sum(amps[np.abs(freqs - target) <= 2.5] ** 2))
    total_energy = float(np.sum(amps**2))
    assert math.sqrt(max(total_energy - declared_energy, 0.0) / declared_energy) < 0.01


def test_zero_amplitude_partial_is_absent_from_audio():
    cfg = make_config([sine_partial(1.0, 1.0), sine_partial(2.0, 0.0), sine_partial(3.0, 0.5)])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    audio = engine.render_samples(SR)
    peaks = find_peaks(audio, SR,
                       expected_frequencies=cfg.base_frequency * np.array([1.0, 2.0, 3.0]),
                       tolerance_hz=2.0)
    assert peaks[1].amplitude < 1e-6  # a diagram entry with zero output proves nothing


# --------------------------------------------------------------------------
# 2. Initial phase produces the expected starting waveform
# --------------------------------------------------------------------------

def test_initial_phase_waveform():
    # One partial each; first sample region must match A*sin(phase) at t->0.
    for phase in (0.0, math.pi / 2, math.pi, -math.pi / 3):
        cfg = make_config([sine_partial(1.0, 0.8, phase=phase, attack=0.0)])
        engine = AdditiveEngine(cfg)
        engine.note_on()
        audio = engine.render_samples(64)
        t = np.arange(8) / SR
        expected = 0.8 * np.sin(2 * math.pi * cfg.base_frequency * t + phase)
        assert np.allclose(audio[:8], expected, atol=1e-9), phase


def test_combined_partials_initial_value():
    cfg = make_config([
        sine_partial(1.0, 1.0, phase=0.0, attack=0.0),
        sine_partial(2.0, 0.5, phase=math.pi / 2, attack=0.0),
        sine_partial(3.0, 0.25, phase=math.pi, attack=0.0),
    ])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    audio = engine.render_samples(16)
    t = np.arange(16) / SR
    expected = sum(
        p.amplitude * np.sin(2 * math.pi * p.ratio * cfg.base_frequency * t + p.phase)
        for p in cfg.partials
    )
    assert np.allclose(audio, expected, atol=1e-9)


# --------------------------------------------------------------------------
# 3. Exponential decay time constants measured from rendered audio
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 0.3, 0.8])
def test_decay_time_constant(tau):
    cfg = make_config([sine_partial(1.0, 1.0, decay=tau, attack=0.002, release=0.0)])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    audio = engine.render_samples(int(1.5 * SR))
    measured, r2 = fit_decay_tau(audio, SR, cfg.base_frequency, fit_start=0.05, fit_end=1.2)
    assert r2 > 0.995
    assert measured == pytest.approx(tau, rel=0.08)


def test_infinite_decay_stays_constant():
    cfg = make_config([sine_partial(1.0, 1.0, decay=math.inf, attack=0.0)])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    audio = engine.render_samples(SR)
    measured, _ = fit_decay_tau(audio, SR, cfg.base_frequency, fit_start=0.05, fit_end=0.9)
    assert math.isinf(measured)


# --------------------------------------------------------------------------
# 4. Continuous phase across processing blocks (no per-block restart)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("block_size", [1, 7, 64, 256, 1000])
def test_block_boundaries_keep_continuous_phase(block_size):
    cfg = make_config([sine_partial(1.0, 1.0, phase=0.7, attack=0.0),
                       sine_partial(2.3, 0.6, phase=1.9, attack=0.0)])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    chunked = engine.render_samples(5000, block_size=block_size)

    engine2 = AdditiveEngine(cfg)
    engine2.note_on()
    whole = engine2.render_samples(5000, block_size=5000)
    # Output must be numerically identical regardless of block framing:
    # phase is continuous state, never restarted. Floating-point accumulation
    # order differs slightly, hence a tight tolerance rather than bit equality.
    assert np.allclose(chunked, whole, atol=1e-12, rtol=0.0)
    # The first block's samples are computed the same way in both cases.
    assert np.array_equal(chunked[:block_size], whole[:block_size])


def test_phase_state_advances_not_resets():
    cfg = make_config([sine_partial(1.0, 1.0, phase=0.0, attack=0.0)])
    voice = NoteVoice(0, cfg)
    voice.render_block(100)
    phase_after_100 = voice.phase.copy()
    expected = (2 * math.pi * cfg.base_frequency * 100 / SR) % (2 * math.pi)
    assert np.allclose(phase_after_100, expected, atol=1e-12)
    voice.render_block(100)
    expected_200 = (2 * math.pi * cfg.base_frequency * 200 / SR) % (2 * math.pi)
    assert np.allclose(voice.phase, expected_200, atol=1e-12)
    assert not np.allclose(voice.phase, expected, atol=1e-9)


# --------------------------------------------------------------------------
# 5. Band limiting: declared unrepresentable partials are dropped/attenuated,
#    never folded back under their original identity
# --------------------------------------------------------------------------

def test_above_nyquist_partial_dropped_and_not_aliased():
    # 400 Hz fundamental; ratio 70 -> 28 kHz, beyond 24 kHz Nyquist.
    cfg = make_config([sine_partial(1.0, 1.0), sine_partial(70.0, 0.9)],
                      base_frequency=400.0, bandlimit_mode="drop", nyquist_margin=0.95)
    engine = AdditiveEngine(cfg)
    report = engine.bandlimit_report()
    assert report[1].action == "dropped"
    assert report[1].gain_scale == 0.0
    engine.note_on()
    audio = engine.render_samples(SR)
    freqs, amps, _ = spectrum(audio, SR)
    # rfft only reaches Nyquist (24 kHz); the alias of 28 kHz would land at
    # 20 kHz. The whole high band above the fundamental must stay silent.
    high_band = freqs > 5_000
    assert np.max(amps[high_band]) < 1e-7
    # Fundamental still present.
    assert np.max(amps[(freqs > 350) & (freqs < 450)]) > 0.5


def test_bandlimit_attenuate_tapers_to_zero():
    cfg = make_config([sine_partial(1.0, 1.0), sine_partial(59.0, 0.9)],
                      base_frequency=400.0, bandlimit_mode="attenuate", nyquist_margin=0.9)
    engine = AdditiveEngine(cfg)
    report = engine.bandlimit_report()
    # 23.6 kHz sits inside the taper band [21.6k, 24k).
    assert report[1].action == "attenuated"
    assert 0.0 < report[1].gain_scale < 1.0
    voice = NoteVoice(0, cfg)
    assert voice.target_gain[1] == pytest.approx(0.9 * report[1].gain_scale)


def test_drop_render_equals_kept_partial_only_reference():
    # Strongest possible no-alias proof: the dropped-partial render must be
    # numerically identical to rendering only the partials that survive.
    partials = [PartialSpec(ratio=float(k), amplitude=1.0 / k, phase=0.3 * k,
                            attack=0.003, decay=0.5, release=0.05) for k in range(1, 22)]
    cfg = TimbreConfig(name="hi", base_frequency=2000.0, partials=partials,
                       bandlimit_mode="drop", nyquist_margin=0.95, sample_rate=SR)
    engine = AdditiveEngine(cfg)
    engine.note_on()
    rendered = engine.render_samples(SR // 2)

    kept = [p for p, r in zip(partials, engine.bandlimit_report()) if r.action == "kept"]
    assert len(kept) < len(partials)  # sanity: something was actually dropped
    ref_cfg = TimbreConfig(name="ref", base_frequency=2000.0, partials=kept,
                           bandlimit_mode="drop", sample_rate=SR)
    ref_engine = AdditiveEngine(ref_cfg)
    ref_engine.note_on()
    reference = ref_engine.render_samples(SR // 2)
    assert np.allclose(rendered, reference, atol=1e-10, rtol=0.0)


def test_attenuate_render_matches_declared_cosine_scales():
    partials = [PartialSpec(ratio=1.0, amplitude=1.0, decay=0.5),
                PartialSpec(ratio=11.6, amplitude=0.5, decay=0.5)]  # 23.2 kHz, in taper band
    cfg = TimbreConfig(name="att", base_frequency=2000.0, partials=partials,
                       bandlimit_mode="attenuate", nyquist_margin=0.95, sample_rate=SR)
    engine = AdditiveEngine(cfg)
    report = engine.bandlimit_report()
    assert report[1].action == "attenuated"
    scale = report[1].gain_scale
    # Reference: same two partials but with the taper scale applied by hand.
    ref_partials = [PartialSpec(ratio=1.0, amplitude=1.0, decay=0.5),
                    PartialSpec(ratio=11.6, amplitude=0.5 * scale, decay=0.5)]
    ref_cfg = TimbreConfig(name="ref", base_frequency=2000.0, partials=ref_partials,
                           bandlimit_mode="drop", nyquist_margin=1.0, sample_rate=SR)
    e1, e2 = AdditiveEngine(cfg), AdditiveEngine(ref_cfg)
    e1.note_on(); e2.note_on()
    a, b = e1.render_samples(SR // 4), e2.render_samples(SR // 4)
    assert np.allclose(a, b, atol=1e-10, rtol=0.0)


def test_partial_at_or_above_nyquist_always_silenced():
    cfg = make_config([sine_partial(1.0, 1.0), sine_partial(60.0, 0.9)],
                      base_frequency=400.0, bandlimit_mode="attenuate")
    engine = AdditiveEngine(cfg)
    report = engine.bandlimit_report()  # 24 kHz == Nyquist
    assert report[1].action == "dropped"
    assert report[1].gain_scale == 0.0


def test_drop_policy_kept_partials_unchanged():
    cfg = make_config([sine_partial(1.0), sine_partial(2.0), sine_partial(3.0)])
    engine = AdditiveEngine(cfg)
    for r in engine.bandlimit_report():
        assert r.action == "kept"
        assert r.gain_scale == 1.0


# --------------------------------------------------------------------------
# 6. Parameter jumps are smoothed over configurable transition time
# --------------------------------------------------------------------------

def test_amplitude_change_ramps_no_click():
    cfg = make_config([sine_partial(1.0, 1.0, attack=0.0)], transition_time=0.05)
    engine = AdditiveEngine(cfg)
    engine.note_on()
    engine.render_samples(SR // 4)
    voice = engine.voices[0]
    engine.set_partial_amplitudes([0.2])
    # The gain state itself must follow a linear ramp, never jump.
    gains = [voice.gain[0]]
    for _ in range(10):
        engine.render_block(1)
        gains.append(voice.gain[0])
    diffs = np.diff(gains)
    # With a true linear ramp every per-sample step equals the nominal step.
    expected_step = (0.2 - 1.0) / (0.05 * SR)
    assert np.allclose(diffs, expected_step, rtol=1e-9, atol=1e-15)
    # After the transition window the target amplitude is reached exactly.
    engine.render_samples(int(0.1 * SR))
    assert voice.gain[0] == pytest.approx(0.2, abs=1e-12)


def test_ramp_eliminates_sample_discontinuity():
    # With transition_time = 0-ish, a jump would create a discontinuity;
    # with the ramp, per-sample output changes stay within the smooth case.
    cfg_ramp = make_config([sine_partial(1.0, 1.0, attack=0.0)], transition_time=0.05)
    eng = AdditiveEngine(cfg_ramp)
    eng.note_on()
    eng.render_samples(2048)
    eng.set_partial_amplitudes([0.1])
    smooth = eng.render_samples(64)
    # No sample differs from its neighbor by more than the steady-state swing.
    steady_swing = 2.0 * 0.1 + 1.0 * (0.9 / (0.05 * SR)) * 2
    assert np.max(np.abs(np.diff(smooth))) <= steady_swing + 1e-9


def test_master_gain_independent_and_ramped():
    cfg = make_config([sine_partial(1.0, 1.0, attack=0.0)], transition_time=0.02)
    engine = AdditiveEngine(cfg)
    engine.note_on()
    engine.render_samples(1024)
    engine.set_master_gain(0.3)
    engine.render_samples(1)
    assert engine.master_gain != 0.3  # still ramping
    engine.render_samples(int(0.05 * SR))
    assert engine.master_gain == pytest.approx(0.3, abs=1e-12)
    # Partial gains are untouched by the master gain.
    assert engine.voices[0].gain[0] == pytest.approx(1.0)


def test_per_note_gain_is_independent():
    cfg = make_config([sine_partial(1.0, 1.0, attack=0.0)])
    engine = AdditiveEngine(cfg)
    a = engine.note_on(note_gain=1.0)
    b = engine.note_on(note_gain=0.4)
    engine.set_note_gain(b, 0.2)
    engine.render_samples(2048)
    assert engine.voices[a].note_gain == pytest.approx(1.0)
    assert engine.voices[b].target_note_gain == 0.2


def test_configurable_transition_changes_ramp_length():
    cfg = make_config([sine_partial(1.0, 1.0, attack=0.0)], transition_time=0.1)
    engine = AdditiveEngine(cfg)
    assert engine.transition_samples == int(0.1 * SR)
    engine.set_transition_time(0.01)
    assert engine.transition_samples == int(0.01 * SR)
    assert engine.voices == {} or all(
        v.transition_samples == int(0.01 * SR) for v in engine.voices.values())


# --------------------------------------------------------------------------
# 7. Solo one partial vs full mix
# --------------------------------------------------------------------------

def test_solo_partial_silences_others():
    cfg = make_config([sine_partial(1.0, 1.0), sine_partial(2.0, 0.8), sine_partial(3.0, 0.5)])
    engine = AdditiveEngine(cfg)
    engine.note_on()
    engine.solo_partial(1)
    # Skip the transition ramp, then analyze the steady soloed state.
    engine.render_samples(int(0.1 * SR))
    audio = engine.render_samples(SR // 2)
    peaks = find_peaks(audio, SR,
                       expected_frequencies=cfg.base_frequency * np.array([1.0, 2.0, 3.0]),
                       tolerance_hz=2.0)
    assert peaks[0].amplitude < 1e-6
    assert peaks[1].amplitude == pytest.approx(0.8, rel=0.03)
    assert peaks[2].amplitude < 1e-6
    engine.solo_partial(None)
    engine.render_samples(int(0.1 * SR))  # ramps back in
    audio3 = engine.render_samples(SR // 2)
    peaks3 = find_peaks(audio3, SR,
                        expected_frequencies=cfg.base_frequency * np.array([1.0, 2.0, 3.0]),
                        tolerance_hz=2.0)
    assert all(p.amplitude > 0.1 for p in peaks3)


# --------------------------------------------------------------------------
# 8. Config round trip + validation
# --------------------------------------------------------------------------

def test_config_roundtrip(tmp_path):
    cfg = make_config([
        sine_partial(1.0, 0.9, 0.3, decay=0.5),
        sine_partial(2.0, 0.4, 0.2, decay=math.inf),
    ])
    path = cfg.save(tmp_path / "t.json")
    loaded = TimbreConfig.load(path)
    assert loaded.base_frequency == cfg.base_frequency
    assert len(loaded.partials) == 2
    assert loaded.partials[0].phase == 0.3
    assert math.isinf(loaded.partials[1].decay)
    assert loaded.bandlimit_mode == "drop"


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        TimbreConfig(name="x", base_frequency=-1, partials=[sine_partial(1.0)], sample_rate=SR)
    with pytest.raises(ValueError):
        TimbreConfig(name="x", base_frequency=100, partials=[], sample_rate=SR)
    with pytest.raises(ValueError):
        PartialSpec(ratio=0, amplitude=1.0)
    with pytest.raises(ValueError):
        PartialSpec(ratio=1, amplitude=-0.1)
    with pytest.raises(ValueError):
        make_config([sine_partial(1.0)], bandlimit_mode="weird")


# --------------------------------------------------------------------------
# 9. Note instances, release and timeline
# --------------------------------------------------------------------------

def test_note_off_release_and_voice_cleanup():
    cfg = make_config([sine_partial(1.0, 1.0, release=0.05)])
    engine = AdditiveEngine(cfg)
    nid = engine.note_on()
    engine.render_samples(1024)
    engine.note_off(nid)
    engine.render_samples(int(0.02 * SR))
    assert nid in engine.voices  # still releasing
    engine.render_samples(int(0.1 * SR))
    assert nid not in engine.voices  # cleaned up after release completes


def test_overlapping_notes_are_independent_voices():
    cfg = make_config([sine_partial(1.0, 1.0, release=0.05)])
    events = [
        NoteEvent(start=0.0, duration=0.3, freq_multiplier=1.0),
        NoteEvent(start=0.15, duration=0.3, freq_multiplier=1.5),
    ]
    audio = render_timeline(cfg, events, total_duration=0.8, block_size=128)
    peaks = find_peaks(audio[int(0.2 * SR): int(0.4 * SR)], SR,
                       expected_frequencies=np.array([200.0, 300.0]), tolerance_hz=3.0)
    assert all(p.amplitude > 0.1 for p in peaks)


# --------------------------------------------------------------------------
# 10. WAV export
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_wav_export_roundtrip(tmp_path, subtype):
    soundfile = pytest.importorskip("soundfile")
    cfg = make_config([sine_partial(1.0, 0.7)])
    audio = render_timeline(cfg, [NoteEvent(0, 0.2)], total_duration=0.3)
    path = export_wav(tmp_path / "out.wav", audio, SR, subtype=subtype)
    data, sr = soundfile.read(str(path))
    assert sr == SR
    assert np.max(np.abs(data)) > 0.1


def test_export_refuses_clipping(tmp_path):
    with pytest.raises(ValueError):
        export_wav(tmp_path / "x.wav", np.array([0.0, 1.5, -1.2]), SR)


# --------------------------------------------------------------------------
# 11. Performance with many partials and many voices
# --------------------------------------------------------------------------

def test_many_partials_runs_faster_than_realtime():
    n_partials = 128
    partials = [
        PartialSpec(ratio=float(k), amplitude=1.0 / k, phase=0.1 * k,
                    attack=0.003, decay=max(0.1, 1.0 / k), release=0.05)
        for k in range(1, n_partials + 1)
    ]
    cfg = TimbreConfig(name="perf", base_frequency=110.0, partials=partials,
                       bandlimit_mode="attenuate", sample_rate=SR)
    engine = AdditiveEngine(cfg)
    for mult in (1.0, 1.2, 1.5):
        engine.note_on(freq_multiplier=mult)
    block = 512
    n_blocks = SR // block  # one second of audio
    start = time.perf_counter()
    for _ in range(n_blocks):
        engine.render_block(block)
    elapsed = time.perf_counter() - start
    print(f"\n{n_partials} partials x 3 voices: 1 s audio in {elapsed:.3f} s "
          f"({1.0 / elapsed:.1f}x realtime)")
    assert elapsed < 1.0, f"not realtime: {elapsed:.3f}s per audio second"


def test_engine_cost_scales_linearly_with_partial_count():
    def cost(n_partials):
        partials = [PartialSpec(ratio=float(k), amplitude=1.0 / k, decay=0.5)
                    for k in range(1, n_partials + 1)]
        cfg = TimbreConfig(name="p", base_frequency=110.0, partials=partials, sample_rate=SR)
        engine = AdditiveEngine(cfg)
        engine.note_on()
        start = time.perf_counter()
        engine.render_samples(SR // 4, block_size=256)
        return time.perf_counter() - start

    c16, c128 = cost(16), cost(128)
    print(f"\n16 partials: {c16 * 1000:.1f} ms/quarter-second; 128: {c128 * 1000:.1f} ms")
    # Allow generous headroom on shared CI machines, but cost must grow sanely.
    assert c128 < c16 * 25

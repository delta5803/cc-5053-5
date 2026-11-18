"""End-to-end tests for the HTTP session/API layer and offline bounces."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from additive.config import PartialSpec, TimbreConfig
from additive.web.server import Session, serve
from additive.render import NoteEvent, render_timeline
from additive.audio_io import export_wav
from additive.analysis import find_peaks

SR = 48000


def make_config(**kwargs):
    partials = [
        PartialSpec(ratio=1.0, amplitude=1.0, phase=0.0, decay=0.5),
        PartialSpec(ratio=2.0, amplitude=0.5, phase=math.pi / 2, decay=math.inf),
        PartialSpec(ratio=3.0, amplitude=0.25, phase=math.pi, decay=0.2),
    ]
    defaults = dict(name="api", base_frequency=220.0, partials=partials,
                    master_amplitude=0.4, transition_time=0.02, sample_rate=SR)
    defaults.update(kwargs)
    return TimbreConfig(**defaults)


def test_session_state_reports_bandlimit_and_solo():
    session = Session(make_config())
    state = session.state()
    assert len(state["bandlimit"]) == 3
    assert all(b["action"] == "kept" for b in state["bandlimit"])
    assert state["solo_index"] is None
    session.set_solo(1)
    assert session.state()["solo_index"] == 1
    session.set_solo(None)
    assert session.state()["solo_index"] is None


def test_session_render_measures_real_peaks():
    session = Session(make_config())
    result = session.render(duration=1.0, block_size=256)
    assert result["sample_rate"] == SR
    assert len(result["samples"]) == SR
    peaks = result["peaks"]
    for target, peak in zip((220.0, 440.0, 660.0), peaks):
        assert abs(peak["frequency"] - target) <= 2.5
        assert peak["amplitude"] > 1e-3
    # Infinite decay round-trips as null.
    assert result["decays"][1]["tau"] is None


def test_session_solo_affects_render():
    session = Session(make_config())
    full = np.array(session.render(1.0, 256)["samples"])
    session.set_solo(0)
    solo = np.array(session.render(1.0, 256)["samples"])
    peaks_full = find_peaks(full, SR, np.array([220.0, 440.0, 660.0]))
    peaks_solo = find_peaks(solo, SR, np.array([220.0, 440.0, 660.0]))
    assert peaks_full[1].amplitude > 0.05
    # Residual energy at a muted partial is only FFT-window sidelobe leakage.
    assert peaks_solo[1].amplitude < peaks_full[1].amplitude * 1e-3
    assert peaks_solo[0].amplitude > 0.1


def test_session_partial_update_and_validation():
    session = Session(make_config())
    session.update_partial(2, {"amplitude": 0.0})
    assert session.config.partials[2].amplitude == 0.0
    with pytest.raises(IndexError):
        session.update_partial(99, {"amplitude": 1.0})
    with pytest.raises(ValueError):
        session.update_partial(0, {"amplitude": -2.0})


def test_session_master_transition_and_bandlimit():
    session = Session(make_config())
    session.set_master(0.3, 0.05, None)
    assert session.config.master_amplitude == 0.3
    assert session.config.transition_time == 0.05
    session.set_master(None, None, "attenuate")
    assert session.config.bandlimit_mode == "attenuate"
    with pytest.raises(ValueError):
        session.set_master(None, None, "illegal")


def test_session_note_lifecycle():
    session = Session(make_config())
    nid = session.note_on(freq_multiplier=1.0, note_gain=1.0)
    assert nid in session.engine.voices
    session.note_off(nid)
    assert session.engine.voices[nid].released_at is not None
    session.note_on()
    session.note_off(None)
    assert all(v.released_at is not None for v in session.engine.voices.values())


def test_http_endpoints_roundtrip():
    httpd = serve(make_config(), host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    import threading
    from urllib import request, error

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with request.urlopen(base + "/api/state") as resp:
            state = json.loads(resp.read())
        assert len(state["config"]["partials"]) == 3

        req = request.Request(base + "/api/solo", data=json.dumps({"index": 2}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req) as resp:
            assert json.loads(resp.read())["solo_index"] == 2

        # Invalid payload must produce a clean 400, not a 500 crash.
        bad = request.Request(base + "/api/load",
                              data=json.dumps({"config": {"name": "x"}}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(error.HTTPError) as exc:
            request.urlopen(bad)
        assert exc.value.code == 400
        assert "partials" in json.loads(exc.value.read())["error"]

        req = request.Request(base + "/api/render",
                              data=json.dumps({"duration": 0.5}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req) as resp:
            rendered = json.loads(resp.read())
        assert len(rendered["samples"]) == SR // 2
    finally:
        httpd.shutdown()


def test_timeline_block_size_independent():
    cfg = make_config()
    events = [NoteEvent(0.0, 0.4, 1.0), NoteEvent(0.2, 0.5, 1.5)]
    a = render_timeline(cfg, events, total_duration=1.0, block_size=64)
    b = render_timeline(cfg, events, total_duration=1.0, block_size=1000)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-12)


def test_inf_decay_json_roundtrip(tmp_path):
    cfg = make_config()
    path = cfg.save(tmp_path / "c.json")
    raw = json.loads(path.read_text())
    assert raw["partials"][1]["decay"] == "inf"
    loaded = TimbreConfig.load(path)
    assert math.isinf(loaded.partials[1].decay)


def test_exported_wav_matches_engine_samples(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    cfg = make_config()
    audio = render_timeline(cfg, [NoteEvent(0.0, 0.3)], total_duration=0.5)
    path = export_wav(tmp_path / "bounce.wav", audio, SR, subtype="FLOAT")
    data, sr = soundfile.read(str(path))
    assert sr == SR
    assert np.allclose(data, audio, atol=1e-6)

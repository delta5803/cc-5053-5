"""Browser interface backend.

A dependency-free HTTP server (standard library only) exposes the additive
engine as a small JSON API. The browser owns transport; every sample is still
computed by the same :mod:`additive.engine` oscillators used offline.

Endpoints
---------
GET  /                 static single-page UI
GET  /api/state        current config, per-partial band-limit actions, solo
POST /api/load         load a config from a JSON payload
POST /api/save         return the current config as a JSON file
POST /api/partial      update one partial (amplitude / phase / envelope)
POST /api/master       master gain + transition time
POST /api/solo         solo one partial or restore the full mix
POST /api/note/on      start a note instance, returns its id
POST /api/note/off     release one note (or all)
POST /api/render       bounce a note and return float32 PCM + measurements
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..analysis import find_peaks, fit_decay_tau
from ..config import PartialSpec, TimbreConfig
from ..engine import AdditiveEngine

STATIC_DIR = Path(__file__).parent / "static"


class Session:
    """Holds the one shared engine behind the browser UI."""

    def __init__(self, config: TimbreConfig) -> None:
        self.lock = threading.Lock()
        self.config = config
        self.engine = AdditiveEngine(config)

    def replace_config(self, config: TimbreConfig) -> None:
        with self.lock:
            self.config = config
            self.engine = AdditiveEngine(config)

    # -- state ---------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        with self.lock:
            reports = self.engine.bandlimit_report()
            return {
                "config": self.config.to_dict(),
                "bandlimit": [
                    {
                        "index": r.index,
                        "ratio": r.ratio,
                        "frequency": r.frequency,
                        "action": r.action,
                        "gain_scale": r.gain_scale,
                        "reason": r.reason,
                    }
                    for r in reports
                ],
                "solo_index": self.engine.solo_index,
                "active_voices": sorted(self.engine.voices),
            }

    # -- mutations -----------------------------------------------------------

    def update_partial(self, index: int, changes: Dict[str, Any]) -> None:
        with self.lock:
            if not 0 <= index < len(self.config.partials):
                raise IndexError("partial index out of range")
            spec = self.config.partials[index]
            values = {
                "ratio": spec.ratio,
                "amplitude": spec.amplitude,
                "phase": spec.phase,
                "attack": spec.attack,
                "decay": spec.decay,
                "release": spec.release,
                "label": spec.label,
            }
            for key in ("amplitude", "phase", "attack", "release", "ratio"):
                if key in changes:
                    values[key] = float(changes[key])
            if "decay" in changes:
                value = changes["decay"]
                values["decay"] = (
                    math.inf if isinstance(value, str) and value.strip().lower() in ("inf", "infinity")
                    else float(value)
                )
            # Rebuild via PartialSpec so all validation rules are enforced.
            self.config.partials[index] = PartialSpec(**values)
            # Rebuild the engine so frequencies and band-limit decisions
            # re-evaluate; re-apply the current solo selection afterwards.
            solo = self.engine.solo_index
            self.engine = AdditiveEngine(self.config)
            if solo is not None:
                self.engine.solo_partial(solo)

    def set_master(self, master: Optional[float], transition: Optional[float],
                   bandlimit_mode: Optional[str]) -> None:
        with self.lock:
            if master is not None:
                self.config.master_amplitude = float(master)
                self.engine.set_master_gain(float(master))
            if transition is not None:
                self.engine.set_transition_time(float(transition))
                self.config.transition_time = float(transition)
            if bandlimit_mode is not None:
                if bandlimit_mode not in ("drop", "attenuate"):
                    raise ValueError("bandlimit_mode must be drop or attenuate")
                self.config.bandlimit_mode = bandlimit_mode
                self.engine = AdditiveEngine(self.config)

    def set_solo(self, index: Optional[int]) -> None:
        with self.lock:
            self.engine.solo_partial(index)

    def note_on(self, freq_multiplier: float = 1.0, note_gain: float = 1.0) -> int:
        with self.lock:
            return self.engine.note_on(freq_multiplier, note_gain)

    def note_off(self, note_id: Optional[int]) -> None:
        with self.lock:
            if note_id is None:
                self.engine.all_notes_off()
            else:
                self.engine.note_off(int(note_id))

    # -- offline bounce ------------------------------------------------------

    def render(self, duration: float, block_size: int) -> Dict[str, Any]:
        with self.lock:
            engine = AdditiveEngine(self.config)
            if self.engine.solo_index is not None:
                engine.solo_partial(self.engine.solo_index)
            engine.note_on()
            n = int(round(duration * self.config.sample_rate))
            audio = engine.render_samples(n, block_size)
            expected = np.array([p.ratio * self.config.base_frequency for p in self.config.partials])
            peaks = find_peaks(audio, self.config.sample_rate, expected_frequencies=expected)
            all_freqs = expected
            decays = []
            for p in self.config.partials[:12]:
                tau, r2 = fit_decay_tau(
                    audio, self.config.sample_rate, p.ratio * self.config.base_frequency,
                    fit_start=max(0.05, 2.0 * p.attack), known_frequencies=all_freqs,
                )
                decays.append({"index": len(decays), "tau": None if math.isinf(tau) else tau, "r2": r2})
            return {
                "sample_rate": self.config.sample_rate,
                "samples": audio.astype(np.float32).tolist(),
                "peaks": [{"frequency": pk.frequency, "amplitude": pk.amplitude,
                           "phase": pk.phase_radians} for pk in peaks],
                "decays": decays,
                "solo_index": self.engine.solo_index,
            }


class Handler(BaseHTTPRequestHandler):
    session: Session  # injected by serve()

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif self.path == "/api/state":
            self._send_json(self.session.state())
        elif self.path.startswith("/static/"):
            target = (STATIC_DIR / self.path.removeprefix("/static/")).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return
            ctype = "application/javascript" if target.suffix == ".js" else "text/css"
            self._serve_file(target, ctype)
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, ctype: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            data = self._read_json()
            if self.path == "/api/load":
                config = TimbreConfig.from_dict(data["config"])
                self.session.replace_config(config)
                self._send_json(self.session.state())
            elif self.path == "/api/save":
                self._send_json(self.session.config.to_dict())
            elif self.path == "/api/partial":
                self.session.update_partial(int(data["index"]), data.get("changes", {}))
                self._send_json(self.session.state())
            elif self.path == "/api/master":
                self.session.set_master(data.get("master"), data.get("transition"),
                                        data.get("bandlimit_mode"))
                self._send_json(self.session.state())
            elif self.path == "/api/solo":
                self.session.set_solo(None if data.get("index") is None else int(data["index"]))
                self._send_json(self.session.state())
            elif self.path == "/api/note/on":
                note_id = self.session.note_on(float(data.get("freq_multiplier", 1.0)),
                                                float(data.get("note_gain", 1.0)))
                self._send_json({"note_id": note_id})
            elif self.path == "/api/note/off":
                self.session.note_off(data.get("note_id"))
                self._send_json({"ok": True})
            elif self.path == "/api/render":
                result = self.session.render(float(data.get("duration", 1.5)),
                                              int(data.get("block_size", 256)))
                self._send_json(result)
            else:
                self.send_error(404)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=400)


def serve(config: TimbreConfig, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    session = Session(config)
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd

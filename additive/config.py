"""Configuration model for the additive reconstruction instrument.

A timbre is fully described by a fundamental frequency and a list of
partials. Each partial is a *real* independently generated sinusoid with its
own amplitude, initial phase and amplitude envelope (attack / exponential
decay / release). Configurations round-trip through JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Literal, Optional

BandlimitMode = Literal["drop", "attenuate"]


@dataclass
class PartialSpec:
    """One declared partial.

    ratio       partial frequency / fundamental (1.0 = fundamental, integer
                values are harmonics, fractional values allow inharmonicity).
    amplitude   linear peak amplitude (0..1 typical).
    phase       initial phase in radians at note-on, t = 0.
    attack      seconds of linear amplitude ramp after note-on.
    decay       exponential decay time constant tau in seconds: the partial
                loses 1/e of its amplitude every `decay` seconds.
    release     seconds of linear ramp to silence after note-off.
    label      optional human readable name.
    """

    ratio: float
    amplitude: float
    phase: float = 0.0
    attack: float = 0.005
    decay: float = 1.0
    release: float = 0.05
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError(f"partial ratio must be > 0, got {self.ratio!r}")
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError(f"partial amplitude must be >= 0, got {self.amplitude!r}")
        if not math.isfinite(self.phase):
            raise ValueError("partial phase must be finite")
        for name in ("attack", "release"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"partial {name} must be >= 0, got {value!r}")
        if not (math.isinf(self.decay) and self.decay > 0) and not (
            math.isfinite(self.decay) and self.decay > 0
        ):
            raise ValueError("decay time constant must be > 0 (use math.inf for no decay)")


@dataclass
class TimbreConfig:
    name: str
    base_frequency: float
    partials: List[PartialSpec]
    master_amplitude: float = 1.0
    transition_time: float = 0.02
    bandlimit_mode: BandlimitMode = "drop"
    nyquist_margin: float = 0.95
    sample_rate: int = 48000
    # Fixed hop (samples) of the chunked frequency-domain backend; its STFT
    # analysis frame is 2 * fd_hop at 50 % overlap.
    fd_hop: int = 256
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_frequency) or self.base_frequency <= 0:
            raise ValueError("base_frequency must be > 0")
        if not self.partials:
            raise ValueError("at least one partial is required")
        if not math.isfinite(self.master_amplitude) or self.master_amplitude < 0:
            raise ValueError("master_amplitude must be >= 0")
        if not math.isfinite(self.transition_time) or self.transition_time < 0:
            raise ValueError("transition_time must be >= 0")
        if self.bandlimit_mode not in ("drop", "attenuate"):
            raise ValueError("bandlimit_mode must be 'drop' or 'attenuate'")
        if not 0.0 < self.nyquist_margin <= 1.0:
            raise ValueError("nyquist_margin must be in (0, 1]")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(self.sample_rate)
        if int(self.fd_hop) <= 0:
            raise ValueError("fd_hop must be positive")
        self.fd_hop = int(self.fd_hop)
        seen: set[int] = set()
        for i, p in enumerate(self.partials):
            if not isinstance(p, PartialSpec):
                raise TypeError(f"partials[{i}] must be a PartialSpec")
            key = i
            if key in seen:
                raise ValueError("duplicate partial index")
            seen.add(key)

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    def partial_frequency(self, partial: PartialSpec) -> float:
        return partial.ratio * self.base_frequency

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # JSON has no infinity; represent +inf decay as the string "inf".
        for p in data["partials"]:
            if math.isinf(p["decay"]):
                p["decay"] = "inf"
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimbreConfig":
        required = {"name", "base_frequency", "partials"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"config missing required keys: {sorted(missing)}")
        partials: List[PartialSpec] = []
        for i, raw in enumerate(data["partials"]):
            if isinstance(raw, PartialSpec):
                partials.append(raw)
                continue
            if "ratio" not in raw:
                raise ValueError(f"partials[{i}] missing 'ratio'")
            kwargs = {
                "ratio": float(raw["ratio"]),
                "amplitude": float(raw.get("amplitude", 1.0)),
                "phase": float(raw.get("phase", 0.0)),
                "attack": float(raw.get("attack", 0.005)),
                "decay": _parse_time(raw.get("decay", 1.0), "decay"),
                "release": float(raw.get("release", 0.05)),
                "label": raw.get("label"),
            }
            partials.append(PartialSpec(**kwargs))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown config keys: {sorted(extra)}")
        return cls(
            name=str(data["name"]),
            base_frequency=float(data["base_frequency"]),
            partials=partials,
            master_amplitude=float(data.get("master_amplitude", 1.0)),
            transition_time=float(data.get("transition_time", 0.02)),
            bandlimit_mode=str(data.get("bandlimit_mode", "drop")),
            nyquist_margin=float(data.get("nyquist_margin", 0.95)),
            sample_rate=int(data.get("sample_rate", 48000)),
            fd_hop=int(data.get("fd_hop", 256)),
            description=str(data.get("description", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimbreConfig":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


def _parse_time(value: Any, field_name: str) -> float:
    if isinstance(value, str):
        if value.strip().lower() in ("inf", "infinity"):
            return math.inf
        raise ValueError(f"invalid {field_name} string {value!r}")
    return float(value)

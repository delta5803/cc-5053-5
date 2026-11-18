"""Audio export.

Writes WAV files. Uses soundfile when available (16/24/32-bit PCM and float),
otherwise falls back to the standard-library ``wave`` module (16-bit PCM).
The exported samples are exactly the engine's summed oscillator output.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Literal

import numpy as np

Subtype = Literal["PCM_16", "PCM_24", "FLOAT"]


def export_wav(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: Subtype = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float64)
    if np.max(np.abs(audio), initial=0.0) > 1.0 + 1e-9:
        # Guard against clipping rather than silently wrapping integer samples.
        raise ValueError("audio exceeds full scale (+/-1); normalize before export")
    try:
        import soundfile  # type: ignore

        soundfile.write(str(path), audio.astype(np.float32), sample_rate, subtype=subtype)
        return path
    except ImportError:
        if subtype != "PCM_16":
            raise RuntimeError("24-bit/float export needs the soundfile package")
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return path

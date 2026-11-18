"""Command line interface for the additive instrument."""

from __future__ import annotations

import argparse
import math
import sys
from typing import List

import numpy as np

from .analysis import find_peaks, fit_decay_tau
from .audio_io import export_wav
from .config import PartialSpec, TimbreConfig
from .engine import AdditiveEngine
from .render import NoteEvent, normalize, render_timeline


def _preset_few_partials(sample_rate: int) -> TimbreConfig:
    # A small, hand-checkable bell-like tone: 4 known partials.
    partials = [
        PartialSpec(ratio=1.0, amplitude=1.0, phase=0.0, attack=0.005, decay=0.80, release=0.10, label="fundamental"),
        PartialSpec(ratio=2.0, amplitude=0.60, phase=math.pi / 2, attack=0.005, decay=0.45, release=0.08, label="2nd"),
        PartialSpec(ratio=3.0, amplitude=0.35, phase=math.pi, attack=0.005, decay=0.25, release=0.06, label="3rd"),
        PartialSpec(ratio=4.2, amplitude=0.18, phase=-math.pi / 2, attack=0.005, decay=0.15, release=0.05, label="inharmonic 4.2"),
    ]
    return TimbreConfig(
        name="four-partial-bell",
        base_frequency=220.0,
        partials=partials,
        # Partial peaks sum to 2.13; keep the unnormalized bus below full scale.
        master_amplitude=0.4,
        transition_time=0.02,
        bandlimit_mode="drop",
        sample_rate=sample_rate,
        description="Four hand-specified partials (one inharmonic) for verification.",
    )


def _preset_many_partials(sample_rate: int, count: int = 64) -> TimbreConfig:
    # Dense harmonic series with a spectral roll-off; exercises real cost.
    partials: List[PartialSpec] = []
    for k in range(1, count + 1):
        partials.append(
            PartialSpec(
                ratio=float(k),
                amplitude=1.0 / (k ** 0.9),
                phase=(0.37 * k * k) % (2 * math.pi),
                attack=0.003 + 0.0005 * k,
                decay=max(0.08, 1.2 / (1.0 + 0.08 * k)),
                release=0.06,
                label=f"h{k}",
            )
        )
    return TimbreConfig(
        name=f"dense-{count}-harmonics",
        base_frequency=110.0,
        partials=partials,
        master_amplitude=0.35,
        transition_time=0.02,
        bandlimit_mode="attenuate",
        sample_rate=sample_rate,
        description=f"{count} real summed oscillators for performance testing.",
    )


def cmd_new(args: argparse.Namespace) -> int:
    if args.preset == "few":
        config = _preset_few_partials(args.sample_rate)
    else:
        config = _preset_many_partials(args.sample_rate, args.count)
    path = config.save(args.output)
    print(f"saved {path} ({len(config.partials)} partials)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    config = TimbreConfig.load(args.config)
    print(f"timbre: {config.name}  f0={config.base_frequency} Hz  sr={config.sample_rate}")
    print(f"master={config.master_amplitude}  transition={config.transition_time}s  "
          f"bandlimit={config.bandlimit_mode}  margin={config.nyquist_margin}")
    print(f"{'#':>3} {'ratio':>8} {'freq Hz':>12} {'amp':>7} {'phase':>8} {'tau s':>8}  action")
    engine = AdditiveEngine(config)
    for i, (p, report) in enumerate(zip(config.partials, engine.bandlimit_report())):
        tau = "inf" if math.isinf(p.decay) else f"{p.decay:8.3f}"
        print(f"{i:>3} {p.ratio:>8.3f} {report.frequency:>12.3f} {p.amplitude:>7.4f} "
              f"{p.phase:>8.3f} {tau:>8}  {report.action} ({report.gain_scale:.3f}) {report.reason}")
    dropped = [r for r in engine.bandlimit_report() if r.action == "dropped"]
    attenuated = [r for r in engine.bandlimit_report() if r.action == "attenuated"]
    print(f"summary: kept={len(config.partials) - len(dropped) - len(attenuated)} "
          f"attenuated={len(attenuated)} dropped={len(dropped)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    config = TimbreConfig.load(args.config)
    if args.sample_rate:
        config.sample_rate = args.sample_rate
    if args.bandlimit:
        config.bandlimit_mode = args.bandlimit
    events = [NoteEvent(start=0.0, duration=args.duration,
                        freq_multiplier=args.freq_mult, note_gain=args.note_gain)]
    tail = max(p.release for p in config.partials) + 0.05
    audio = render_timeline(config, events, total_duration=args.duration + tail, block_size=args.block_size)
    if args.normalize:
        audio = normalize(audio)
    path = export_wav(args.output, audio, config.sample_rate, subtype=args.subtype)
    print(f"rendered {audio.size} samples ({audio.size / config.sample_rate:.3f} s) -> {path}")

    peaks = find_peaks(audio, config.sample_rate,
                       expected_frequencies=np.array(
                           [p.ratio * config.base_frequency * args.freq_mult for p in config.partials]
                       ))
    print("measured spectral peaks (from rendered audio):")
    for i, (p, peak) in enumerate(zip(config.partials, peaks)):
        print(f"  partial {i:>3} f={peak.frequency:9.3f} Hz  amp={peak.amplitude:.5f}  phase={peak.phase_radians:+.3f}")
    return 0


def cmd_solo_compare(args: argparse.Namespace) -> int:
    """Render the full mix and one soloed partial, so they can be A/B heard."""
    config = TimbreConfig.load(args.config)
    if not 0 <= args.index < len(config.partials):
        print(f"index out of range (0..{len(config.partials) - 1})", file=sys.stderr)
        return 2
    tail = max(p.release for p in config.partials) + 0.05
    total = args.duration + tail

    def bounce(solo: int | None) -> np.ndarray:
        engine = AdditiveEngine(config)
        if solo is not None:
            engine.solo_partial(solo)
        engine.note_on()
        audio = engine.render_samples(int(round(total * config.sample_rate)), args.block_size)
        engine.note_off(0)
        audio2 = engine.render_samples(int(round(tail * config.sample_rate)), args.block_size)
        return np.concatenate([audio, audio2])

    full = normalize(bounce(None))
    solo = normalize(bounce(args.index))
    full_path = export_wav(args.full_output, full, config.sample_rate)
    solo_path = export_wav(args.solo_output, solo, config.sample_rate)
    p = config.partials[args.index]
    print(f"full mix  -> {full_path}")
    print(f"solo partial {args.index} (ratio={p.ratio}, f={p.ratio * config.base_frequency:.2f} Hz) -> {solo_path}")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Render and measure spectrum positions, initial phase and decay tau."""
    config = TimbreConfig.load(args.config)
    engine = AdditiveEngine(config)
    note_id = engine.note_on()
    n = int(round(args.duration * config.sample_rate))
    audio = engine.render_samples(n, args.block_size)

    expected = np.array([p.ratio * config.base_frequency for p in config.partials])
    peaks = find_peaks(audio, config.sample_rate, expected_frequencies=expected)
    print("frequency placement check:")
    for p, peak in zip(config.partials, peaks):
        target = p.ratio * config.base_frequency
        err = peak.frequency - target
        flag = "OK " if abs(err) <= 3.0 and peak.amplitude > 1e-4 else "FAIL"
        print(f"  [{flag}] target={target:9.3f} Hz  measured={peak.frequency:9.3f} Hz  "
              f"delta={err:+.3f}  amp={peak.amplitude:.5f}")

    print("decay time-constant fit (rendered audio vs config):")
    all_freqs = np.array([p.ratio * config.base_frequency for p in config.partials])
    for i, p in enumerate(config.partials[: args.max_partials]):
        tau, r2 = fit_decay_tau(audio, config.sample_rate, p.ratio * config.base_frequency,
                                fit_start=max(0.05, 2.0 * p.attack),
                                known_frequencies=all_freqs)
        tau_str = "inf" if math.isinf(tau) else f"{tau:.3f}s"
        cfg_str = "inf" if math.isinf(p.decay) else f"{p.decay:.3f}s"
        print(f"  partial {i:>2}: measured tau={tau_str:>8}  config tau={cfg_str:>8}  R^2={r2:.4f}")

    head = audio[: int(0.002 * config.sample_rate)]
    print(f"first {head.size} sample values (initial-phase waveform): "
          f"{np.array2string(head[:8], precision=5, separator=', ')}")
    engine.note_off(note_id)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.server import serve

    config = TimbreConfig.load(args.config) if args.config else _preset_few_partials(args.sample_rate)
    httpd = serve(config, host=args.host, port=args.port)
    print(f"serving additive instrument on http://{args.host}:{args.port} (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="additive", description="Additive harmonic reconstruction instrument")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="create an example configuration")
    p_new.add_argument("preset", choices=["few", "many"])
    p_new.add_argument("-o", "--output", required=True)
    p_new.add_argument("--sample-rate", type=int, default=48000)
    p_new.add_argument("--count", type=int, default=64, help="partial count for the 'many' preset")
    p_new.set_defaults(func=cmd_new)

    p_info = sub.add_parser("info", help="show config and band-limit decisions")
    p_info.add_argument("config")
    p_info.set_defaults(func=cmd_info)

    p_render = sub.add_parser("render", help="render one note to a WAV file")
    p_render.add_argument("config")
    p_render.add_argument("-o", "--output", required=True)
    p_render.add_argument("--duration", type=float, default=2.0)
    p_render.add_argument("--freq-mult", type=float, default=1.0)
    p_render.add_argument("--note-gain", type=float, default=1.0)
    p_render.add_argument("--block-size", type=int, default=256)
    p_render.add_argument("--sample-rate", type=int, default=0)
    p_render.add_argument("--bandlimit", choices=["drop", "attenuate"], default=None)
    p_render.add_argument("--subtype", choices=["PCM_16", "PCM_24", "FLOAT"], default="PCM_24")
    p_render.add_argument("--normalize", action="store_true")
    p_render.set_defaults(func=cmd_render)

    p_solo = sub.add_parser("solo-compare", help="bounce full mix vs one soloed partial")
    p_solo.add_argument("config")
    p_solo.add_argument("index", type=int)
    p_solo.add_argument("--full-output", required=True)
    p_solo.add_argument("--solo-output", required=True)
    p_solo.add_argument("--duration", type=float, default=2.0)
    p_solo.add_argument("--block-size", type=int, default=256)
    p_solo.set_defaults(func=cmd_solo_compare)

    p_meas = sub.add_parser("measure", help="verify spectrum, phase waveform and decay on rendered audio")
    p_meas.add_argument("config")
    p_meas.add_argument("--duration", type=float, default=2.0)
    p_meas.add_argument("--block-size", type=int, default=256)
    p_meas.add_argument("--max-partials", type=int, default=8)
    p_meas.set_defaults(func=cmd_measure)

    p_serve = sub.add_parser("serve", help="run the browser interface")
    p_serve.add_argument("config", nargs="?", default=None)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--sample-rate", type=int, default=48000)
    p_serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

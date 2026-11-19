#!/usr/bin/env python3
"""Transcribe local audio with the configured MLX model (Rumble-2-Flash or Fluid Speech Pro)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "hotkey/config.json"

FLUID_ALIASES = {"fluid-speech", "fluid_speech", "fluid", "fluid-speech-pro", "fluid_speech_pro", "parakeet"}
RUMBLE_ALIASES = {"rumble-2-flash", "rumble_2_flash", "rumble"}


def read_config():
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def normalize_model(value: str | None) -> str:
    raw = str(value or "rumble-2-flash").strip().lower()
    if raw in FLUID_ALIASES:
        return "fluid-speech"
    if raw in RUMBLE_ALIASES:
        return "rumble-2-flash"
    raise argparse.ArgumentTypeError(f"Unknown model {value!r}. Expected 'rumble-2-flash' or 'fluid-speech'.")


def main():
    cfg = read_config()
    try:
        default_model = normalize_model(cfg.get("model") or cfg.get("backend"))
    except argparse.ArgumentTypeError:
        default_model = "rumble-2-flash"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio or video file supported by ffmpeg")
    parser.add_argument(
        "--model", "--backend",
        dest="model",
        type=normalize_model,
        choices=["rumble-2-flash", "fluid-speech"],
        default=default_model,
        help=f"ASR model: 'rumble-2-flash' or 'fluid-speech' (default from hotkey/config.json: {default_model})",
    )
    parser.add_argument("--language", type=str, default=None, help="Force language for Rumble-2-Flash (e.g. English, Hindi)")
    parser.add_argument("--clean", action="store_true", help="Remove fillers and repeated words")
    parser.add_argument("--json", type=Path, help="Save text, timestamps, and timings as JSON")
    parser.add_argument("--repeat", type=int, default=1, help="Runs with the same loaded model")
    args = parser.parse_args()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(args.audio)],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    active_model = args.model
    start = time.perf_counter()
    if active_model == "fluid-speech":
        from fluid_speech_mlx import load
        model = load(str(ROOT / "models/fluid-speech-pro-mlx"), bits=8, clean_mode=args.clean)
        mx.eval(model.parameters())
    else:
        from mlx_audio.stt import load as load_stt
        rcfg = cfg.get("rumble_2_flash", {}) if isinstance(cfg.get("rumble_2_flash"), dict) else {}
        rel_model = rcfg.get("model_path", "models/rumble-2-flash-8bit")
        model_path = Path(rel_model) if Path(rel_model).is_absolute() else ROOT / rel_model
        model = load_stt(str(model_path))
        mx.eval(model.parameters())
        language = args.language if args.language is not None else (rcfg.get("language") or None)
        system_prompt = "<digits><nofiller><noverbatim>" if args.clean else rcfg.get("system_prompt", "<digits><nofiller><verbatim>")
        chunk_duration = float(rcfg.get("chunk_duration", 30.0))
    load_seconds = time.perf_counter() - start

    runs = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        if active_model == "fluid-speech":
            result = model.transcribe(str(args.audio.resolve()), chunk_duration=120.0)
        else:
            result = model.generate(
                str(args.audio.resolve()),
                language=language,
                system_prompt=system_prompt,
                chunk_duration=chunk_duration,
            )
        elapsed = time.perf_counter() - start
        runs.append({"seconds": elapsed, "times_realtime": duration / elapsed})
        print(result.text.strip())
        print(f"{duration:.2f}s audio in {elapsed:.2f}s ({duration / elapsed:.1f}x real time)", file=sys.stderr)

    if active_model == "fluid-speech":
        output = {
            "model_key": "fluid-speech",
            "model": "FluidVoice-Labs/fluid-speech-pro-mlx",
            "revision": "aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0",
            "bits": 8, "clean_mode": args.clean, "audio": str(args.audio.resolve()),
            "audio_seconds": duration, "model_load_seconds": load_seconds,
            "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
            "runs": runs, **asdict(result),
        }
    else:
        output = {
            "model_key": "rumble-2-flash",
            "model": "TypeWhisper/Rumble-2-Flash-8bit",
            "base_model": "Reson8/Rumble-2-Flash",
            "bits": 8, "clean_mode": args.clean, "system_prompt": system_prompt,
            "audio": str(args.audio.resolve()),
            "audio_seconds": duration, "model_load_seconds": load_seconds,
            "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
            "runs": runs,
            "text": result.text.strip(),
            "language": getattr(result, "language", None),
            "segments": getattr(result, "segments", None),
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

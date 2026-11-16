#!/usr/bin/env python3
"""Transcribe local audio with the pinned Fluid Speech Pro MLX weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import mlx.core as mx

from fluid_speech_mlx import load

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio or video file supported by ffmpeg")
    parser.add_argument("--clean", action="store_true", help="Remove fillers and repeated words")
    parser.add_argument("--json", type=Path, help="Save text, timestamps, and timings as JSON")
    parser.add_argument("--repeat", type=int, default=1, help="Runs with the same loaded model")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not mx.metal.is_available():
        parser.error("MLX Metal GPU support is unavailable")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(args.audio)],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    start = time.perf_counter()
    model = load(str(ROOT / "models/fluid-speech-pro-mlx"), bits=8, clean_mode=args.clean)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - start
    runs = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = model.transcribe(str(args.audio.resolve()), chunk_duration=120.0)
        elapsed = time.perf_counter() - start
        runs.append({"seconds": elapsed, "times_realtime": duration / elapsed})
        print(result.text)
        print(f"{duration:.2f}s audio in {elapsed:.2f}s ({duration / elapsed:.1f}x real time)", file=sys.stderr)
    output = {
        "model": "FluidVoice-Labs/fluid-speech-pro-mlx",
        "revision": "aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0",
        "bits": 8, "clean_mode": args.clean, "audio": str(args.audio.resolve()),
        "audio_seconds": duration, "model_load_seconds": load_seconds,
        "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
        "runs": runs, **asdict(result),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""MLX inference child; parent terminates it to release idle model memory."""
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def emit(**message):
    print(json.dumps(message), flush=True)

try:
    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from fluid_speech_mlx import load
        from vocabulary_boost import decoding_config
        model = load(str(ROOT / 'models/fluid-speech-pro-mlx'), bits=8)
        mx.eval(model.parameters())
        mx.set_cache_limit(128 * 1024 * 1024)
    emit(status='ready')
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            audio = Path(request['path'])
            samples, rate = sf.read(audio)
            duration = len(samples) / rate if rate else 0.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
            if duration < 0.3 or (duration < 0.5 and rms < 0.001):
                emit(status='result', text='', seconds=0.0, audio_seconds=duration, rms=rms)
                continue
            if rms < 0.001:
                raise ValueError('No audible speech. Check the selected microphone.')
            start = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                config, boost_info = decoding_config(model)
                result = model.transcribe(str(audio), decoding_config=config, chunk_duration=30.0, overlap_duration=5.0)
            emit(status='result', text=result.text, seconds=time.perf_counter()-start,
                 audio_seconds=len(samples)/rate, rms=float(np.sqrt(np.mean(samples**2))), **boost_info)
        except Exception as exc:
            emit(status='error', message=str(exc))
except Exception as exc:
    emit(status='fatal', message=str(exc))
    sys.exit(1)

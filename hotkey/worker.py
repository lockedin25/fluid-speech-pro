"""Keep MLX in a child process; unload after a configurable idle timeout."""
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get('FLUID_CONFIG_PATH', ROOT / 'config.json'))
STATE = Path(os.environ.get('FLUID_MODEL_STATE_PATH', ROOT / 'model-state.json'))
child = None
last_completed = None


def emit(**message):
    print(json.dumps(message), flush=True)


def idle_seconds():
    try:
        value = float(json.loads(CONFIG.read_text())['idle_unload_seconds'])
        return value if math.isfinite(value) and 0 <= value <= 3600 else 10.0
    except (OSError, ValueError, KeyError, TypeError):
        return 10.0


def state(loaded, **extra):
    STATE.write_text(json.dumps({
        'model_loaded': loaded,
        'model_pid': child.pid if loaded and child is not None else None,
        'idle_unload_seconds': idle_seconds(),
        'updated_at': time.time(), **extra,
    }, indent=2) + '\n')


def unload():
    global child, last_completed
    process = child
    child = None
    last_completed = None
    # Do not re-enter cleanup if Quit arrives during the idle timeout's wait.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
        state(False)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def shutdown(*_):
    unload()
    sys.exit(0)


def read_child(timeout):
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError('Model process timed out')
    line = child.stdout.readline()
    if not line:
        raise RuntimeError('Model process stopped unexpectedly')
    return json.loads(line)


def ensure_loaded():
    global child
    if child is not None and child.poll() is None:
        return
    unload()
    child = subprocess.Popen(
        [sys.executable, '-u', str(ROOT / 'inference.py')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    message = read_child(45)
    if message.get('status') != 'ready':
        raise RuntimeError(message.get('message', 'Model failed to load'))
    state(True)


def handle(line):
    global last_completed
    started = time.perf_counter()
    try:
        request = json.loads(line)
        if not isinstance(request.get('path'), str):
            raise ValueError('Expected an audio file path')
        ensure_loaded()
        child.stdin.write(json.dumps(request) + '\n')
        child.stdin.flush()
        result = read_child(150)
        result['request_seconds'] = time.perf_counter() - started
        result['model_pid'] = child.pid
        last_completed = time.monotonic()
        state(True, last_request_seconds=result['request_seconds'])
        emit(**result)
    except Exception as exc:
        unload()
        emit(status='error', message=str(exc))


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
state(False)
emit(status='ready')
buffer = b''
try:
    with selectors.DefaultSelector() as selector:
        selector.register(sys.stdin, selectors.EVENT_READ)
        while True:
            timeout = 1.0
            if child is not None and last_completed is not None:
                remaining = idle_seconds() - (time.monotonic() - last_completed)
                if remaining <= 0:
                    unload()
                else:
                    timeout = min(timeout, remaining)
            if not selector.select(timeout):
                continue
            data = os.read(sys.stdin.fileno(), 65536)
            if not data:
                break
            buffer += data
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                handle(line)
finally:
    unload()

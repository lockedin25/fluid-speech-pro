"""Measure physical footprint while transcribing five minutes of test audio."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

root = Path(__file__).resolve().parents[1]
points = []
with tempfile.TemporaryDirectory(prefix='fluid-memory-test-') as tmp:
    tmp = Path(tmp)
    audio = tmp/'five-minutes.wav'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-stream_loop','-1','-i',str(root/'samples/recorded-speech.flac'),'-t','300','-ar','16000','-ac','1',str(audio)],check=True)
    config, state = tmp/'config.json', tmp/'state.json'
    config.write_text(json.dumps(dict(json.loads((root/'hotkey/config.json').read_text()), idle_unload_seconds=0)))
    env = dict(os.environ, HF_HUB_OFFLINE='1', FLUID_CONFIG_PATH=str(config), FLUID_MODEL_STATE_PATH=str(state))
    process = subprocess.Popen([str(root/'.venv/bin/python'),'-u',str(root/'hotkey/worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    def sample():
        while process.poll() is None:
            try:
                pid = json.loads(state.read_text()).get('model_pid')
                if pid:
                    result = subprocess.run(['footprint','--noCategories','-p',str(pid)],capture_output=True,text=True)
                    points.extend(float(x) for x in re.findall(r'phys_footprint_peak:\s+([\d.]+) MB',result.stdout))
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
    try:
        assert json.loads(process.stdout.readline())['status']=='ready'
        thread = threading.Thread(target=sample, daemon=True);thread.start()
        process.stdin.write(json.dumps({'path':str(audio)})+'\n');process.stdin.flush()
        result = json.loads(process.stdout.readline());assert result['status']=='result', result
        report = {k:v for k,v in result.items() if k!='text'}
        report['observed_peak_model_footprint_mb'] = max(points) if points else None
        report['audio_file_bytes'] = audio.stat().st_size
        report['chunk_duration_seconds'] = 30
        report['overlap_seconds'] = 5
        (root/'hotkey/long-audio-verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        process.terminate();process.wait(timeout=8)
        thread.join(timeout=2)
        assert process.returncode == 0, process.returncode

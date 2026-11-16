"""Compare greedy, unboosted beam and production boosts on controlled audio."""
import json
from pathlib import Path
import sys
import tempfile
import time

import mlx.core as mx
from parakeet_mlx.parakeet import Beam, DecodingConfig, Greedy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fluid_speech_mlx import load
from vocabulary_boost import decoding_config

model = load(str(ROOT/'models/fluid-speech-pro-mlx'), bits=8)
mx.eval(model.parameters())
mx.set_cache_limit(128*1024*1024)
results = []
with tempfile.TemporaryDirectory(prefix='fluid-boost-check-') as tmp:
    cfg = Path(tmp)/'config.json'
    settings = {'vocabulary_boost': {'enabled': True, 'beam_size': 2, 'terms_file': 'vocabulary/active-boost.json'}}
    cfg.write_text(json.dumps(settings))
    boosted, metadata = decoding_config(model, cfg)
    assert metadata['boost_enabled'] and metadata['boost_term_count'] == 186
    assert decoding_config(model, cfg)[0] is boosted, 'Decoder config cache failed'
    for filename in ['smoke-test.wav', 'recorded-speech.flac', 'vocabulary-check.wav', 'common-words-check.wav']:
        by_mode = {}
        for mode, config in [('greedy', DecodingConfig(decoding=Greedy())), ('beam-unboosted', DecodingConfig(decoding=Beam(beam_size=2))), ('boosted', boosted)]:
            started = time.perf_counter()
            text = model.transcribe(str(ROOT/'samples'/filename), decoding_config=config, chunk_duration=30, overlap_duration=5).text
            row = {'audio': filename, 'mode': mode, 'seconds': time.perf_counter()-started, 'text': text}
            by_mode[mode] = text
            results.append(row)
            print(json.dumps(row), flush=True)
        if filename == 'smoke-test.wav':
            assert 'Rahul' in by_mode['boosted'], by_mode
            assert 'Rahul' not in by_mode['beam-unboosted'], by_mode
        if filename == 'recorded-speech.flac':
            assert by_mode['boosted'] == by_mode['greedy'], by_mode
        if filename == 'common-words-check.wav':
            # The negative control should not grow any targeted names.
            for term in ['Shubham', 'Rahul', 'Span AI', 'Qwen', 'LoRA', 'Sefer']:
                assert term.casefold() not in by_mode['boosted'].casefold(), by_mode
    for invalid in ['[]', '{"vocabulary_boost": []}', '{"vocabulary_boost": {"enabled": true, "beam_size": 99}}', '{']:
        cfg.write_text(invalid)
        fallback, metadata = decoding_config(model, cfg)
        assert isinstance(fallback.decoding, Greedy) and not metadata['boost_enabled'], metadata
    cfg.write_text('{"vocabulary_boost": {"enabled": false}}')
    assert isinstance(decoding_config(model, cfg)[0].decoding, Greedy)
(ROOT/'results/vocabulary-boost-verification.json').write_text(json.dumps({'results': results, 'checks': ['boost improves Rahul relative to unboosted beam', 'recorded negative control unchanged', 'no target names inserted into common-word control', 'cached config reused', 'invalid config falls back to greedy', 'disabled setting restores greedy']}, indent=2)+'\n')
print('PASS: vocabulary-boost comparisons and fallback checks', flush=True)

"""Read the local vocabulary profile and construct a cached decoder config."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_cached_key = None
_cached_config = None


def decoding_config(model, config_path=None):
    from parakeet_mlx import boost
    from parakeet_mlx.parakeet import Beam, DecodingConfig, Greedy

    global _cached_key, _cached_config
    config_path = Path(config_path or os.environ.get('FLUID_CONFIG_PATH', ROOT/'hotkey/config.json'))
    try:
        contents = json.loads(config_path.read_text())
        if not isinstance(contents, dict):
            raise ValueError('Config must be a JSON object')
        settings = contents.get('vocabulary_boost', {})
        if not isinstance(settings, dict):
            raise ValueError('vocabulary_boost must be a JSON object')
        if settings.get('enabled', False) is not True:
            return DecodingConfig(decoding=Greedy()), {'boost_enabled': False, 'boost_term_count': 0}
        size = settings.get('beam_size', 2)
        if type(size) is not int or not 1 <= size <= 5:
            raise ValueError('beam_size must be an integer from 1 to 5')
        profile_path = ROOT / settings.get('terms_file', 'vocabulary/active-boost.json')
        profile = json.loads(profile_path.read_text())
        weights = profile['weights']
        if not isinstance(weights, dict) or not 1 <= len(weights) <= 300:
            raise ValueError('Vocabulary must contain 1 to 300 terms')
        for term, weight in weights.items():
            if not isinstance(term, str) or not term.strip() or len(term) > 100 or '@' in term or any(ord(c) < 32 for c in term):
                raise ValueError('Vocabulary contains an invalid term')
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 < weight <= 2:
                raise ValueError('Boost weights must be finite numbers above 0 and at most 2')
        signature = (id(model), size, tuple(sorted(weights.items())))
        if signature != _cached_key:
            # Recognition bias should not depend on sentence capitalization.
            folded = {}
            for term, weight in weights.items():
                folded[term.casefold()] = max(weight, folded.get(term.casefold(), 0))
            machine = boost.lift(boost.prebind(str.casefold, boost.words(folded)), model.vocabulary)
            _cached_config = DecodingConfig(decoding=Beam(beam_size=size, booster=machine))
            _cached_key = signature
        return _cached_config, {'boost_enabled': True, 'boost_term_count': len(weights), 'beam_size': size}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A malformed user-edited profile must not disable ordinary dictation.
        print(f'Vocabulary boost disabled: {exc}', file=sys.stderr, flush=True)
        return DecodingConfig(decoding=Greedy()), {'boost_enabled': False, 'boost_term_count': 0, 'boost_error': str(exc)}

---
license: cc-by-4.0
language:
- bg
- hr
- cs
- da
- nl
- en
- et
- fi
- fr
- de
- el
- hu
- it
- lv
- lt
- mt
- pl
- pt
- ro
- sk
- sl
- es
- sv
- ru
- uk
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model:
- FluidVoice-Labs/fluid-speech-pro
tags:
- mlx
- parakeet-mlx
- parakeet
- tdt
- fastconformer
- apple-silicon
- automatic-speech-recognition
---

# 🎙️ Fluid Speech Pro (MLX)

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-0.63B-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-25_European_Languages-blue#model-badge)](#languages)
| [![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey#model-badge)](#license)

[Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) for Apple Silicon, running on
[parakeet-mlx](https://github.com/senstella/parakeet-mlx). Speech-to-text for English and 24 other European languages,
with punctuation and capitals.

## All versions

| Version | Best for | Clean mode |
|:---|:---|:--|
| [NeMo + 🤗 Transformers](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) | NVIDIA GPUs and servers; the reference model | Yes (NeMo) |
| [ONNX](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-onnx) | CPUs, with onnx-asr or sherpa-onnx | Yes |
| **[MLX](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx)** (this one) | Apple Silicon Macs, with parakeet-mlx | Yes |
| [GGUF](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-gguf) | C++ with parakeet.cpp, CPU or GPU, no Python | No |
| [Core ML](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-coreml) | Apple apps in Swift, on the Neural Engine, with FluidAudio | Not yet |

## Description

Fluid Speech Pro is a 0.63-billion-parameter speech recognition model for fast, accurate speech-to-text, from the
team behind [FluidVoice](https://fluidvoice.com). It is built on NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and keeps its 25 European languages.
It detects the language on its own, with no prompt needed.

It is the #2 open model on the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), with a 6.06 average word error
rate. It is the most accurate open model under 1B parameters, #2 among open models on noisy speech, and it runs at
about 3,700× real time on one GPU.

This model is ready for commercial and non-commercial use.

## Key features

- **Accurate:** #2 open model on the Open ASR Leaderboard (6.06 average WER)
- **Best under 1B:** 10% fewer errors than the next open model under 1B parameters
- **Strong in noise:** #2 open model on URGENT 2024, the leaderboard's noisy-speech set (9.83 WER)
- **Fast:** about 3,700× real time on one GPU, so an hour of audio takes about a second
- **Punctuation and capitalization** included
- **Word- and segment-level timestamps** (NeMo)
- **Clean mode** (optional): edited text without repeated words and filler words (NeMo, ONNX and MLX)
- **25 European languages**, detected automatically
- **Five versions:** NeMo and 🤗 Transformers, ONNX, MLX, GGUF and Core ML
- **Permissive license:** CC BY 4.0

## Performance

![Accuracy against size for open models on the Open ASR Leaderboard](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/resolve/main/assets/accuracy_vs_size.png)

*Figure 1: Average word error rate (up is better) against model size for open models on the Open ASR Leaderboard.
Fluid Speech Pro is the only top-5 open model under 1B parameters.*

![The most accurate open model under 1B parameters](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/resolve/main/assets/best_under_1b.png)

*Figure 2: The best open models at 1B parameters and under, ranked by average word error rate (lower is better).*

![Second among open models on noisy speech](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/resolve/main/assets/noise_urgent24.png)

*Figure 3: Word error rate on URGENT 2024, the leaderboard's noisy-speech set (lower is better).*

## Quick start

```bash
pip install parakeet-mlx          # reads audio through ffmpeg: brew install ffmpeg
hf download FluidVoice-Labs/fluid-speech-pro-mlx fluid_speech_mlx.py --local-dir .
```

```python
from fluid_speech_mlx import load

model = load("FluidVoice-Labs/fluid-speech-pro-mlx", bits=8)
print(model.transcribe("audio.wav").text)
```

Leave out `bits=8` for the full weights.

## Clean mode

Clean mode drops filler words ("um", "uh") and repeated words ("I I think"). It's a small add-on (55 MB) that the
loader applies on top of either set of weights:

```python
clean = load("FluidVoice-Labs/fluid-speech-pro-mlx", bits=8, clean_mode=True)
```

## Why the small loader

Fluid Speech Pro is Parakeet TDT 0.6B v3 plus one small block between the encoder's subsampling and its first
Conformer layer. parakeet-mlx doesn't know that block, so a plain `parakeet_mlx.from_pretrained` refuses these weights
instead of running without it. `fluid_speech_mlx.py` (one file) adds the block, and computes the audio features
exactly the way NeMo does. With parakeet-mlx's own features, 1 in 4 transcripts came out different in our check.

Everything else is parakeet-mlx: `transcribe` returns its usual result with sentence and token timestamps, and
`chunk_duration` works for long files.

## Files

| File | What | Download |
|:---|:---|--:|
| `model.8bit.safetensors` | 8-bit weights | 751 MB |
| `model.safetensors` | fp32 weights | 2.52 GB |
| `clean-mode.safetensors` | clean mode add-on, for either | 55 MB |
| `config.json`, `fluid_speech_mlx.py` | | |

`load()` downloads only the files it needs. Compute runs in bfloat16 by default; pass `dtype=mx.float32` for full
precision.

## Accuracy of this version

400 clips from 8 public test sets (AMI, Earnings-22, GigaSpeech, LibriSpeech clean and other, Monsoon, SPGISpeech,
VoxPopuli), compared with the NeMo model. All runs decode greedily.

| Run | Same text as NeMo | WER |
|:---|--:|--:|
| NeMo (reference) | 400 / 400 | 3.53 |
| MLX, fp32 | 398 / 400 | 3.53 |
| MLX, default (bfloat16) | 390 / 400 | 3.50 |
| MLX, 8-bit | 354 / 400 | 3.56 |

Clean mode against NeMo's clean mode (WER 3.98): fp32 394 / 400 the same, WER 4.00; 8-bit 341 / 400, WER 3.97.

WER is over the 350 clips from the 7 sets with per-clip references (Earnings-22's references are whole calls).

## Good to know

- Speed on an Apple M5 Max, one clip at a time: fp32 about 128× real time, 8-bit about 109×.
- Offline transcription only: the extra block looks a few frames ahead, so `transcribe_stream` isn't supported.
- The reference checkpoint is [the NeMo version](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro).

## Use case

For developers, researchers and companies building anything that turns speech into text: dictation, meeting notes,
captions and subtitles, voice assistants, call and media analytics, and transcription services.

## Release date

Hugging Face, November 2026.

## Model architecture

**Architecture type:** FastConformer-TDT

**Network architecture:**

- A FastConformer encoder and a TDT (token-and-duration transducer) decoder, as in Parakeet TDT 0.6B v3
- One small extra block between the encoder's subsampling and its first Conformer layer
- 0.63 billion parameters

## Input

- **Input type:** audio, 16 kHz, mono
- **Input format:** `.wav`, `.flac` and other formats your audio loader reads

## Output

- **Output type:** text, with punctuation and capitalization
- **Optional:** word and segment timestamps (NeMo)

## Model version

Fluid Speech Pro 1.0

## Evaluation datasets

The Open ASR Leaderboard's public test sets: AMI (cleaned), Earnings-22 (cleaned, chunked), GigaSpeech (cleaned),
SPGISpeech, URGENT 2024 (noisy speech), Voice Arena Monsoon (Indian English) and VoxPopuli (cleaned).

## Results

The leaderboard numbers below are for the main NeMo model with 4-beam search. This MLX version runs the same model, extra block included; see [Accuracy of this version](#accuracy-of-this-version) for how closely it matches.

### Open ASR Leaderboard

Open models on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): average word
error rate over its seven public and two private test sets (lower is better):

| Model | Parameters | Average WER |
|:---|--:|--:|
| Musci-ASR | 2B | 5.70 |
| **Fluid Speech Pro** | **0.63B** | **6.06** |
| NVIDIA Nemotron 3.5 Transcribe | 30B | 6.13 |
| Qwen3-ASR | 1.7B | 6.15 |
| IBM Granite Speech 5.0 | 2B | 6.68 |
| Cohere Transcribe | 2B | 6.72 |
| NVIDIA Parakeet TDT 0.6B v2 | 0.6B | 6.93 |
| NVIDIA Parakeet TDT 0.6B v3 | 0.6B | 7.04 |
| Fermion Research Phonon-2 | 0.6B | 7.86 |

### Results by test set

Word error rate (%) on the leaderboard's seven public test sets:

| Model | Public average | AMI | Earnings-22 | GigaSpeech | SPGISpeech | URGENT 2024 | Monsoon | VoxPopuli |
|:---|--:|--:|--:|--:|--:|--:|--:|--:|
| Musci-ASR | 5.15 | 6.53 | 5.83 | 6.41 | 1.86 | 10.49 | 3.44 | 1.47 |
| **Fluid Speech Pro** | **5.43** | **7.55** | **5.42** | **7.51** | **2.31\*** | **9.83** | **3.71** | **1.70** |
| NVIDIA Nemotron 3.5 Transcribe | 5.51 | 7.78 | 5.25 | 7.18 | 1.91 | 10.94 | 3.63 | 1.89 |
| Qwen3-ASR | 5.75 | 8.31 | 5.84 | 7.22 | 2.57 | 9.93 | 3.50 | 2.85 |
| IBM Granite Speech 5.0 | 6.05 | 6.96 | 6.22 | 8.14 | 3.36 | 10.22 | 3.94 | 3.52 |
| Cohere Transcribe | 6.26 | 7.02 | 7.88 | 7.88 | 2.74 | 9.50 | 3.44 | 5.38 |
| NVIDIA Parakeet TDT 0.6B v2 | 6.43 | 9.09 | 6.76 | 8.15 | 1.94 | 11.42 | 3.89 | 3.78 |
| NVIDIA Parakeet TDT 0.6B v3 | 6.51 | 9.42 | 5.85 | 7.99 | 3.63 | 11.52 | 4.14 | 3.05 |
| Fermion Research Phonon-2 | 7.23 | 9.32 | 6.96 | 8.35 | 3.70 | 14.90 | 4.91 | 2.46 |

\* In clean mode, which matches SPGISpeech's edited references; the leaderboard allows one transcription mode per test
set.

### Other languages

Fluid Speech Pro supports 25 languages. On the Open ASR Leaderboard's multilingual track it makes 12% fewer errors
than Parakeet TDT 0.6B v3, and it is better in all seven of the track's languages:

| Language | Parakeet TDT 0.6B v3 | Fluid Speech Pro | Fewer errors |
|:---|--:|--:|--:|
| German | 4.12 | **3.76** | 9% |
| French | 5.40 | **5.10** | 6% |
| Italian | 4.73 | **4.16** | 12% |
| Spanish | 3.71 | **3.38** | 9% |
| Portuguese | 6.08 | **4.88** | 20% |
| Dutch | 7.53 | **6.82** | 9% |
| English | 7.04 | **6.06** | 14% |
| **Average** | 5.52 | **4.88** | **12%** |

*Word error rate (%) on the [Open ASR Leaderboard's multilingual track](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard?tab=multilingual),
lower is better.*

It also got better than Parakeet TDT 0.6B v3 in every one of the 24 languages beyond English. On FLEURS, scored the
way the leaderboard scores its multilingual track, it makes nearly 20% fewer errors on average.

![Better than Parakeet TDT 0.6B v3 in all 24 other languages](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/resolve/main/assets/languages_24.png)

*Figure 4: Word error rate on FLEURS for Parakeet TDT 0.6B v3 (grey) and Fluid Speech Pro (blue), lower is better.*

| Language | Parakeet TDT 0.6B v3 | Fluid Speech Pro | Fewer errors |
|:---|--:|--:|--:|
| Estonian | 13.28 | **9.51** | 28% |
| Latvian | 21.76 | **16.39** | 25% |
| Italian | 2.48 | **1.91** | 23% |
| Lithuanian | 21.04 | **16.21** | 23% |
| Maltese | 19.23 | **15.05** | 22% |
| Hungarian | 13.58 | **10.65** | 22% |
| Slovak | 9.18 | **7.20** | 22% |
| Finnish | 11.19 | **8.82** | 21% |
| Czech | 11.06 | **8.83** | 20% |
| Portuguese | 4.47 | **3.57** | 20% |
| Slovenian | 22.48 | **17.97** | 20% |
| Ukrainian | 5.95 | **4.77** | 20% |
| Greek | 21.16 | **17.10** | 19% |
| Romanian | 11.38 | **9.28** | 18% |
| Spanish | 3.22 | **2.66** | 17% |
| Swedish | 13.39 | **11.10** | 17% |
| Polish | 6.84 | **5.70** | 17% |
| German | 4.18 | **3.52** | 16% |
| Danish | 17.20 | **14.57** | 15% |
| Bulgarian | 11.94 | **10.19** | 15% |
| Croatian | 11.40 | **9.73** | 15% |
| Dutch | 6.39 | **5.51** | 14% |
| Russian | 4.92 | **4.35** | 12% |
| French | 4.75 | **4.65** | 2% |
| **Average of all 24** | 11.35 | **9.13** | **20%** |

*Word error rate (%) on FLEURS test sets, greedy decoding.* A future version will be trained on more multilingual data.

## Limitations

- Please check the model on your own audio, accents and languages before you rely on it.
- Like every speech model, it makes more mistakes on heavy noise, overlapping speakers, strong accents it has rarely
  heard, and rare names and terms.
- Clean mode edits the text. Don't use it where you need an exact record.

## License

CC BY 4.0. If you use the model, please credit it as **Fluid Speech Pro by FluidVoice**.

Built on [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and
[Moondream Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra), both CC BY 4.0.

## Citation

```bibtex
@misc{fluidspeechpro2026,
  title  = {Fluid Speech Pro},
  author = {{FluidVoice}},
  year   = {2026},
  url    = {https://huggingface.co/FluidVoice-Labs/fluid-speech-pro}
}
```

## References

1. [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), NVIDIA
2. [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
3. [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)
4. [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), Hugging Face

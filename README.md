# Fluid Speech Pro on this Mac

Fluid Speech Pro by FluidVoice is installed for local speech-to-text on the Apple M5, 16 GB Mac. It uses `parakeet-mlx` and the official Fluid Speech loader. `mlx-lm` is for language models and is not the runtime specified for this model.

## Dictate into any app

The installed **Fluid Dictation** menu-bar app uses **Control + Option + Space**: press once to record, press again to transcribe and paste. It starts at login, uses a conservative 186-term vocabulary boost, and unloads model weights after 10 idle seconds. See [hotkey setup and permissions](hotkey/README.md).

## Transcribe your audio

```bash
cd /Users/shubhamkorde/Desktop/Projects/fluid-speech-pro
./transcribe /absolute/path/to/recording.m4a
./transcribe /absolute/path/to/recording.m4a --clean
./transcribe /absolute/path/to/recording.m4a --json results/my-recording.json
```

The launcher uses the local Python environment and weights, with `HF_HUB_OFFLINE=1`. Audio stays on this machine. WAV, M4A, MP3, FLAC and other ffmpeg-readable files work. The library resamples audio as needed; decoding uses 120-second chunks with the library's default 15-second overlap for long files.

`--clean` enables the downloaded adapter that removes fillers and repetitions. Use default mode when preserving the spoken wording matters. This model supports file transcription; its loader explicitly does not support streaming. It supports English and 24 other European languages, not Hindi.

## Verified on 2026-11-15

Both default and clean mode completed offline on the Metal GPU.

| Test | Audio length | Second pass | Speed |
|---|---:|---:|---:|
| macOS Samantha synthetic speech, default | 9.47 s | 0.41 s | 23.2x real time |
| Recorded speech, default | 10.44 s | 0.68 s | 15.4x real time |
| macOS Samantha synthetic speech, clean | 9.47 s | 0.64 s | 14.9x real time |

These timings reuse a loaded model. The command loads the model each time it starts; initialization and the first pass take longer. Results include model load time, each transcription time, timestamps and peak MLX allocation in `results/*.json`. MLX allocation is not total process or system memory.

The synthetic sentence included "Hello Shubham" and "Please send the notes to Rahul after the call." Default mode returned "Hello Shoebom" and "Rattle"; clean mode returned "Hello Shubaum" and "Rahul". Names need checking. These short samples establish that inference works; they do not establish accuracy on your own recordings.

Replay the tests:

```bash
./transcribe samples/smoke-test.wav --repeat 2 --json results/smoke-test.json
./transcribe samples/recorded-speech.flac --repeat 2 --json results/recorded-speech.json
./transcribe samples/smoke-test.wav --clean --repeat 2 --json results/smoke-test-clean.json
```

The recorded sample is [narsil/asr_dummy, 1.flac](https://huggingface.co/datasets/narsil/asr_dummy/blob/8d141c84e3f84c54cd7bbaa851d24edd0f559734/1.flac). The synthetic clip was created locally with macOS `say`, voice Samantha. No microphone recording was taken.

## Installed files and reproduction

- `models/fluid-speech-pro-mlx/model.8bit.safetensors`: 751 MB 8-bit weights.
- `models/fluid-speech-pro-mlx/clean-mode.safetensors`: 55 MB adapter.
- `fluid_speech_mlx.py`: official loader, needed for the model's extra encoder block and audio preprocessing.
- `requirements.lock.txt`: exact installed Python package versions, including `parakeet-mlx==0.7.0` and `mlx==0.32.4`.
- `SHA256SUMS`: hashes of the downloaded loader, configuration and weights.
- `MODEL_CARD.md`: upstream model card. Model license and attribution are in `models/fluid-speech-pro-mlx/LICENSE` and `NOTICE`.

Model and loader revision: `aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/hf download FluidVoice-Labs/fluid-speech-pro-mlx config.json model.8bit.safetensors clean-mode.safetensors LICENSE NOTICE --revision aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0 --local-dir models/fluid-speech-pro-mlx
shasum -a 256 -c SHA256SUMS
```

`ffmpeg` and `ffprobe` are already installed through Homebrew on this Mac. The full-precision 2.52 GB weights were not downloaded; the runner selects 8-bit weights.

Source: [official MLX model and usage instructions](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx). License: CC BY 4.0; credit "Fluid Speech Pro by FluidVoice" when using the model.

# Hinglish support without fine-tuning

Decision on 2026-11-15: skip speculative Fluid Speech Pro fine-tuning. No training, paid GPU launch or private audio upload occurred. With permission to inspect Wispr, a read-only metadata audit found 777 saved WAV recordings totalling 16043.76 seconds (about 4.46 hours). The automatic transcripts have no verified-correction marker; no dataset or audio export was created.

The proposed first test is Qwen3-ASR-0.6B through an MLX runtime. Its official model card lists Hindi and English among 30 supported languages; that makes it a candidate for mixed Hindi/English audio, not proof of good Hinglish recognition on Shubham's voice. Compare it on representative code-switched clips before changing the working app. The 1.7B model is a follow-up candidate if the smaller model is inadequate.

The community `mlx-qwen3-asr` project runs the original checkpoints locally and supports automatic language detection and domain-vocabulary context. `mlx-audio` also provides Qwen3-ASR support and MLX-community converted checkpoints. The official Qwen implementation uses PyTorch; the Mac MLX runtimes are community implementations. Neither has been installed or benchmarked as part of this check.

Preserve the existing hotkey, temporary-file cleanup, paste handling and 10-second model idle timeout when integrating a candidate engine. Do not describe the current Fluid engine as having acquired Hindi support; no engine switch has been made.

For Roman-script Hinglish, add a transliteration step for Hindi-script spans while retaining English tokens and names. AI4Bharat IndicXlit is a possible local native-to-Roman transliterator. This is script conversion, not translation into English. Natural Roman spellings and preservation of names must be tested. Shubham selected Roman-script Hinglish, for example: "Kal meeting hai, please notes bhej dena".

Sources:

- [Official Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Community MLX Qwen3-ASR runtime](https://github.com/moona3k/mlx-qwen3-asr)
- [MLX-community 0.6B 8-bit checkpoint](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit)
- [MLX Audio](https://github.com/Neywa-Labs/mlx-audio)
- [AI4Bharat IndicXlit](https://github.com/AI4Bharat/IndicXlit)

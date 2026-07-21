"""Standalone Stable Audio Open worker (run as a subprocess).

Generates audio of an exact duration from a text prompt using Stability AI's
**Stable Audio Open** via the diffusers ``StableAudioPipeline``. ONE model turns
text into BOTH sound effects and music, stereo at 44.1 kHz, up to 47 s.

  * mode "sfx"   -> environmental sound effects / ambience of the scene
  * mode "music" -> a musical score / bed
  * mode "both"  -> an SFX pass mixed over an attenuated music pass

Runs as its own process so all audio-model VRAM is reclaimed by the OS on exit.
Writes a single WAV at the model's native 44.1 kHz to --out.

Requires (installed by setup.sh): diffusers>=0.29 (StableAudioPipeline), soundfile,
sentencepiece. The model is GATED: accept the license once and have HF_TOKEN in the
environment (the studio inherits it) -> https://huggingface.co/stabilityai/stable-audio-open-1.0
diffusers/huggingface_hub read HF_TOKEN from the env automatically, so we never
pass a token on the command line.

Usage:
  python -m studio.audio_worker --mode both --prompt "..." --duration 6.5 --out a.wav
"""
from __future__ import annotations

import argparse
import sys


def _log(msg: str) -> None:
    print(f"[audio] {msg}", flush=True)


def _load_tf32():
    import torch

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:  # noqa: BLE001
        pass


def _load_pipe(model_id: str):
    """Load StableAudioPipeline on GPU (fp16) or CPU (fp32) fallback."""
    import torch
    from diffusers import StableAudioPipeline

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    _log(f"loading {model_id} ({'cuda/fp16' if use_cuda else 'cpu/fp32'})")
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda" if use_cuda else "cpu")
    return pipe


def _generate(pipe, prompt: str, negative: str, duration: float,
              steps: int, guidance: float, seed):
    """One Stable Audio pass -> (wav (channels, samples) float32 cpu, sample_rate)."""
    import torch

    gen = None
    if seed is not None:
        gen = torch.Generator("cpu").manual_seed(int(seed))
    end = float(max(1.0, min(duration, 47.0)))  # model caps at 47 s
    _log(f"generate: '{prompt[:70]}…' ({end:.1f}s, {steps} steps, guidance {guidance})")
    audios = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        num_inference_steps=int(steps),
        audio_end_in_s=end,
        num_waveforms_per_prompt=1,
        guidance_scale=float(guidance),
        generator=gen,
    ).audios
    wav = audios[0].to(torch.float32).cpu()  # (channels, samples)
    sr = int(pipe.vae.sampling_rate)
    return wav, sr


def _to_stereo(wav):
    # wav: (channels, samples) -> (2, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav[:2]


def _fit_len(wav, samples: int):
    import torch

    cur = wav.shape[-1]
    if cur == samples:
        return wav
    if cur > samples:
        return wav[..., :samples]
    return torch.nn.functional.pad(wav, (0, samples - cur))


def _save(path: str, wav, sr: int) -> None:
    """Write (channels, samples) tensor as a WAV. Prefer soundfile; torchaudio fallback."""
    try:
        import soundfile as sf

        data = wav.transpose(0, 1).contiguous().numpy()  # (samples, channels)
        sf.write(path, data, sr)
    except Exception:  # noqa: BLE001
        import torchaudio

        torchaudio.save(path, wav, sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sfx", "music", "both"], default="sfx")
    ap.add_argument("--prompt", required=True)          # SFX / scene sound prompt
    ap.add_argument("--music_prompt", default=None)     # music-bed prompt
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--negative", default="low quality, average quality, distorted")
    ap.add_argument("--music_under_db", type=float, default=-8.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    _load_tf32()
    import torch

    dur = float(args.duration)
    music_prompt = args.music_prompt or args.prompt
    pipe = _load_pipe(args.model)

    try:
        if args.mode == "sfx":
            wav, sr = _generate(pipe, args.prompt, args.negative, dur,
                                args.steps, args.guidance, args.seed)
            wav = _to_stereo(wav)
        elif args.mode == "music":
            wav, sr = _generate(pipe, music_prompt, args.negative, dur,
                                args.steps, args.guidance, args.seed)
            wav = _to_stereo(wav)
        else:  # both -> mix SFX over an attenuated music bed
            sfx, sr = _generate(pipe, args.prompt, args.negative, dur,
                                args.steps, args.guidance, args.seed)
            seed2 = None if args.seed is None else args.seed + 1
            mus, _ = _generate(pipe, music_prompt, args.negative, dur,
                               args.steps, args.guidance, seed2)
            sfx, mus = _to_stereo(sfx), _to_stereo(mus)
            target = int(dur * sr)
            sfx, mus = _fit_len(sfx, target), _fit_len(mus, target)
            gain = 10.0 ** (args.music_under_db / 20.0)
            wav = sfx + gain * mus
            peak = wav.abs().max().clamp(min=1e-6)
            if peak > 1.0:
                wav = wav / peak * 0.98
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wav = _fit_len(wav, int(dur * sr)).clamp(-1.0, 1.0)
    _save(args.out, wav, sr)
    _log(f"wrote {args.out} @ {sr} Hz, {wav.shape[-1] / sr:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

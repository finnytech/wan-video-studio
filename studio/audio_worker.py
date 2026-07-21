"""Standalone AudioCraft worker (run as a subprocess).

Generates audio of an exact duration from a text prompt using Meta's AudioCraft:
  * mode "sfx"   -> AudioGen (environmental sound effects)
  * mode "music" -> MusicGen (music bed)
  * mode "both"  -> AudioGen SFX with a MusicGen bed mixed underneath

Runs as its own process so all audio-model VRAM is reclaimed by the OS on exit.
Writes a single WAV at the requested sample rate to --out.

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


def _gen_audiogen(prompt: str, duration: float):
    import torch
    from audiocraft.models import AudioGen

    model = AudioGen.get_pretrained(_AUDIOGEN)
    model.set_generation_params(duration=max(1.0, duration))
    _log(f"AudioGen: '{prompt}' ({duration:.1f}s)")
    wav = model.generate([prompt])[0].detach().cpu()  # (channels, samples)
    sr = model.sample_rate
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wav, sr


def _gen_musicgen(prompt: str, duration: float):
    import torch
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained(_MUSICGEN)
    model.set_generation_params(duration=max(1.0, duration))
    _log(f"MusicGen: '{prompt}' ({duration:.1f}s)")
    wav = model.generate([prompt])[0].detach().cpu()
    sr = model.sample_rate
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wav, sr


def _to_stereo(wav):
    # wav: (channels, samples) -> (2, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav[:2]


def _resample(wav, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, src_sr, dst_sr)


def _fit_len(wav, samples: int):
    import torch

    cur = wav.shape[-1]
    if cur == samples:
        return wav
    if cur > samples:
        return wav[..., :samples]
    pad = samples - cur
    return torch.nn.functional.pad(wav, (0, pad))


def main() -> int:
    global _AUDIOGEN, _MUSICGEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sfx", "music", "both"], default="sfx")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--music_prompt", default=None)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audiogen_model", default="facebook/audiogen-medium")
    ap.add_argument("--musicgen_model", default="facebook/musicgen-large")
    ap.add_argument("--music_under_db", type=float, default=-8.0)
    args = ap.parse_args()

    _AUDIOGEN = args.audiogen_model
    _MUSICGEN = args.musicgen_model
    _load_tf32()

    import torch
    import torchaudio

    dur = float(args.duration)
    music_prompt = args.music_prompt or args.prompt

    if args.mode == "sfx":
        wav, sr = _gen_audiogen(args.prompt, dur)
        wav = _to_stereo(wav)
    elif args.mode == "music":
        wav, sr = _gen_musicgen(music_prompt, dur)
        wav = _to_stereo(wav)
    else:  # both -> mix SFX over an attenuated music bed
        sfx, sr_sfx = _gen_audiogen(args.prompt, dur)
        mus, sr_mus = _gen_musicgen(music_prompt, dur)
        sr = max(sr_sfx, sr_mus)
        sfx = _to_stereo(_resample(_to_stereo(sfx), sr_sfx, sr))
        mus = _to_stereo(_resample(_to_stereo(mus), sr_mus, sr))
        target = int(dur * sr)
        sfx, mus = _fit_len(sfx, target), _fit_len(mus, target)
        gain = 10.0 ** (args.music_under_db / 20.0)
        wav = sfx + gain * mus
        peak = wav.abs().max().clamp(min=1e-6)
        if peak > 1.0:
            wav = wav / peak * 0.98

    # trim/pad to exact duration and save
    wav = _fit_len(wav, int(dur * sr))
    wav = wav.clamp(-1.0, 1.0)
    torchaudio.save(args.out, wav, sr)
    _log(f"wrote {args.out} @ {sr} Hz, {wav.shape[-1]/sr:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

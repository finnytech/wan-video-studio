# 🎬 WAN Video Studio — WAN 2.2 video + AudioCraft sound, one click

Text → **WAN 2.2 (T2V-A14B, 14B)** video (5–10 s) → **AudioCraft** sound
(**AudioGen** SFX + **MusicGen** music) matched to the timeline → muxed MP4 with a
**download button** in a web UI, behind a **one-time OAuth-style token** so bots
can't spam the endpoint and burn your GPU hours.

Tuned for a single **RTX PRO 6000** (Blackwell, **96 GB VRAM**, 48 vCPU, ~500 TFLOPS)
on Lightning.ai. With 96 GB the 14B model fits **without offload**, so it's kept fully
resident for maximum speed.

> Sister project of [`hunyuan-video-studio`](https://github.com/finnytech/hunyuan-video-studio)
> (HunyuanVideo 1.5 + HunyuanVideo-Foley on A100). Same architecture, different stack.

> **Real, not a toy.** Video drives WAN's **official `generate.py`**; sound drives Meta's
> **AudioCraft** (`AudioGen`/`MusicGen`). Every flag is a real, tested upstream option.

---

## 0. Run on the RTX PRO 6000 VM

```bash
git clone https://github.com/finnytech/wan-video-studio.git
cd wan-video-studio
export HF_TOKEN=***           # for HuggingFace weight downloads (avoids throttling)
bash scripts/setup.sh            # clones WAN 2.2, downloads weights, installs AudioCraft
bash scripts/run.sh              # prints your PRIVATE share URL + access token
```

`run.sh` prints your `https://<random>.gradio.live` link **plus** an access token —
paste the token as the password on the login screen (username can be anything).

---

## 1. Pipeline

```
prompt ─► WAN 2.2 T2V-A14B (official generate.py) ─► silent .mp4  (saved locally)
                                                         │
                            generate.py exits ─► GPU VRAM fully released
                                                         ▼
prompt ─► AudioCraft: AudioGen (SFX) + MusicGen (music), exact video length ─► sound.wav
                                                         ▼
                              ffmpeg mux + normalize ─► download.mp4 (H.264+AAC, faststart)
```

- **Length**: **5–10 s** (81–161 frames at 16 fps). WAN 2.2 stays coherent in this
  range; past ~10 s motion drifts / loops, so the slider is capped at 10 s.
- **Resolution**: 480p (`832×480`) / 720p (`1280×720`) — T2V-A14B's native sizes.
- **Sound modes**:
  - `sfx` → AudioGen environmental sound effects
  - `music` → MusicGen music bed
  - `both` → AudioGen SFX over an attenuated MusicGen bed (mixed, length-matched)

  AudioGen/MusicGen are **text-conditioned** (not video-conditioned), so audio is
  generated from the prompt at the exact clip duration and aligned to the timeline
  by muxing at t=0 with length == video length.

## 2. Speed / optimization (RTX PRO 6000, 96 GB)

| Setting | Why |
|---------|-----|
| **No offload** (`WAN_OFFLOAD_MODEL=0`) | 96 GB fits the full 14B model → no CPU↔GPU streaming, max speed |
| **T5 on GPU** (`WAN_T5_CPU=0`) | text encoder stays resident |
| **30 sample steps** (default) | good speed/quality balance for T2V-A14B; lower = faster |
| **TF32 + cuDNN autotune** | free throughput on Blackwell, no quality loss |
| **48-vCPU threading** | VAE/data work parallelized (`STUDIO_CPU_THREADS=40`) |
| **FlashAttention** | installed best-effort for a big attention speedup |
| **Per-stage subprocess** | video VRAM 100% reclaimed before audio runs — no contention |
| **hf_transfer** | fast, un-throttled weight downloads |

All knobs live in `studio/config.py` and are env-overridable from `run.sh`.

## 3. Uncensored / custom checkpoint

The default video weights are the official `Wan-AI/Wan2.2-T2V-A14B`. To use an
**uncensored community finetune**, just point the studio at it — no code changes:

```bash
export VIDEO_MODEL_REPO="<hf-org>/<uncensored-wan22-t2v-a14b-repo>"
# or drop weights straight into models/Wan2.2-T2V-A14B-weights/
bash scripts/reset_weights.sh   # wipe + re-fetch from the new repo
```

## 4. Security model

- Public `*.gradio.live` link, but wrapped in a **token gate** (`studio/auth.py`,
  constant-time compare). Fresh token each launch, or pin via `STUDIO_TOKEN`.
- Wrong token → no access, no GPU spent. `--auth-only-local` skips the public link.
- `MAX_CONCURRENCY=1` + a GPU lock → requests never pile up on the single GPU.
- **Reveal-when-done UI**: nothing streams to the browser mid-render; the video
  appears (and download unlocks) only after video **and** sound are finished.

## 5. Files

```
scripts/setup.sh        installer: env, torch, WAN 2.2, weights, AudioCraft (idempotent)
scripts/run.sh          launches the UI + prints the private link & token
scripts/reset_weights.sh wipe + re-download WAN weights (e.g. to switch checkpoints)
studio/config.py        all knobs: repos, weights, resolutions, speed flags, audio modes
studio/models.py        clone WAN + download weights; verify AudioCraft
studio/video.py         WAN 2.2 driver (official generate.py)
studio/audio_worker.py  standalone AudioGen/MusicGen generator (own process)
studio/audio.py         audio stage wrapper + timeline mux
studio/mux.py           ffmpeg mux + browser-friendly normalize
studio/runner.py        shared subprocess runner (timeout / retry / logs / TF32 env)
studio/auth.py          one-time token gate
studio/app.py           Gradio UI (prompt / length / resolution / sound → video + download)
```

## 6. Requirements

RTX PRO 6000 / 80–96 GB-class GPU, CUDA 12.4, Python 3.10+, Linux, ffmpeg, git-lfs.
`setup.sh` handles the env, torch, WAN 2.2 + weights, and AudioCraft.

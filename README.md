# 🎬 WAN Video Studio — WAN 2.2 video + Stable Audio Open sound, one click

Text → **WAN 2.2 (T2V-A14B, 14B)** video (5–10 s) → **Stable Audio Open** sound
(SFX + music) matched to the timeline → muxed MP4 with a
**download button** in a web UI, behind a **one-time OAuth-style token** so bots
can't spam the endpoint and burn your GPU hours.

Tuned for a single **RTX PRO 6000** (Blackwell, **96 GB VRAM**, 48 vCPU, ~500 TFLOPS)
on Lightning.ai. With 96 GB the 14B model fits **without offload**, so it's kept fully
resident for maximum speed.

> Sister project of [`hunyuan-video-studio`](https://github.com/finnytech/hunyuan-video-studio)
> (HunyuanVideo 1.5 + HunyuanVideo-Foley on A100). Same architecture, different stack.

> **Real, not a toy.** Video drives WAN's **official `generate.py`**; sound drives Stability AI's
> **Stable Audio Open** (diffusers `StableAudioPipeline`). Every flag is a real, tested upstream option.

---

## 0. Run on the RTX PRO 6000 VM

```bash
git clone https://github.com/finnytech/wan-video-studio.git
cd wan-video-studio
# OPTIONAL: paste a REAL token (starts with hf_). Otherwise just skip this line —
# the WAN video weights are a public repo and download fine anonymously.
# Do NOT paste a placeholder with '...'/'…' in it; a bad token is now ignored
# automatically, but a real token avoids HF rate-limiting.
export HF_TOKEN=hf_your_real_token_here
bash scripts/setup.sh            # clones WAN 2.2, downloads weights, installs Stable Audio Open
bash scripts/run.sh              # prints your PRIVATE share URL + access token
```

> **No token? Fine.** The video model (`Wan-AI/Wan2.2-T2V-A14B`) is public, so
> setup/run work with **no** `HF_TOKEN` at all. A real token only speeds up
> downloads (avoids throttling) and unlocks the gated Stable Audio Open weights.

> **Sound is a gated model.** Stable Audio Open needs a one-time license accept on
> your HF account (the same one behind `HF_TOKEN`):
> https://huggingface.co/stabilityai/stable-audio-open-1.0 —
> without it, video still works and audio just stays off.

`run.sh` prints your `https://<random>.gradio.live` link **plus** an access token —
paste the token as the password on the login screen (username can be anything).

---

## 1. Pipeline

```
prompt ─► WAN 2.2 T2V-A14B (official generate.py) ─► silent .mp4  (saved locally)
                                                         │
                            generate.py exits ─► GPU VRAM fully released
                                                         ▼
prompt ─► Stable Audio Open (SFX + music), exact video length ─► sound.wav
                                                         ▼
                              ffmpeg mux + normalize ─► download.mp4 (H.264+AAC, faststart)
```

- **Length**: **5–10 s** (81–161 frames at 16 fps). WAN 2.2 stays coherent in this
  range; past ~10 s motion drifts / loops, so the slider is capped at 10 s.
- **Resolution**: 480p (`832×480`) / 720p (`1280×720`) — T2V-A14B's native sizes.
- **Sound modes** (all one model — Stable Audio Open does SFX *and* music):
  - `sfx` → environmental sound effects / ambience of the scene
  - `music` → a cinematic music score / bed
  - `both` → SFX over an attenuated music bed (two passes, mixed, length-matched)

  Stable Audio Open is **text-conditioned** (not video-conditioned), so audio is
  generated from the prompt at the exact clip duration and aligned to the timeline
  by muxing at t=0 with length == video length. The video prompt is auto-reshaped
  into an audio prompt so effects actually match the scene.

## 2. Speed / optimization (RTX PRO 6000, 96 GB)

| Setting | Why |
|---------|-----|
| **No offload** (`WAN_OFFLOAD_MODEL=0`) | 96 GB fits the full 14B model → no CPU↔GPU streaming, max speed |
| **T5 on GPU** (`WAN_T5_CPU=0`) | text encoder stays resident |
| **40 sample steps** (default) | max realism/detail at full precision; drop to 30 for faster drafts |
| **Cinematic enhancement** (`STUDIO_ENHANCE=1`) | auto film-look descriptors (ARRI/35mm/photoreal/real physics) for the film-action look |
| **Audio guidance 7.0** (`STUDIO_AUDIO_GUIDANCE`) + **100 steps** (`STUDIO_AUDIO_STEPS`) | stronger prompt adherence → crisper, more realistic SFX/music |
| **TF32 + cuDNN autotune** | free throughput on Blackwell, no quality loss |
| **48-vCPU threading** | VAE/data work parallelized (`STUDIO_CPU_THREADS=40`) |
| **FlashAttention** | installed best-effort for a big attention speedup |
| **Per-stage subprocess** | video VRAM 100% reclaimed before audio runs — no contention |
| **hf_transfer** | fast, un-throttled weight downloads |

All knobs live in `studio/config.py` and are env-overridable from `run.sh`.

### 2a. Never crash on VRAM — the auto-OOM ladder

The 96 GB card normally keeps the whole 14B model resident (fastest). But if VRAM
ever gets tight — a shared VM, fragmentation on a long session, or an unusually
heavy render — the studio **degrades gracefully instead of crashing**.

At launch it **probes free VRAM** (NVML, so it also sees other processes' usage)
and picks the fastest *rung* that fits. On any CUDA out-of-memory it **climbs one
rung and retries the same render**, spilling the idle parts of the model to pinned
host RAM over PCIe (block-swap) and pulling them back when needed:

| Rung | WAN flags added | What moves to host RAM | Speed |
|------|-----------------|------------------------|-------|
| 0 `resident` | *(none)* | nothing — whole 14B model on GPU | fastest |
| 1 `offload-expert` | `--offload_model True` | the idle noise-expert streams to pinned RAM | ~10–20% slower |
| 2 `+ t5-cpu` | `--t5_cpu` | T5 text encoder stays on CPU (~11 GB freed) | a bit slower |
| 3 `+ dtype` | `--convert_model_dtype` | weights cast to bf16/fp16 on load (~half VRAM) | slowest, most frugal |

Rungs are monotonic (each only *adds* savings) and every flag is a real, tested
WAN `generate.py` option — nothing hacky. The allocator is also tuned with
`expandable_segments:True,garbage_collection_threshold:0.9` to kill fragmentation
OOMs before the ladder is even needed.

**Honest note:** "instant microsecond swapping" is marketing. Real block-swap is
PCIe-bound (tens of GB/s), so a spilled render is somewhat slower — but it
*finishes* instead of dying with `CUDA out of memory`. That's the whole point.

| Env knob | Default | Meaning |
|----------|---------|---------|
| `WAN_AUTO_OOM` | `1` | master switch for the probe + escalation ladder (`0` = old fixed config) |
| `WAN_VRAM_RESIDENT_GB` | `55` | free-VRAM ≥ this → start on rung 0 (fully resident) |
| `WAN_VRAM_OFFLOAD_GB` | `32` | free-VRAM ≥ this → start on rung 1 (offload expert) |
| `WAN_VRAM_T5CPU_GB` | `20` | free-VRAM ≥ this → start on rung 2 (+ T5 on CPU); below → rung 3 |
| `WAN_OFFLOAD_MODEL` / `WAN_T5_CPU` / `WAN_CONVERT_DTYPE` | `0` | force a *minimum* rung by hand; the ladder never starts weaker than these |

The live rung and each escalation are printed to the log, e.g.
`VRAM probe: 41.3 GB free / 96 GB total -> starting offload rung 1 (offload-expert)`
and `video: CUDA OOM -> escalating offload to variant 3 (offload+t5-cpu) ...`.

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
scripts/setup.sh        installer: env, torch, WAN 2.2, weights, Stable Audio Open (idempotent)
scripts/run.sh          launches the UI + prints the private link & token
scripts/reset_weights.sh wipe + re-download WAN weights (e.g. to switch checkpoints)
studio/config.py        all knobs: repos, weights, resolutions, speed flags, audio modes
studio/models.py        clone WAN + download weights; verify Stable Audio Open
studio/video.py         WAN 2.2 driver (official generate.py) + auto-OOM ladder wiring
studio/gpu.py           VRAM probe (NVML/torch) + OOM escalation ladder (rungs 0-3)
studio/audio_worker.py  standalone Stable Audio Open generator (own process)
studio/audio.py         audio stage wrapper + timeline mux
studio/mux.py           ffmpeg mux + browser-friendly normalize
studio/runner.py        shared subprocess runner (timeout / retry / logs / TF32 env / OOM auto-escalate)
studio/auth.py          one-time token gate
studio/app.py           Gradio UI (prompt / length / resolution / sound → video + download)
```

## 6. Requirements

RTX PRO 6000 / 80–96 GB-class GPU, CUDA 12.4, Python 3.10+, Linux, ffmpeg, git-lfs.
`setup.sh` handles the env, torch, WAN 2.2 + weights, and Stable Audio Open.

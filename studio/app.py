"""Gradio web UI: prompt / length / resolution / sound -> video with sound.

Order of operations:
  1. WAN 2.2 renders the video (own process; full 96 GB to the video model).
  2. That process EXITS -> its VRAM is fully released. Silent video saved locally.
  3. AudioCraft (AudioGen SFX and/or MusicGen music) generates audio of the exact
     video length, muxed onto the timeline.
  4. Only once BOTH video and sound are done is the finished MP4 revealed and the
     download button shown. Nothing is streamed to the web mid-render.

The whole app sits behind a one-time token gate so bots can't spam the endpoint.
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

from . import audio, config, mux
from .auth import generate_token, make_auth_callback
from .video import VideoGenerator

_VIDEO = VideoGenerator()
_GPU_LOCK = threading.Lock()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _busy(msg: str):
    return (msg, None, gr.update(visible=False), gr.update(interactive=False))


def generate(prompt, seconds, resolution, steps, seed, audio_mode, music_prompt):
    prompt = (prompt or "").strip()
    if not prompt:
        yield ("⚠️ Please enter a prompt.", None,
               gr.update(visible=False), gr.update(interactive=True))
        return
    if not _GPU_LOCK.acquire(blocking=False):
        yield ("⏳ The GPU is busy with another generation. Try again shortly.", None,
               gr.update(visible=False), gr.update(interactive=True))
        return

    t0 = time.time()
    try:
        yield _busy("🎬 Stage 1/3 — rendering video (WAN 2.2 · T2V-A14B)…")
        stamp = _stamp()
        work = config.OUTPUT_DIR / stamp
        work.mkdir(parents=True, exist_ok=True)
        seed_val = int(seed) if str(seed).strip() else None

        silent = _VIDEO.generate(
            prompt=prompt, seconds=float(seconds), resolution=resolution,
            steps=int(steps), seed=seed_val, out_path=work / "silent.mp4",
        )
        t_video = time.time() - t0

        yield _busy(f"🧹 Stage 2/3 — video done in {t_video:.0f}s, freeing VRAM…")
        _VIDEO.unload()

        yield _busy(f"🔊 Stage 3/3 — generating {audio_mode} sound (AudioCraft)…")
        with_sound = audio.add_sound(
            silent, prompt, mode=audio_mode,
            music_prompt=(music_prompt or "").strip() or None,
            out_dir=work / "audio",
        )
        final = mux.normalize_final(with_sound, work / "final.mp4")

        dest = config.VIDEOS_DIR / f"wan_{stamp}.mp4"
        shutil.copy2(final, dest)

        dt = time.time() - t0
        msg = (f"✅ Done in {dt:.0f}s  ·  video {t_video:.0f}s + sound "
               f"{dt - t_video:.0f}s  ·  {int(float(seconds))}s @ {resolution} · {audio_mode}")
        yield (msg, str(dest), gr.update(value=str(dest), visible=True),
               gr.update(interactive=True))

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        yield (f"❌ Generation failed: {e}", None,
               gr.update(visible=False), gr.update(interactive=True))
    finally:
        _GPU_LOCK.release()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WAN Video Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 WAN Video Studio\n"
            "Text → **WAN 2.2 (14B)** video → **AudioCraft** sound (AudioGen SFX + "
            "MusicGen music). Tuned for RTX PRO 6000 · 96 GB. "
            "_The video appears only once both video and sound are finished._"
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="A cinematic wave crashing on rocks at sunset, seagulls overhead",
                )
                with gr.Row():
                    seconds = gr.Slider(config.MIN_SECONDS, config.MAX_SECONDS,
                                        value=config.DEFAULT_SECONDS, step=1,
                                        label="Length (seconds)")
                    resolution = gr.Radio(
                        config.available_resolutions(),
                        value=config.DEFAULT_RESOLUTION, label="Resolution",
                    )
                with gr.Row():
                    steps = gr.Slider(4, 50, value=config.DEFAULT_STEPS, step=1,
                                      label="Sample steps (higher = better/slower)")
                    seed = gr.Textbox(label="Seed (blank = random)", value="")
                with gr.Row():
                    audio_mode = gr.Radio(
                        ["sfx", "music", "both"], value=config.DEFAULT_AUDIO_MODE,
                        label="Sound (AudioGen=sfx · MusicGen=music · both=mixed)",
                    )
                    music_prompt = gr.Textbox(
                        label="Music prompt (optional; for music/both)",
                        placeholder="warm ambient cinematic score",
                    )
                go = gr.Button("Generate 🎬", variant="primary")
                status = gr.Markdown("")
            with gr.Column(scale=4):
                video_out = gr.Video(label="Result (video + sound)", autoplay=True)
                download = gr.DownloadButton(label="⬇️ Download MP4", visible=False)

        go.click(
            generate,
            inputs=[prompt, seconds, resolution, steps, seed, audio_mode, music_prompt],
            outputs=[status, video_out, download, go],
            concurrency_limit=config.MAX_CONCURRENCY,
        )
    return demo


def main() -> None:
    try:
        from .models import ensure_all

        ensure_all()
    except Exception as e:  # noqa: BLE001
        print(f"[app] model check warning: {e}")

    token = generate_token()
    demo = build_ui()
    demo.queue(max_size=8)

    print("\n" + "=" * 62)
    print("  WAN Video Studio is starting")
    print("  🔑 Access token (use as the PASSWORD on login):")
    print(f"       {token}")
    print("  Username can be anything.")
    print("=" * 62 + "\n", flush=True)

    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        share=config.PUBLIC_SHARE,
        auth=make_auth_callback(token),
        auth_message="Enter the access token (as password) to use the studio.",
        show_api=False,
        max_threads=config.MAX_CONCURRENCY + 2,
    )


if __name__ == "__main__":
    main()

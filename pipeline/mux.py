"""
Bhasha-Setu — Audio-Video Mux Service
Merges dubbed audio with original video using FFmpeg.
Supports optional background music mixing from the original video.

Pipeline: MP3 → WAV → atempo speed-match → (optional mix with BG) → AAC encode → MP4
"""

import subprocess
import logging
import os
from typing import Callable, Optional

# Ensure FFmpeg is available on PATH
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

_ffmpeg_fallback = r"C:\Users\Abhimanyu\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\static_ffmpeg\bin\win32"
if os.path.exists(_ffmpeg_fallback) and _ffmpeg_fallback not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_fallback + os.pathsep + os.environ.get("PATH", "")

log = logging.getLogger(__name__)


def _run(cmd: list, step: str) -> None:
    """Run FFmpeg command. Raises with clean error message on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"{step} failed: {e}")
    if result.returncode != 0:
        err = result.stderr.strip()[-600:]
        raise RuntimeError(f"{step} failed:\n{err}")


def get_duration(path: str) -> float:
    """Get media file duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"Cannot read duration of '{path}': {e}")
    out = result.stdout.strip()
    if not out or result.returncode != 0:
        raise RuntimeError(
            f"Cannot read duration of '{path}'. "
            f"Is FFprobe installed? stderr: {result.stderr[:200]}"
        )
    return float(out)


def _has_audio(path: str) -> bool:
    """Return True if the video file has at least one audio stream."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return bool(result.stdout.strip())


def _mp3_to_wav(mp3_path: str) -> str:
    """Convert MP3 → WAV (PCM 16-bit, 44100 Hz, mono)."""
    wav_path = os.path.splitext(mp3_path)[0] + "_raw.wav"
    _run([
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        wav_path
    ], "MP3→WAV conversion")
    return wav_path


def _build_atempo_chain(ratio: float, vol_boost: float = 2.0) -> list[str]:
    """
    Build FFmpeg atempo filter chain for speed adjustment + volume boost.
    Each atempo value must be in [0.5, 2.0].
    """
    if ratio <= 0: raise RuntimeError("Audio duration is zero or negative — TTS may have produced an empty file.")
    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    if abs(r - 1.0) > 0.005:
        filters.append(f"atempo={r:.6f}")
    safe_vol = max(0.1, min(8.0, float(vol_boost)))
    filters.append(f"volume={safe_vol:.2f}")
    filters.append("aresample=44100")
    return filters


def _adjust_speed(wav_path: str, ratio: float, vol_boost: float = 2.0) -> str:
    """Apply speed adjustment + volume boost to WAV. Returns adjusted WAV path."""
    adj_path = os.path.splitext(wav_path)[0].replace("_raw", "") + "_adj.wav"
    filter_str = ",".join(_build_atempo_chain(ratio, vol_boost))
    log.info(f"Speed ratio: {ratio:.4f} | Volume: {vol_boost:.2f}x | Filter: {filter_str}")
    _run([
        "ffmpeg", "-y",
        "-i", wav_path,
        "-filter:a", filter_str,
        "-ar", "44100",
        "-ac", "1",
        adj_path
    ], "Audio speed adjustment")
    return adj_path


def mux(
    video_path: str,
    mp3_path: str,
    output_path: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    vol_boost: float = 2.0,
    bg_music_vol: float = 0.0,
) -> str:
    """
    Merge dubbed audio (MP3) with original video.

    Pipeline steps:
      10% — MP3 → WAV conversion
      30% — Duration analysis
      60% — Speed / tempo adjustment
      75% — Background music mix (if enabled)
      85% — Final merge
     100% — Verification & done

    Args:
        video_path:   Original video file (.mp4).
        mp3_path:     Dubbed audio (.mp3).
        output_path:  Final dubbed video path (.mp4).
        progress_cb:  Optional callable(pct: float, message: str).
        vol_boost:    Volume multiplier for dubbed voice (default 2.0).
        bg_music_vol: Background music volume from original video (0.0 = off, 0.15 = 15%).
                      When > 0 the original audio is mixed in behind the dubbed voice.

    Returns:
        output_path
    """
    def _p(pct: float, msg: str):
        log.info(f"[Mux {pct:.0f}%] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    wav_path = None
    adj_wav = None
    try:
        _p(5, "Starting mux…")

        # Step 1: MP3 → WAV
        _p(10, "Converting MP3 → WAV…")
        wav_path = _mp3_to_wav(mp3_path)

        # Step 2: Measure durations
        _p(30, "Measuring durations…")
        video_dur = get_duration(video_path)
        if video_dur <= 0: raise RuntimeError(f"Video has zero duration: {video_path}")
        audio_dur = get_duration(wav_path)
        diff_pct  = abs(1.0 - audio_dur / video_dur) * 100
        log.info(f"Video={video_dur:.2f}s | Audio={audio_dur:.2f}s | Diff={diff_pct:.1f}%")

        ratio = audio_dur / video_dur
        _p(45, f"Speed ratio: {ratio:.3f} (audio {audio_dur:.1f}s → video {video_dur:.1f}s)")

        # Step 3: Speed adjustment
        _p(55, f"Applying tempo & volume filters (boost={vol_boost:.1f}x)…")
        adj_wav = _adjust_speed(wav_path, ratio, vol_boost)
        adj_dur = get_duration(adj_wav)
        log.info(f"Adjusted audio: {adj_dur:.2f}s (target: {video_dur:.2f}s)")

        # Step 4: Decide merge strategy
        use_bg = (
            bg_music_vol > 0.0
            and _has_audio(video_path)
        )

        if use_bg:
            # ── Mix dubbed voice + original background music ──────────────
            _p(70, f"Mixing background music at {bg_music_vol:.0%} volume…")
            safe_bg = max(0.01, min(1.0, float(bg_music_vol)))

            # filter_complex:
            #   [0:a] = original video audio → lower to bg_music_vol
            #   [1:a] = dubbed voice (already boosted) → keep as-is
            #   amix both, duration=first (= video duration), normalize off
            filter_complex = (
                f"[0:a]volume={safe_bg:.3f},aresample=44100[bg];"
                f"[1:a]aresample=44100[dub];"
                f"[dub][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
            )

            _p(80, "Merging video + dubbed voice + background music…")
            _run([
                "ffmpeg", "-y",
                "-i", video_path,   # 0: video + original audio
                "-i", adj_wav,      # 1: dubbed voice
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-map", "[out]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "44100",
                "-t", str(video_dur),
                "-movflags", "+faststart",
                output_path
            ], "Audio-video merge with background music")

        else:
            # ── Dubbed voice only (original behaviour) ────────────────────
            _p(75, "Merging dubbed audio with video…")
            _run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", adj_wav,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "44100",
                "-t", str(video_dur),
                "-movflags", "+faststart",
                output_path
            ], "Audio-video merge")

        # Verify output
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise RuntimeError(f"Output video missing or too small: {output_path}")

        out_size = os.path.getsize(output_path) / (1024 * 1024)
        out_dur  = get_duration(output_path)
        bg_note  = f" (background music: {bg_music_vol:.0%})" if use_bg else ""
        _p(100, f"Done! {out_size:.1f} MB, {out_dur:.1f}s{bg_note}")
        log.info(f"Mux complete: {output_path} ({out_size:.1f} MB, {out_dur:.2f}s)")

        return output_path
    finally:
        # Cleanup temp files
        _p(90, "Cleaning up temp files…")
        for f in [wav_path, adj_wav]:
            if f:
                try:
                    os.remove(f)
                except Exception:
                    pass

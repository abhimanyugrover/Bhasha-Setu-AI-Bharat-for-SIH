"""
Bhasha-Setu — Speech Synthesis Service
Three-engine strategy with per-chunk progress callbacks.

Engine strategy:
  polly  -> AWS Polly Kajal Neural  -> Hindi (best quality)
  edge   -> Microsoft edge-tts Neural -> Tamil, Telugu, Kannada, Malayalam,
                                         Bengali, Marathi, Gujarati, Urdu
  gtts   -> Google TTS              -> Punjabi, Odia, Assamese

FIX: Fallback now uses the correct language config (lang_cfg) instead of
hardcoded Hindi ("hi"), so a failed engine falls back in the same language.
"""

import os
import re
import asyncio
import logging
import subprocess
import tempfile
from typing import Callable, Optional
from xml.sax.saxutils import escape as _xml_escape

import boto3

from pipeline.config import (
    AWS_REGION,
    LANGUAGES,
    POLLY_CHUNK_CHARS,
    EDGE_CHUNK_CHARS,
    GTTS_CHUNK_CHARS,
)

log = logging.getLogger(__name__)


def _chunk_text(text: str, max_chars: int) -> list:
    """Split text into chunks at sentence boundaries."""
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    chunks, current = [], ""
    normalized = text.replace("। ", "।\n").replace("॥ ", "॥\n").replace("| ", "|\n").replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n").replace("؟ ", "؟\n")

    for sentence in normalized.splitlines():
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def _concat_mp3_files(mp3_files: list, output_path: str) -> None:
    """Concatenate multiple MP3 chunks into one using FFmpeg or binary fallback."""
    if not mp3_files:
        return
    if len(mp3_files) == 1:
        import shutil
        shutil.copyfile(mp3_files[0], output_path)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in mp3_files:
            abs_path = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output_path
        ], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.warning(f"FFmpeg concat unavailable ({e}), using direct MP3 frame concatenation.")
        with open(output_path, "wb") as out_f:
            for p in mp3_files:
                with open(p, "rb") as in_f:
                    out_f.write(in_f.read())
    finally:
        if os.path.exists(list_file):
            try:
                os.unlink(list_file)
            except Exception:
                pass


# ── POLLY ─────────────────────────────────────────────────────────────────────

def _wrap_ssml(text: str, pitch_percent: int = 0, rate: str = "medium") -> str:
    """Wrap text in SSML for pitch + rate control (Polly only) with entity and control char sanitization."""
    clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    escaped_text = _xml_escape(clean_text, entities={'"': "&quot;", "'": "&apos;"})
    safe_pitch = max(-20, min(20, int(pitch_percent)))
    sign       = "+" if safe_pitch > 0 else ""
    pitch_attr = f' pitch="{sign}{safe_pitch}%"' if safe_pitch else ""
    allowed_rates = {"x-slow", "slow", "medium", "fast", "x-fast"}
    safe_rate = rate if rate in allowed_rates else "medium"
    rate_attr  = f' rate="{safe_rate}"'
    return f'<speak><prosody{pitch_attr}{rate_attr}>{escaped_text}</prosody></speak>'


def _synthesize_polly(
    text: str,
    lang_cfg: dict,
    output_mp3: str,
    progress_cb=None,
    pitch_percent: int = 0,
) -> str:
    client      = boto3.client("polly", region_name=AWS_REGION)
    chunks      = _chunk_text(text, POLLY_CHUNK_CHARS)
    total       = len(chunks)
    chunk_files = []

    try:
        for i, chunk in enumerate(chunks):
            if progress_cb:
                progress_cb(i, total)
            tmp = os.path.splitext(output_mp3)[0] + f"_chunk{i}.mp3"

            if pitch_percent:
                ssml = _wrap_ssml(chunk, pitch_percent)
                resp = client.synthesize_speech(
                    Text=ssml, TextType="ssml",
                    OutputFormat="mp3",
                    VoiceId=lang_cfg["polly_voice"],
                    Engine=lang_cfg["polly_engine"],
                    LanguageCode=lang_cfg["polly_lang_code"],
                )
            else:
                resp = client.synthesize_speech(
                    Text=chunk, OutputFormat="mp3",
                    VoiceId=lang_cfg["polly_voice"],
                    Engine=lang_cfg["polly_engine"],
                    LanguageCode=lang_cfg["polly_lang_code"],
                )
            with open(tmp, "wb") as f:
                f.write(resp["AudioStream"].read())
            chunk_files.append(tmp)

        if progress_cb:
            progress_cb(total, total)

        _concat_mp3_files(chunk_files, output_mp3)
        return output_mp3
    finally:
        for f in chunk_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


# ── EDGE-TTS ──────────────────────────────────────────────────────────────────

async def _edge_tts_chunk_async(text: str, voice: str, out_path: str, pitch_hz: str = "+0Hz") -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, voice, pitch=pitch_hz)
    await comm.save(out_path)


def _pitch_percent_to_hz(pitch_percent: int) -> str:
    hz = int(round(pitch_percent * 2))
    hz = max(-200, min(200, hz))
    return f"{'+' if hz >= 0 else ''}{hz}Hz"


def _synthesize_edge(
    text: str,
    lang_cfg: dict,
    output_mp3: str,
    progress_cb=None,
    pitch_percent: int = 0,
) -> str:
    voice    = lang_cfg["edge_voice"]
    pitch_hz = _pitch_percent_to_hz(pitch_percent)
    chunks   = _chunk_text(text, EDGE_CHUNK_CHARS)
    total    = len(chunks)
    log.info(f"edge-tts | voice={voice} | pitch={pitch_hz} | {total} chunk(s) (parallel mode)")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    chunk_files = []
    try:
        async def _run_concurrent():
            tasks = []
            for i, chunk in enumerate(chunks):
                tmp = os.path.splitext(output_mp3)[0] + f"_chunk{i}.mp3"
                chunk_files.append(tmp)
                tasks.append(_edge_tts_chunk_async(chunk, voice, tmp, pitch_hz))
            await asyncio.gather(*tasks)

        try:
            if progress_cb:
                progress_cb(0, total)
            loop.run_until_complete(_run_concurrent())
            if progress_cb:
                progress_cb(total, total)
        finally:
            loop.close()

        _concat_mp3_files(chunk_files, output_mp3)
        return output_mp3
    finally:
        for f in chunk_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


# ── GTTS ──────────────────────────────────────────────────────────────────────

def _synthesize_gtts(
    text: str,
    lang_cfg: dict,
    output_mp3: str,
    progress_cb=None,
    pitch_percent: int = 0,   # gTTS does not support pitch control
) -> str:
    from gtts import gTTS, lang as gtts_lang

    target_lang     = lang_cfg.get("gtts_lang", "hi")
    supported_langs = gtts_lang.tts_langs()

    if target_lang not in supported_langs:
        log.warning(f"gTTS: '{target_lang}' not supported. Falling back to Hindi ('hi').")
        target_lang = "hi"

    chunks      = _chunk_text(text, GTTS_CHUNK_CHARS)
    total       = len(chunks)
    chunk_files = []

    try:
        for i, chunk in enumerate(chunks):
            if progress_cb:
                progress_cb(i, total)
            tmp = os.path.splitext(output_mp3)[0] + f"_chunk{i}.mp3"
            gTTS(text=chunk, lang=target_lang, slow=False).save(tmp)
            chunk_files.append(tmp)

        if progress_cb:
            progress_cb(total, total)

        _concat_mp3_files(chunk_files, output_mp3)
        return output_mp3
    finally:
        for f in chunk_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def synthesize_speech(
    text: str,
    output_mp3: str,
    lang_name: str,
    progress_cb=None,
    pitch_percent: int = 0,
    tts_engine: str = "auto",
    sarvam_speaker: str = "priya",
) -> str:
    """
    Synthesize speech for the given text and language with multi-engine support:
      - 'sarvam': Sarvam AI (Bulbul:v3) - Indic expressive neural voices
      - 'edge': Microsoft Edge Neural
      - 'polly': AWS Polly Neural
      - 'gtts': Google TTS fallback
      - 'auto': Uses Sarvam AI if available, else language default (Polly/Edge/gTTS)

    Args:
        text:           Text to synthesize.
        output_mp3:     Output MP3 file path.
        lang_name:      Language name from LANGUAGES registry (e.g. "Hindi").
        progress_cb:    Optional callable(chunk_index, total_chunks).
        pitch_percent:  Pitch offset percent (-20 to +20).
        tts_engine:     'auto', 'sarvam', 'edge', 'polly', or 'gtts'.
        sarvam_speaker: Sarvam speaker name ('priya', 'aditya', 'ritu', etc.).

    Returns:
        output_mp3 path
    """
    if not text or not text.strip():
        raise ValueError("TTS text is empty.")

    if lang_name not in LANGUAGES:
        raise ValueError(f"Unsupported language for synthesis: {lang_name}")

    lang_cfg = LANGUAGES[lang_name]

    # Check for Sarvam AI engine request or auto-detection
    from pipeline.sarvam import is_sarvam_available, synthesize_sarvam, SARVAM_LANGUAGES

    use_sarvam = False
    if tts_engine == "sarvam":
        use_sarvam = True
    elif tts_engine == "auto" and is_sarvam_available() and lang_name in SARVAM_LANGUAGES:
        use_sarvam = True

    if use_sarvam:
        try:
            log.info(f"Synthesizing with Sarvam AI Bulbul:v3 ({lang_name}, speaker={sarvam_speaker})")
            return synthesize_sarvam(
                text=text,
                output_mp3=output_mp3,
                lang_name=lang_name,
                speaker=sarvam_speaker,
                pitch=pitch_percent,
                progress_cb=progress_cb,
            )
        except Exception as e:
            log.warning(f"Sarvam AI TTS encountered an error: {e}. Gracefully falling back to native engine.")

    # Native engine routing
    engine = lang_cfg["tts"] if tts_engine in ("auto", "sarvam") else tts_engine

    try:
        if engine == "polly":
            return _synthesize_polly(text, lang_cfg, output_mp3, progress_cb, pitch_percent)
        elif engine == "edge":
            return _synthesize_edge(text, lang_cfg, output_mp3, progress_cb, pitch_percent)
        elif engine == "gtts":
            return _synthesize_gtts(text, lang_cfg, output_mp3, progress_cb, pitch_percent)
        else:
            return _synthesize_edge(text, lang_cfg, output_mp3, progress_cb, pitch_percent)

    except Exception as e:
        log.error(f"Engine '{engine}' failed for {lang_name}: {e}. Trying gTTS fallback.")
        try:
            return _synthesize_gtts(text, lang_cfg, output_mp3, None)
        except Exception as fallback_err:
            raise RuntimeError(
                f"All TTS engines failed for {lang_name}. Last error: {fallback_err}"
            ) from fallback_err

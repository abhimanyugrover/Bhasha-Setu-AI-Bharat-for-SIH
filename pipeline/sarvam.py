"""
Bhasha-Setu — Sarvam AI Integration Service
Provides state-of-the-art Indian Language Text-to-Speech (Bulbul:v1)
and Translation (Mayura:v1) tailored for Indian accents and dialects.
"""

import os
import re
import base64
import logging
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Dict, List, Tuple

import requests
import streamlit as st

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

_custom_ffmpeg = os.environ.get("FFMPEG_PATH")
if _custom_ffmpeg and os.path.exists(_custom_ffmpeg) and _custom_ffmpeg not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _custom_ffmpeg + os.pathsep + os.environ.get("PATH", "")

log = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# Maximum characters allowed per chunk by Sarvam TTS API is 500. We use 450 for safety.
SARVAM_MAX_CHUNK_CHARS = 450

# Supported languages in Sarvam AI
SARVAM_LANGUAGES = {
    "Hindi":     "hi-IN",
    "Tamil":     "ta-IN",
    "Telugu":    "te-IN",
    "Kannada":   "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali":   "bn-IN",
    "Marathi":   "mr-IN",
    "Gujarati":  "gu-IN",
    "Punjabi":   "pa-IN",
    "Odia":      "od-IN",
    "English":   "en-IN",
}

# Sarvam AI Bulbul Voices (bulbul:v3)
SARVAM_VOICES = {
    "priya":   {"name": "Priya",   "gender": "Female", "desc": "Warm, conversational & engaging (Recommended)"},
    "aditya":  {"name": "Aditya",  "gender": "Male",   "desc": "Clear, professional & steady narration"},
    "ritu":    {"name": "Ritu",    "gender": "Female", "desc": "Polite, melodic & clear"},
    "rahul":   {"name": "Rahul",   "gender": "Male",   "desc": "Deep, confident & dynamic"},
    "pooja":   {"name": "Pooja",   "gender": "Female", "desc": "Youthful, energetic & cheerful"},
    "rohan":   {"name": "Rohan",   "gender": "Male",   "desc": "Casual, friendly & modern"},
    "shreya":  {"name": "Shreya",  "gender": "Female", "desc": "Polite, expressive & clear"},
    "simran":  {"name": "Simran",  "gender": "Female", "desc": "Natural, expressive & clear narration"},
    "dev":     {"name": "Dev",     "gender": "Male",   "desc": "Energetic, dynamic & articulate"},
    "amit":    {"name": "Amit",    "gender": "Male",   "desc": "Warm, trustworthy & calm"},
}

SPEAKER_ALIASES = {
    "meera":   "priya",
    "arvind":  "aditya",
    "amartya": "rahul",
    "ananya":  "ritu",
    "dhruv":   "aditya",
}


def get_sarvam_api_key() -> str:
    """Retrieve Sarvam API Key from st.secrets or os.environ."""
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        try:
            key = st.secrets.get("SARVAM_API_KEY", "").strip()
        except Exception:
            key = ""
    return key


def is_sarvam_available() -> bool:
    """Return True if Sarvam API key is configured."""
    return bool(get_sarvam_api_key())


def _chunk_text_for_sarvam(text: str, max_chars: int = SARVAM_MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks at sentence or punctuation boundaries without exceeding max_chars."""
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    # Normalize sentence terminators across Indic scripts and English
    normalized = (
        text.replace("। ", "।\n")
            .replace("॥ ", "॥\n")
            .replace(". ", ".\n")
            .replace("? ", "?\n")
            .replace("! ", "!\n")
            .replace("؛ ", "؛\n")
            .replace(" | ", " |\n")
    )

    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        # If line itself is too long, split by comma or words
        if len(line) > max_chars:
            words = line.split(" ")
            sub = ""
            for w in words:
                if len(sub) + len(w) + 1 <= max_chars:
                    sub = (sub + " " + w).strip()
                else:
                    if sub:
                        chunks.append(sub)
                    sub = w
            if sub:
                chunks.append(sub)
        else:
            if len(current) + len(line) + 1 <= max_chars:
                current = (current + " " + line).strip()
            else:
                if current:
                    chunks.append(current)
                current = line

    if current:
        chunks.append(current)

    return chunks or [text]


def _synthesize_chunk_worker(
    chunk_idx: int,
    chunk_text: str,
    target_lang_code: str,
    speaker: str,
    pace: float,
    pitch: float,
    api_key: str,
) -> Tuple[int, bytes]:
    """Worker to request a single TTS chunk from Sarvam AI with retry logic."""
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
    }

    # Bounded pitch (-20 to +20 mapped to Sarvam range)
    # Sarvam pitch range is approx -1.0 to 1.0 or integer depending on model version
    clamped_pitch = max(-2.0, min(2.0, float(pitch) / 10.0))
    clamped_pace = max(0.5, min(2.0, float(pace)))

    # Resolve speaker & alias
    speaker = speaker.lower().strip()
    speaker = SPEAKER_ALIASES.get(speaker, speaker)
    if speaker not in SARVAM_VOICES:
        speaker = "priya"

    payload = {
        "inputs": [chunk_text],
        "target_language_code": target_lang_code,
        "speaker": speaker,
        "pitch": clamped_pitch,
        "pace": clamped_pace,
        "loudness": 1.5,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
        "model": "bulbul:v3",
    }

    for attempt in range(3):
        try:
            resp = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                audios = data.get("audios", [])
                if audios:
                    raw_wav = base64.b64decode(audios[0])
                    return chunk_idx, raw_wav
            elif resp.status_code == 429:
                time_sleep = 1.0 * (attempt + 1)
                import time
                time.sleep(time_sleep)
                continue
            else:
                log.warning(f"Sarvam TTS HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log.warning(f"Sarvam TTS attempt {attempt+1} error: {e}")

    raise RuntimeError(f"Sarvam AI TTS failed for chunk {chunk_idx}: '{chunk_text[:30]}...'")


def synthesize_sarvam(
    text: str,
    output_mp3: str,
    lang_name: str,
    speaker: str = "meera",
    pitch: float = 0.0,
    pace: float = 1.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Synthesize high-quality speech using Sarvam AI Bulbul:v1 with parallel chunk processing.

    Args:
        text: Text to synthesize
        output_mp3: Destination audio file (.mp3 or .wav)
        lang_name: Language name (e.g. "Hindi", "Tamil")
        speaker: Sarvam speaker name ('meera', 'arvind', 'priya', etc.)
        pitch: Pitch adjustment
        pace: Pace/speed multiplier (default 1.0)
        progress_cb: Callable(completed_chunks, total_chunks)

    Returns:
        Path to output_mp3
    """
    api_key = get_sarvam_api_key()
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not configured.")

    target_lang_code = SARVAM_LANGUAGES.get(lang_name, "hi-IN")
    speaker = speaker.lower().strip()
    speaker = SPEAKER_ALIASES.get(speaker, speaker)
    if speaker not in SARVAM_VOICES:
        speaker = "priya"

    chunks = _chunk_text_for_sarvam(text, SARVAM_MAX_CHUNK_CHARS)
    total_chunks = len(chunks)

    if progress_cb:
        progress_cb(0, total_chunks)

    temp_files = []
    chunk_results = [None] * total_chunks

    # Use ThreadPoolExecutor to process chunks in parallel for minimum latency!
    max_workers = min(4, total_chunks) if total_chunks > 1 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _synthesize_chunk_worker,
                idx, chunk, target_lang_code, speaker, pace, pitch, api_key
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        completed = 0
        for future in as_completed(futures):
            idx, raw_wav = future.result()
            chunk_results[idx] = raw_wav
            completed += 1
            if progress_cb:
                progress_cb(completed, total_chunks)

    # Save chunks to temporary WAV files and concatenate
    try:
        for idx, wav_bytes in enumerate(chunk_results):
            tmp_f = tempfile.NamedTemporaryFile(suffix=f"_sarvam_{idx}.wav", delete=False)
            tmp_f.write(wav_bytes)
            tmp_f.flush()
            tmp_f.close()
            temp_files.append(tmp_f.name)

        if len(temp_files) == 1:
            # Single chunk: convert directly to output format
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", temp_files[0],
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    output_mp3
                ], check=True, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                log.warning(f"FFmpeg transcode failed ({e}), copying audio directly.")
                import shutil
                shutil.copyfile(temp_files[0], output_mp3)
        else:
            # Multi-chunk: concatenate using concat demuxer
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as list_f:
                list_file = list_f.name
                for p in temp_files:
                    clean_p = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
                    list_f.write(f"file '{clean_p}'\n")

            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0",
                    "-i", list_file,
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    output_mp3
                ], check=True, capture_output=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                log.warning(f"FFmpeg concat unavailable ({e}), using direct audio concatenation fallback.")
                with open(output_mp3, "wb") as out_f:
                    for p in temp_files:
                        with open(p, "rb") as in_f:
                            out_f.write(in_f.read())
            finally:
                if os.path.exists(list_file):
                    try:
                        os.unlink(list_file)
                    except Exception:
                        pass

        return output_mp3

    finally:
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


SARVAM_MAX_TRANSLATE_CHARS = 800


def _translate_sarvam_chunk(
    chunk_idx: int,
    chunk_text: str,
    src_code: str,
    tgt_code: str,
    mode: str,
    api_key: str,
) -> Tuple[int, str]:
    """Translate an individual text chunk (< 1800 chars) via Sarvam AI Mayura:v1."""
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
    }
    payload = {
        "input": chunk_text,
        "source_language_code": src_code,
        "target_language_code": tgt_code,
        "speaker_gender": "Female",
        "mode": mode,
        "model": "mayura:v1",
    }
    for attempt in range(3):
        try:
            resp = requests.post(SARVAM_TRANSLATE_URL, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get("translated_text", "")
                if translated:
                    return chunk_idx, translated
            elif resp.status_code == 429:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                log.warning(f"Sarvam translation API HTTP {resp.status_code}: {resp.text[:180]}")
        except Exception as e:
            log.warning(f"Sarvam translation chunk {chunk_idx} attempt {attempt+1} error: {e}")
            import time
            time.sleep(0.5)

    raise RuntimeError(f"Sarvam translation failed for chunk {chunk_idx}")


def translate_sarvam(
    text: str,
    source_lang_name: str,
    target_lang_name: str,
    mode: str = "formal",
) -> Optional[str]:
    """
    Translate text using Sarvam AI Mayura:v1 (Indic-focused neural translation).
    Automatically chunks inputs exceeding 1600 characters to strictly comply with
    Sarvam's 2000-character payload limit.
    """
    if not text or not text.strip():
        return text

    api_key = get_sarvam_api_key()
    if not api_key:
        return None

    src_code = SARVAM_LANGUAGES.get(source_lang_name, "en-IN")
    tgt_code = SARVAM_LANGUAGES.get(target_lang_name, "hi-IN")

    if src_code == tgt_code:
        return text

    chunks = _chunk_text_for_sarvam(text, max_chars=SARVAM_MAX_TRANSLATE_CHARS)
    if not chunks:
        return text

    if len(chunks) == 1:
        try:
            _, translated = _translate_sarvam_chunk(0, chunks[0], src_code, tgt_code, mode, api_key)
            return translated
        except Exception as e:
            log.warning(f"Sarvam translation error: {e}")
            return None

    # Multi-chunk concurrent translation
    chunk_results = [None] * len(chunks)
    max_workers = min(4, len(chunks))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _translate_sarvam_chunk, idx, chunk, src_code, tgt_code, mode, api_key
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx, translated = future.result()
                chunk_results[idx] = translated

        if any(r is None for r in chunk_results):
            log.warning("One or more Sarvam translation chunks failed.")
            return None

        return " ".join(chunk_results)
    except Exception as e:
        log.warning(f"Sarvam parallel translation failed: {e}")
        return None

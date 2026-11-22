"""
Bhasha-Setu — Pipeline Orchestrator
Runs all 5 stages with granular sub-stage progress reporting.
Supports: SRT generation, LLM translation polishing, preview mode,
          any-source-language → any-target-language.

KEY FIXES:
  1. source_language parameter threads through all stages.
  2. Detected language from transcribe is passed to translate as source.
  3. English-only error guard removed — any language with speech is valid.
"""

import os
import re
import uuid
import logging
import logging.handlers

from pipeline.config import (
    OUTPUT_DIR, LOG_DIR, LANGUAGES,
    SOURCE_LANGUAGES, TRANSCRIBE_TO_TRANSLATE_CODE,
)
from pipeline.storage    import upload_to_s3, delete_from_s3
from pipeline.transcribe import transcribe_video
from pipeline.translate  import translate_text, polish_translation_with_llm
from pipeline.synthesize import synthesize_speech
from pipeline.mux        import mux
from pipeline.concurrency import acquire_dubbing_slot

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)


from typing import Optional

def _safe_cleanup_temp(temp_path: str):
    """Safely delete downloaded temp files and folders without risking system temp dirs."""
    if not temp_path or not os.path.exists(temp_path):
        return
    try:
        parent = os.path.dirname(os.path.abspath(temp_path))
        parent_name = os.path.basename(parent)
        if parent_name.startswith("bhasha_setu_") and os.path.isdir(parent):
            import shutil
            shutil.rmtree(parent, ignore_errors=True)
        elif os.path.isfile(temp_path):
            os.remove(temp_path)
    except Exception as e:
        log.warning(f"Failed to cleanup temp file {temp_path}: {e}")

def _setup_logging():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)
    # Check if we already added our handler
    if not any(getattr(h, '_bhasha_setu', False) for h in pipeline_logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch._bhasha_setu = True
        pipeline_logger.addHandler(ch)
        try:
            fh = logging.handlers.RotatingFileHandler(
                os.path.join(LOG_DIR, "pipeline.log"),
                maxBytes=5_000_000, backupCount=3,
            )
            fh.setFormatter(fmt)
            fh._bhasha_setu = True
            pipeline_logger.addHandler(fh)
        except Exception:
            pass


log = logging.getLogger(__name__)


def _resolve_source_lang_code(source_language_name: Optional[str]) -> Optional[str]:
    """
    Convert a display name (e.g. "Hindi") to a Transcribe BCP-47 code (e.g. "hi-IN").
    Returns None for 'Auto-detect' or None input (triggers IdentifyLanguage=True).
    """
    if not source_language_name or source_language_name == "Auto-detect":
        return None
    return SOURCE_LANGUAGES.get(source_language_name)


def run_transcribe_and_translate(
    video_path:          str  = "",
    target_language:     str  = "",
    progress_cb=None,
    generate_srt:        bool = False,
    polish_translation:  bool = False,
    video_url:           str  = "",
    words_per_subtitle:  int  = 8,
    source_language:     str  = "Auto-detect",
    translate_engine:    str  = "auto",
    tts_engine:          str  = "auto",
    sarvam_speaker:      str  = "meera",
) -> dict:
    """
    Phase 1 of the pipeline for human-in-the-loop review:
    upload → transcribe (any lang) → translate (any → any) → optional SRT.

    Accepts either a local video_path OR a video_url.

    Returns a dict with transcript, translation, detected_language,
    SRT path, and video_path for HIL Phase 2.
    """
    _setup_logging()

    _url_mode    = bool(video_url and not video_path)
    _temp_to_del = ""

    if not _url_mode and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if target_language not in LANGUAGES:
        raise ValueError(f"Unsupported target language: {target_language!r}")

    job_id     = uuid.uuid4().hex[:8]
    lang_cfg   = LANGUAGES[target_language]
    lang_code  = lang_cfg["aws_translate_code"]
    src_tc     = _resolve_source_lang_code(source_language)  # Transcribe code or None
    src_label  = video_url if _url_mode else video_path
    log.info(f"=== HIL PHASE-1 {job_id} | {source_language} → {target_language} | {src_label} ===")

    def _prog(stage: int, pct: float, msg: str):
        log.info(f"[Stage {stage}/5 | {pct:.0f}%] {msg}")
        if progress_cb:
            try: progress_cb(stage, pct, msg)
            except Exception: pass

    with acquire_dubbing_slot(job_id=job_id, status_cb=lambda m: _prog(1, 0, m)):
        # Stage 1: Fetch (URL) or Upload
        if _url_mode:
            _prog(1, 5, "Connecting to video source…")
            def _dl_prog(pct: float, msg: str):
                _prog(1, 5 + pct * 0.6, msg)
            from pipeline.downloader import download_to_temp
            video_path   = download_to_temp(video_url, progress_cb=_dl_prog)
            _temp_to_del = video_path
            _prog(1, 70, "Uploading to S3…")
        else:
            _prog(1, 10, "Connecting to S3…")
            _prog(1, 40, f"Uploading {os.path.basename(video_path)}…")

        clean_base = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(video_path.replace("\\", "/")))
        s3_key = f"inputs/{job_id}_{clean_base}"
        s3_uri = upload_to_s3(video_path, s3_key)
        _prog(1, 100, f"Uploaded → {s3_uri}")

        try:
            # Stage 2: Transcribe (any language)
            lang_note = f"({source_language})" if source_language != "Auto-detect" else "(auto-detect)"
            _prog(2, 5, f"Starting transcription job… {lang_note}")

            def transcribe_progress(elapsed_pct: float, status: str):
                _prog(2, min(95, elapsed_pct), f"AWS Transcribe: {status}")

            transcript, word_timestamps, detected_tc = transcribe_video(
                s3_uri,
                progress_cb=transcribe_progress,
                return_timestamps=generate_srt,
                source_language_code=src_tc,
            )
            _prog(2, 100, f"Transcript ready ({len(transcript or '')} chars) — detected: {detected_tc}")
        finally:
            # Remove video from S3 as soon as Transcribe has ingested it
            delete_from_s3(s3_key)

        # KEY FIX: Removed English-only guard. Any language is valid.
        if not transcript or not transcript.strip():
            raise ValueError(
                "No speech detected in this video.\n\n"
                "Possible reasons:\n"
                "• The video has background music but no spoken words\n"
                "• The audio is too quiet or muffled\n"
                "• The video has no audio track\n\n"
                "Please use a video with clear spoken speech."
            )

        # Resolve source translate code from detected Transcribe language
        detected_translate_code = TRANSCRIBE_TO_TRANSLATE_CODE.get(detected_tc, "en")
        log.info(f"Transcribe lang: {detected_tc} → Translate source: {detected_translate_code}")

        # Stage 3: Translate (any → any)
        _prog(3, 5, f"Translating {detected_translate_code} → {lang_code}…")

        def translate_progress(chunk_i: int, total_chunks: int):
            pct = 10 + (chunk_i / max(1, total_chunks)) * 75
            _prog(3, pct, f"Translating chunk {chunk_i}/{total_chunks}…")

        translation = translate_text(
            transcript,
            target_lang_code=lang_code,
            source_lang_code=detected_translate_code,   # KEY FIX: was always "en"
            progress_cb=translate_progress,
            translate_engine=translate_engine,
        )

        if not translation:
            log.warning("Translation result was empty or None; defaulting to transcript.")
            translation = transcript

        if polish_translation:
            _prog(3, 90, "LLM polishing translation…")
            try:
                polished = polish_translation_with_llm(
                    translation, target_language, transcript,
                    source_language=source_language or "the source language",
                )
                if polished:
                    translation = polished
                _prog(3, 100, "Translation polished")
            except Exception as e:
                log.warning(f"LLM polish failed (non-fatal): {e}")
                _prog(3, 100, "Translation complete (polish skipped)")
        else:
            _prog(3, 100, f"Translation ready ({len(translation or '')} chars)")

        srt_path = ""
        if generate_srt and word_timestamps:
            try:
                from pipeline.transcribe import generate_srt_file
                srt_path = os.path.join(OUTPUT_DIR, f"{job_id}_subtitles.srt")
                generate_srt_file(word_timestamps, srt_path, words_per_subtitle=words_per_subtitle)
            except Exception as e:
                log.warning(f"SRT generation failed (non-fatal): {e}")

        return {
            "job_id":                  job_id,
            "video_path":              video_path,
            "language":                target_language,
            "source_language":         source_language,
            "detected_language_code":  detected_tc,
            "transcript":              transcript,
            "translation":             translation,
            "srt_path":                srt_path,
            "_temp_path":              _temp_to_del,
        }


def run_tts_and_mux(
    video_path:      str,
    target_language: str,
    final_text:      str,
    job_id:          str | None = None,
    progress_cb=None,
    srt_path:        str = "",
    voice_pitch:     int = 0,
    vol_boost:       float = 2.0,
    bg_music_vol:    float = 0.0,
    tts_engine:      str = "auto",
    sarvam_speaker:  str = "meera",
) -> dict:
    """
    Phase 2 of the pipeline for human-in-the-loop:
    synthesize speech from reviewed text and mux with video.
    """
    _setup_logging()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if target_language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {target_language!r}")

    if job_id is None:
        job_id = uuid.uuid4().hex[:8]

    lang_cfg = LANGUAGES[target_language]
    log.info(f"=== HIL PHASE-2 {job_id} | {target_language} | {video_path} ===")

    def _prog(stage: int, pct: float, msg: str):
        log.info(f"[Stage {stage}/5 | {pct:.0f}%] {msg}")
        if progress_cb:
            try: progress_cb(stage, pct, msg)
            except Exception: pass

    with acquire_dubbing_slot(job_id=job_id, status_cb=lambda m: _prog(4, 0, m)):
        # Stage 4: Synthesize
        _prog(4, 5, f"Initializing {lang_cfg['tts'].upper()} engine…")

        def synth_progress(chunk_i: int, total_chunks: int):
            pct = 10 + (chunk_i / max(1, total_chunks)) * 85
            _prog(4, pct, f"Synthesizing chunk {chunk_i}/{total_chunks}…")

        mp3_path = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.mp3")
        try:
            synthesize_speech(
                final_text, mp3_path, target_language,
                progress_cb=synth_progress, pitch_percent=voice_pitch,
                tts_engine=tts_engine, sarvam_speaker=sarvam_speaker,
            )
            _prog(4, 100, f"Audio synthesized")

            # Stage 5: Mux
            _prog(5, 10, "Converting MP3 → WAV and muxing…")

            def mux_progress(step_pct: float, msg: str):
                _prog(5, step_pct, msg)

            output_path = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.mp4")
            mux(video_path, mp3_path, output_path,
                progress_cb=mux_progress, vol_boost=vol_boost, bg_music_vol=bg_music_vol)
            _prog(5, 100, f"Mux complete → {os.path.basename(output_path)}")
        finally:
            try: os.remove(mp3_path)
            except Exception: pass

        return {
            "job_id":      job_id,
            "output_path": output_path,
            "language":    target_language,
            "translation": final_text,
            "srt_path":    srt_path,
        }


def run_pipeline(
    video_path:          str   = "",
    target_language:     str   = "",
    progress_cb=None,
    generate_srt:        bool  = False,
    polish_translation:  bool  = False,
    voice_pitch:         int   = 0,
    vol_boost:           float = 2.0,
    video_url:           str   = "",
    bg_music_vol:        float = 0.0,
    words_per_subtitle:  int   = 8,
    source_language:     Optional[str] = "Auto-detect",
    translate_engine:    str   = "auto",
    tts_engine:          str   = "auto",
    sarvam_speaker:      str   = "meera",
) -> dict:
    """
    Run the full Bhasha-Setu any-to-any dubbing pipeline.

    Accepts either video_path (upload) or video_url (YouTube/Vimeo/etc).

    Args:
        video_path:         Local path to source video.
        target_language:    Target language name (e.g. "Hindi", "Tamil").
        progress_cb:        Optional callable(stage, sub_pct, message).
        generate_srt:       Generate .srt subtitle file.
        polish_translation: LLM pass to improve translation naturalness.
        voice_pitch:        Pitch shift percent (-20..+20).
        vol_boost:          Volume multiplier (default 2.0).
        video_url:          Public video URL.
        bg_music_vol:       Background music volume (0.0 = off).
        words_per_subtitle: Words per SRT block.
        source_language:    Source language name or "Auto-detect".

    Returns:
        dict: output_path, transcript, translation, job_id, language, srt_path,
              source_language, detected_language_code
    """
    _setup_logging()

    _url_mode    = bool(video_url and not video_path)
    _temp_to_del = ""

    if not _url_mode and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if target_language not in LANGUAGES:
        raise ValueError(f"Unsupported target language: {target_language!r}")

    job_id    = uuid.uuid4().hex[:8]
    lang_cfg  = LANGUAGES[target_language]
    lang_code = lang_cfg["aws_translate_code"]
    src_tc    = _resolve_source_lang_code(source_language)
    src_label = video_url if _url_mode else video_path
    log.info(f"=== JOB {job_id} | {source_language} → {target_language} | {src_label} ===")

    def _prog(stage: int, pct: float, msg: str):
        log.info(f"[Stage {stage}/5 | {pct:.0f}%] {msg}")
        if progress_cb:
            try: progress_cb(stage, pct, msg)
            except Exception: pass

    s3_key = ""
    try:
        with acquire_dubbing_slot(job_id=job_id, status_cb=lambda m: _prog(1, 0, m)):
            # Stage 1: Fetch + Upload
            if _url_mode:
                _prog(1, 5, "Connecting to video source…")
                def _dl_prog(pct: float, msg: str):
                    _prog(1, 5 + pct * 0.60, msg)
                from pipeline.downloader import download_to_temp
                video_path   = download_to_temp(video_url, progress_cb=_dl_prog)
                _temp_to_del = video_path
                _prog(1, 70, "Uploading to S3…")
            else:
                _prog(1, 10, "Connecting to S3…")
                _prog(1, 40, f"Uploading {os.path.basename(video_path)}…")

            clean_base = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(video_path.replace("\\", "/")))
            s3_key = f"inputs/{job_id}_{clean_base}"
            s3_uri = upload_to_s3(video_path, s3_key)
            _prog(1, 100, f"Uploaded → {s3_uri}")

            # Record provenance node in Neo4j Knowledge Graph
            try:
                from pipeline.graph_db import record_video_pipeline_start
                record_video_pipeline_start(job_id, clean_base, source_language or "Auto-detect", target_language, video_path)
            except Exception as ge:
                log.debug(f"Neo4j record start skipped: {ge}")

            try:
                # Stage 2: Transcribe (any language)
                lang_note = f"({source_language})" if source_language != "Auto-detect" else "(auto-detect)"
                _prog(2, 5, f"Starting transcription… {lang_note}")

                def transcribe_progress(elapsed_pct: float, status: str):
                    _prog(2, min(95, elapsed_pct), f"AWS Transcribe: {status}")

                transcript, word_timestamps, detected_tc = transcribe_video(
                    s3_uri,
                    progress_cb=transcribe_progress,
                    return_timestamps=generate_srt,
                    source_language_code=src_tc,
                )
                _prog(2, 100, f"Transcript ready ({len(transcript or '')} chars) — detected: {detected_tc}")

                # Record transcript & extract mentioned Indic entities in Neo4j
                try:
                    from pipeline.graph_db import record_transcript_event
                    record_transcript_event(job_id, transcript, detected_tc, word_timestamps)
                except Exception as ge:
                    log.debug(f"Neo4j transcript logging skipped: {ge}")
            finally:
                # Delete staging video from S3 once transcription is complete or if it failed
                if s3_key:
                    delete_from_s3(s3_key)

            if not transcript or not transcript.strip():
                raise ValueError(
                    "No speech detected in this video.\n\n"
                    "Possible reasons:\n"
                    "• The video has background music but no spoken words\n"
                    "• The audio is too quiet or muffled\n\n"
                    "Please use a video with clear spoken speech."
                )

            # Resolve translate source code
            detected_translate_code = TRANSCRIBE_TO_TRANSLATE_CODE.get(detected_tc, "en")

            # Stage 3: Translate (any → any)
            _prog(3, 5, f"Translating {detected_translate_code} → {lang_code}…")

            def translate_progress(chunk_i: int, total_chunks: int):
                pct = 10 + (chunk_i / max(1, total_chunks)) * 75
                _prog(3, pct, f"Translating chunk {chunk_i}/{total_chunks}…")

            translation = translate_text(
                transcript,
                target_lang_code=lang_code,
                source_lang_code=detected_translate_code,   # KEY FIX
                progress_cb=translate_progress,
                translate_engine=translate_engine,
            )

            if not translation:
                log.warning("Translation result was empty or None; defaulting to transcript.")
                translation = transcript

            if polish_translation:
                _prog(3, 90, "LLM polishing translation…")
                try:
                    polished = polish_translation_with_llm(
                        translation, target_language, transcript,
                        source_language=source_language or "source",
                    )
                    if polished:
                        translation = polished
                    _prog(3, 100, "Translation polished")
                except Exception as e:
                    log.warning(f"LLM polish failed: {e}")
                    _prog(3, 100, "Translation complete (polish skipped)")
            else:
                _prog(3, 100, f"Translation ready ({len(translation or '')} chars)")

            # Record translation node in Neo4j Knowledge Graph
            try:
                from pipeline.graph_db import record_translation_event
                record_translation_event(job_id, target_language, translation, engine=translate_engine or "Mayura:v1")
            except Exception as ge:
                log.debug(f"Neo4j translation logging skipped: {ge}")

            srt_path = ""
            if generate_srt and word_timestamps:
                try:
                    from pipeline.transcribe import generate_srt_file
                    srt_path = os.path.join(OUTPUT_DIR, f"{job_id}_subtitles.srt")
                    generate_srt_file(word_timestamps, srt_path, words_per_subtitle=words_per_subtitle)
                except Exception as e:
                    log.warning(f"SRT generation failed: {e}")

            # Stage 4: Synthesize
            _prog(4, 5, f"Initializing {lang_cfg['tts'].upper()} engine…")

            def synth_progress(chunk_i: int, total_chunks: int):
                pct = 10 + (chunk_i / max(1, total_chunks)) * 85
                _prog(4, pct, f"Synthesizing chunk {chunk_i}/{total_chunks}…")

            mp3_path = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.mp3")
            try:
                synthesize_speech(
                    translation, mp3_path, target_language,
                    progress_cb=synth_progress, pitch_percent=voice_pitch,
                    tts_engine=tts_engine, sarvam_speaker=sarvam_speaker,
                )
                _prog(4, 100, "Audio synthesized")

                # Stage 5: Mux
                _prog(5, 10, "Converting MP3 → WAV…")

                def mux_progress(step_pct: float, msg: str):
                    _prog(5, step_pct, msg)

                output_path = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.mp4")
                mux(video_path, mp3_path, output_path,
                    progress_cb=mux_progress, vol_boost=vol_boost, bg_music_vol=bg_music_vol)
                _prog(5, 100, f"Mux complete → {os.path.basename(output_path)}")

                # Complete full DAG provenance trace in Neo4j
                try:
                    from pipeline.graph_db import record_synthesis_event, complete_video_pipeline
                    record_synthesis_event(job_id, target_language, sarvam_speaker or "meera", engine=tts_engine or "Bulbul:v3", audio_path=output_path)
                    complete_video_pipeline(job_id, output_path)
                except Exception as ge:
                    log.debug(f"Neo4j completion logging skipped: {ge}")
            finally:
                if os.path.exists(mp3_path):
                    try: os.remove(mp3_path)
                    except Exception: pass
    finally:
        _safe_cleanup_temp(_temp_to_del)

    log.info(f"=== JOB {job_id} COMPLETE === Output: {output_path}")
    return {
        "job_id":                  job_id,
        "output_path":             output_path,
        "transcript":              transcript,
        "translation":             translation,
        "language":                target_language,
        "source_language":         source_language,
        "detected_language_code":  detected_tc,
        "srt_path":                srt_path,
    }

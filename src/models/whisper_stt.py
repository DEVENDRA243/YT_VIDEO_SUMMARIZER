# ===== PART B: SPEECH-TO-TEXT (WHISPER) =====

import os
import sys
import shutil
from typing import List, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AUDIO_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cache", "audio"
)

_whisper_asr_pipe = None


def get_whisper_pipeline():
    """
    Lazy-loads Whisper ASR pipeline using lightweight openai/whisper-tiny (~39MB).
    """
    global _whisper_asr_pipe
    if _whisper_asr_pipe is None:
        from transformers import pipeline
        _whisper_asr_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            chunk_length_s=30
        )
    return _whisper_asr_pipe


def download_audio(video_id: str) -> Optional[str]:
    """
    Downloads audio stream from YouTube using yt-dlp into data/cache/audio/.
    """
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    audio_path = os.path.join(AUDIO_CACHE_DIR, f"{video_id}.m4a")

    if os.path.exists(audio_path):
        return audio_path

    try:
        import yt_dlp
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': os.path.join(AUDIO_CACHE_DIR, f"{video_id}.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # Check for downloaded audio file
        for ext in ["m4a", "webm", "mp3", "wav"]:
            fpath = os.path.join(AUDIO_CACHE_DIR, f"{video_id}.{ext}")
            if os.path.exists(fpath):
                return fpath
    except Exception as e:
        print(f"[!] Warning: Audio download failed ({e}).")
        return None

    return None


def transcribe_video_audio(
    video_id: str,
    verbose: bool = True
) -> List[Dict]:
    """
    Main Part B Whisper STT function.
    Triggered when a YouTube video contains no captions.
    - Downloads audio stream
    - Runs Whisper speech-to-text with timestamps
    - Returns standard transcript chunk dictionaries: [{"text": ..., "start": ..., "duration": ...}]
    """
    if verbose:
        print("\n" + "=" * 50)
        print("===== PART B: SPEECH-TO-TEXT (WHISPER) =====")
        print("=" * 50)
        print(f"--- [BEFORE] Processing Video Audio with Whisper ---")
        print(f"  Target Video ID: {video_id}")
        print(f"  Model          : openai/whisper-tiny (CPU optimized)")

    audio_file = download_audio(video_id)
    if not audio_file or not os.path.exists(audio_file):
        raise RuntimeError(
            f"Cannot transcribe video '{video_id}'. YouTube captions were absent, "
            f"and audio extraction could not proceed (FFmpeg or network issue)."
        )

    asr = get_whisper_pipeline()
    result = asr(audio_file, return_timestamps=True)

    chunks = result.get("chunks", [])
    raw_segments = []

    for c in chunks:
        ts = c.get("timestamp", (0.0, 0.0))
        start = ts[0] if ts and ts[0] is not None else 0.0
        end = ts[1] if ts and len(ts) > 1 and ts[1] is not None else start + 2.0
        duration = max(round(end - start, 2), 0.5)

        raw_segments.append({
            "start": round(start, 2),
            "duration": duration,
            "text": c.get("text", "").strip()
        })

    if verbose:
        print(f"\n--- [AFTER] Whisper Transcription Complete ---")
        print(f"  Generated Segments: {len(raw_segments)}")
        if raw_segments:
            print(f"  First Segment: [{raw_segments[0]['start']}s]: \"{raw_segments[0]['text']}\"")
        print("=" * 50 + "\n")

    return raw_segments


if __name__ == "__main__":
    print("\n--- Testing Whisper STT Module Structure ---")
    print("Whisper module loaded cleanly. Ready to transcribe videos with missing subtitles.")

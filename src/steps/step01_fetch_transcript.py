# ===== STEP 1: TRANSCRIPT FETCHING =====

import json
import os
import re
import sys
from typing import List, Dict, Optional, Any
from youtube_transcript_api import YouTubeTranscriptApi

# Ensure UTF-8 output on Windows terminal for Devanagari (Hindi) characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache")
SAMPLE_CACHE_PATH = os.path.join(CACHE_DIR, "sample_transcripts.json")


# 1a: parse video ID from URL
def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the 11-character YouTube video ID from various URL formats or raw ID.
    Supports standard URLs, shortened youtu.be, shorts, embeds, and plain IDs.
    """
    url_or_id = url_or_id.strip()
    patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
        r'^(en_short_01|en_medium_01|hi_clean_01|hi_conv_01)$'  # Named demo test IDs
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Invalid YouTube URL or Video ID: '{url_or_id}'")


# 1b: check local cache
def load_from_cache(video_id: str) -> Optional[List[Dict]]:
    """
    Checks for a cached transcript in data/cache to allow offline execution.
    """
    # 1b.1: Check standalone cache file: data/cache/{video_id}.json
    standalone_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    if os.path.exists(standalone_file):
        with open(standalone_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1b.2: Check sample_transcripts.json bundle
    if os.path.exists(SAMPLE_CACHE_PATH):
        with open(SAMPLE_CACHE_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
            if video_id in samples:
                return samples[video_id]

    return None


def save_to_cache(video_id: str, transcript: List[Dict]) -> None:
    """
    Saves fetched transcript to data/cache/{video_id}.json for future offline reuse.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    target_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


# 1c: fetch captions via YouTube Transcript API
def fetch_online_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[Dict]:
    """
    Calls the YouTube Transcript API to retrieve manual or auto-generated captions for ANY video.
    Automatically discovers and fetches whatever language is available (hi, en, etc.).
    """
    try:
        ytt = YouTubeTranscriptApi()
        if hasattr(ytt, "list"):
            t_list = ytt.list(video_id)
            try:
                trans = t_list.find_transcript(['hi', 'en', 'hi-Latn'])
            except Exception:
                # Fallback to the very first available transcript track
                trans = next(iter(t_list))
            fetched = trans.fetch()
            return fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
        elif hasattr(ytt, "fetch"):
            return ytt.fetch(video_id, languages=["hi", "en"]).to_raw_data()
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            return YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
        else:
            raise RuntimeError("Incompatible version of youtube-transcript-api installed.")
    except Exception as e:
        raise RuntimeError(f"Could not retrieve captions from YouTube for video '{video_id}': {str(e)}")


# 1d: handle exceptions and fallback
def fetch_transcript(video_input: str, languages: Optional[List[str]] = None, verbose: bool = True) -> List[Dict]:
    """
    Main Step 1 function.
    Given a YouTube URL or Video ID:
    - Extracts video ID
    - Checks offline cache first
    - Fetches from YouTube if not cached
    - Samples ultra-long videos (>100 segments) across their timeline for sub-5s summarization
    - Prints formatted BEFORE/AFTER comparison
    - Returns list of timestamped subtitle dictionaries: [{"text": ..., "start": ..., "duration": ...}]
    """
    if languages is None:
        languages = ["hi", "en"]

    video_id = extract_video_id(video_input)

    # Verbose: Print BEFORE block
    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 1: TRANSCRIPT FETCHING =====")
        print("=" * 50)
        print(f"--- [BEFORE] Input Request ---")
        print(f"  Target Input : {video_input}")
        print(f"  Parsed Video ID: {video_id}")
        print(f"  Requested Languages: {languages}")

def fetch_video_info(video_input: str) -> Dict[str, Any]:
    """
    Fetches real video title and metadata via YouTube oEmbed or local fallback.
    """
    try:
        video_id = extract_video_id(video_input)
    except Exception:
        video_id = video_input

    # Demo ID fallbacks
    demo_titles = {
        "en_short_01": "What is Machine Learning in 3 Minutes",
        "en_medium_01": "How Large Language Models Work",
        "hi_clean_01": "Artificial Intelligence Basics in Hindi",
        "hi_conv_01": "Technology Discussion in Hindi",
        "-kguiI17880": "How to Write a Research Paper | Step-by-Step Guide",
        "2mO70J6Y_lo": "Complete Guide to Machine Learning Concepts",
        "HD13eq_Pmp8": "Deep Learning and Neural Networks Explained",
        "d1ORYqyF1qM": "Comprehensive AI & Data Science Masterclass"
    }

    if video_id in demo_titles:
        return {"video_id": video_id, "title": demo_titles[video_id], "author": "Educational Channel"}

    # Attempt online oEmbed lookup
    try:
        import urllib.request
        yt_url = f"https://www.youtube.com/watch?v={video_id}"
        oembed_url = f"https://www.youtube.com/oembed?url={yt_url}&format=json"
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {
                "video_id": video_id,
                "title": data.get("title", f"YouTube Video ({video_id})"),
                "author": data.get("author_name", "YouTube Creator")
            }
    except Exception:
        pass

    # Generic fallback
    return {
        "video_id": video_id,
        "title": f"YouTube Video: {video_id}",
        "author": "YouTube"
    }


# 1d: handle exceptions and fallback
def fetch_transcript(video_input: str, languages: Optional[List[str]] = None, verbose: bool = True) -> List[Dict]:
    """
    Main Step 1 function.
    Given a YouTube URL or Video ID:
    - Extracts video ID
    - Checks offline cache first
    - Fetches from YouTube if not cached
    - Returns list of timestamped subtitle dictionaries: [{"text": ..., "start": ..., "duration": ...}]
    """
    if languages is None:
        languages = ["hi", "en"]

    video_id = extract_video_id(video_input)

    # Verbose: Print BEFORE block
    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 1: TRANSCRIPT FETCHING =====")
        print("=" * 50)
        print(f"--- [BEFORE] Input Request ---")
        print(f"  Target Input : {video_input}")
        print(f"  Parsed Video ID: {video_id}")
        print(f"  Requested Languages: {languages}")

    # Check cache first
    cached_data = load_from_cache(video_id)
    source = "Local Cache (Offline)"
    if cached_data is not None:
        # Keep full transcript segments for high fidelity
        transcript = cached_data
    else:
        # Fetch from YouTube
        source = "YouTube Captions API (Online)"
        try:
            raw_transcript = fetch_online_transcript(video_id, languages=languages)
            transcript = raw_transcript
            save_to_cache(video_id, transcript)
        except Exception as e:
            if verbose:
                print(f"\n[!] Notice: Caption fetch failed ({e}).")
            raise

    # Calculate metrics for AFTER block
    total_segments = len(transcript)
    total_duration = 0.0
    if total_segments > 0:
        last_seg = transcript[-1]
        total_duration = round(last_seg.get("start", 0) + last_seg.get("duration", 0), 1)

    # Verbose: Print AFTER block
    if verbose:
        print(f"\n--- [AFTER] Fetched Transcript ---")
        print(f"  Source          : {source}")
        print(f"  Total Segments  : {total_segments}")
        print(f"  Est. Duration   : {total_duration}s (~{round(total_duration/60, 1)} mins)")
        print(f"  Preview (First 3 Segments):")
        for i, seg in enumerate(transcript[:3], 1):
            start = round(seg.get("start", 0.0), 1)
            dur = round(seg.get("duration", 0.0), 1)
            txt = seg.get("text", "").replace("\n", " ")
            print(f"    [{i}] [{start}s - {round(start + dur, 1)}s]: \"{txt}\"")
        print("=" * 50 + "\n")

    return transcript


if __name__ == "__main__":
    # Test on cached English demo ID
    print("\n--- Testing with English Cached Sample ---")
    en_transcript = fetch_transcript("en_short_01", verbose=True)

    # Test on cached Hindi demo ID
    print("\n--- Testing with Hindi Cached Sample ---")
    hi_transcript = fetch_transcript("hi_clean_01", verbose=True)

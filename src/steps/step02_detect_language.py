# ===== STEP 2: LANGUAGE DETECTION =====

import re
import sys
from typing import List, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def detect_language(transcript: List[Dict], verbose: bool = True) -> str:
    """
    Detects whether the transcript is in English ('en') or Hindi ('hi').
    Uses character script distribution:
    - Devanagari Unicode block: \u0900 - \u097F
    - Latin Alphabet: [a-zA-Z]
    """
    # 2a: aggregate text sample
    sample_text = " ".join([seg.get("text", "") for seg in transcript[:25]])

    # Count Devanagari and Latin characters
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', sample_text))
    latin_chars = len(re.findall(r'[a-zA-Z]', sample_text))
    total_letters = devanagari_chars + latin_chars

    # 2b: determine dominant language
    if total_letters == 0:
        detected_lang = "en"
        confidence = 0.0
    elif devanagari_chars > latin_chars:
        detected_lang = "hi"
        confidence = round(devanagari_chars / total_letters, 3)
    else:
        detected_lang = "en"
        confidence = round(latin_chars / total_letters, 3)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 2: LANGUAGE DETECTION =====")
        print("=" * 50)
        print("--- [BEFORE] Sample Text Preview ---")
        preview = sample_text[:120] + ("..." if len(sample_text) > 120 else "")
        print(f"  \"{preview}\"")
        print("\n--- [AFTER] Language Detection Result ---")
        lang_name = "Hindi (Devanagari)" if detected_lang == "hi" else "English (Latin)"
        print(f"  Detected Code   : {detected_lang}")
        print(f"  Language Name   : {lang_name}")
        print(f"  Script Counts   : Devanagari={devanagari_chars}, Latin={latin_chars}")
        print(f"  Confidence      : {confidence * 100:.1f}%")
        print("=" * 50 + "\n")

    return detected_lang


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript

    print("\n--- Testing Step 2 with English ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    detect_language(en_raw, verbose=True)

    print("\n--- Testing Step 2 with Hindi ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    detect_language(hi_raw, verbose=True)

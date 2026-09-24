# ===== STEP 3: CLEANING =====

import re
import sys
from typing import List, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Filler dictionaries
ENGLISH_FILLERS = [
    r'\b(?:um+)\b',
    r'\b(?:uh+)\b',
    r'\b(?:er+)\b',
    r'\bah+\b',
    r'\byou know\b',
    r'\bbasically\b',
    r'\bi mean\b',
    r'\bsort of\b',
    r'\bkind of\b'
]

HINDI_FILLERS = [
    r'\bमतलब\b',
    r'\bतो\b',
    r'\bअरे\b',
    r'\bयार\b',
    r'\bवैसे\b',
    r'\b(?:um+)\b',
    r'\b(?:uh+)\b'
]


# 3a: remove [Music] tags
def remove_sound_tags(text: str) -> str:
    """
    Removes audio transcription annotations such as [Music], [Applause], [संगीत], etc.
    """
    # Matches bracketed sound tags like [Music], [Applause], (cheering), [संगीत]
    text = re.sub(r'\[(?:Music|Applause|Laughter|Sound|संगीत|तालियां)[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((?:Music|Applause|Laughter|Sound)[^\)]*\)', '', text, flags=re.IGNORECASE)
    return text


# 3b: remove filler words
def remove_fillers(text: str, language: str = "en") -> str:
    """
    Removes conversational filler words based on detected language.
    """
    patterns = HINDI_FILLERS if language == "hi" else ENGLISH_FILLERS
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


# 3c: normalize whitespace
def normalize_spacing(text: str) -> str:
    """
    Collapses repeated whitespace, cleans dangling commas or colons, and trims.
    """
    text = re.sub(r'\s+', ' ', text)
    # Clean up isolated commas like "Today , we are" -> "Today, we are"
    text = re.sub(r'\s+([,.:;?!।])', r'\1', text)
    # Clean up commas left by removed fillers like "Today,, we" -> "Today, we"
    text = re.sub(r',+', ',', text)
    text = re.sub(r'^[,.\s]+', '', text)
    return text.strip()


def clean_transcript(transcript: List[Dict], language: str = "en", verbose: bool = True) -> List[Dict]:
    """
    Main Step 3 function.
    Cleans transcript segments by:
    - # 3a: removing sound tags
    - # 3b: removing verbal fillers
    - # 3c: normalizing whitespace
    Discards any empty chunks resulting from cleaning.
    """
    cleaned_segments = []
    before_samples = []
    after_samples = []

    for seg in transcript:
        orig_text = seg.get("text", "")
        # Apply cleaning pipeline
        t = remove_sound_tags(orig_text)
        t = remove_fillers(t, language=language)
        t = normalize_spacing(t)

        if t:  # Keep non-empty
            cleaned_seg = {
                "start": seg.get("start", 0.0),
                "duration": seg.get("duration", 0.0),
                "text": t
            }
            cleaned_segments.append(cleaned_seg)

        # Collect sample pairs for BEFORE/AFTER display
        if len(before_samples) < 3 and orig_text.strip() != t.strip():
            before_samples.append(orig_text)
            after_samples.append(t)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 3: CLEANING =====")
        print("=" * 50)
        print(f"--- [BEFORE] Raw Segments with Noise ---")
        if before_samples:
            for i, b in enumerate(before_samples, 1):
                print(f"  [{i}] \"{b}\"")
        else:
            print(f"  (Previewing first segment): \"{transcript[0].get('text', '')}\"")

        print(f"\n--- [AFTER] Cleaned Segments (Tags & Fillers Removed) ---")
        if after_samples:
            for i, a in enumerate(after_samples, 1):
                print(f"  [{i}] \"{a}\"")
        else:
            print(f"  (Previewing first cleaned segment): \"{cleaned_segments[0].get('text', '')}\"")

        print(f"\n  Segments Remaining: {len(cleaned_segments)} (from {len(transcript)} raw)")
        print("=" * 50 + "\n")

    return cleaned_segments


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language

    print("\n--- Testing Step 3 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    clean_transcript(en_raw, language=en_lang, verbose=True)

    print("\n--- Testing Step 3 with Hindi Conversational Sample ---")
    hi_raw = fetch_transcript("hi_conv_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    clean_transcript(hi_raw, language=hi_lang, verbose=True)

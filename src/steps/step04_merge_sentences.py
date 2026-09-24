# ===== STEP 4: MERGE SUBTITLES AND RESTORE PUNCTUATION =====

import re
import sys
from typing import List, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Known acronyms and proper names to restore case
ACRONYMS = {
    "iit": "IIT", "iits": "IITs", "nit": "NIT", "nits": "NITs",
    "ntu": "NTU", "ai": "AI", "ml": "ML", "nlp": "NLP",
    "llm": "LLM", "llms": "LLMs", "phd": "PhD", "usa": "USA", "us": "US",
    "uk": "UK", "mit": "MIT", "gpu": "GPU", "gpus": "GPUs",
    "cpu": "CPU", "cpus": "CPUs", "api": "API", "apis": "APIs",
    "youtube": "YouTube", "wiseup": "WiseUp"
}


def clean_and_punctuate_sentence(text: str, language: str = "en") -> str:
    """
    Cleans, restores casing, acronyms, and proper terminal punctuation.
    """
    text = text.strip()
    if not text:
        return ""

    # Collapse internal duplicate consecutive words like "the the"
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Clean stray spaces before punctuation: "word ," -> "word,"
    text = re.sub(r'\s+([,.:;?!।])', r'\1', text)
    text = re.sub(r'^[,.:;?!।\s]+', '', text).strip()
    if not text:
        return ""

    # Capitalize the first letter
    text = text[0].upper() + text[1:]

    # Capitalize standalone pronouns "I", "I'm", "I've", "I'll", "I'd"
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r"\bi'(m|ve|ll|d)\b", r"I'\1", text, flags=re.IGNORECASE)

    # Restore uppercase for domain acronyms
    for acr_lower, acr_proper in ACRONYMS.items():
        text = re.sub(r'\b' + re.escape(acr_lower) + r'\b', acr_proper, text, flags=re.IGNORECASE)

    # Ensure valid terminal punctuation
    terminal_chars = ('.', '?', '!', '।')
    if not text.endswith(terminal_chars):
        if language == "hi":
            text += " ।"
        else:
            text += "."

    return text


def is_sentence_terminal(text: str) -> bool:
    """
    Checks if a chunk terminates an existing sentence.
    """
    text = text.strip()
    return text.endswith(('.', '?', '!', '।'))


INCOMPLETE_ENDINGS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'so', 'for', 'with', 'that', 'at',
    'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'i', "i'm", 'im', "i've", 'ive', 'you', 'we', 'they', 'he', 'she', 'it',
    'my', 'your', 'our', 'their', 'his', 'her', 'this', 'these', 'those', 'which', 'who', 'whom',
    'but', 'because', 'just', 'from', 'into', 'by', 'on', 'about'
}


def merge_subtitles_to_sentences(
    cleaned_segments: List[Dict],
    language: str = "en",
    max_pause_seconds: float = 0.75,
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 4 function.
    Merges fragmented subtitle chunks into grammatically whole, punctuated sentences
    while strictly preserving sentence start and end timestamps.
    """
    merged_sentences = []
    current_text_parts = []
    current_start = 0.0
    current_end = 0.0

    for i, seg in enumerate(cleaned_segments):
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0.0)
        seg_duration = seg.get("duration", 0.0)
        seg_end = seg_start + seg_duration

        if not seg_text:
            continue

        if not current_text_parts:
            current_start = seg_start
            current_end = seg_end
            current_text_parts.append(seg_text)
        else:
            pause = seg_start - current_end
            curr_full_text = " ".join(current_text_parts).strip()
            words_list = curr_full_text.split()
            word_count = len(words_list)
            last_word = words_list[-1].lower().strip(".,;:?!") if words_list else ""

            is_dangling = last_word in INCOMPLETE_ENDINGS

            # Evaluate sentence completion conditions:
            # 1. Existing terminal punctuation
            # 2. Meaningful pause (>= 0.75s) with a minimum word count, not dangling
            # 3. Moderate pause (>= 0.4s) when sentence already has 14+ words, not dangling
            # 4. Maximum word cap (24 words)
            should_break = False
            if is_sentence_terminal(current_text_parts[-1]):
                should_break = True
            elif word_count >= 24:
                should_break = True
            elif not is_dangling:
                if pause >= max_pause_seconds and word_count >= 8:
                    should_break = True
                elif word_count >= 14 and pause >= 0.4:
                    should_break = True

            if should_break:
                final_sentence = clean_and_punctuate_sentence(curr_full_text, language=language)
                if len(final_sentence) > 3:
                    merged_sentences.append({
                        "sentence_id": len(merged_sentences) + 1,
                        "text": final_sentence,
                        "start": round(current_start, 2),
                        "end": round(current_end, 2),
                        "duration": round(current_end - current_start, 2)
                    })
                current_text_parts = [seg_text]
                current_start = seg_start
                current_end = seg_end
            else:
                current_text_parts.append(seg_text)
                current_end = seg_end


    # Handle remaining tail fragment
    if current_text_parts:
        final_sentence = clean_and_punctuate_sentence(" ".join(current_text_parts), language=language)
        if len(final_sentence) > 3:
            merged_sentences.append({
                "sentence_id": len(merged_sentences) + 1,
                "text": final_sentence,
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "duration": round(current_end - current_start, 2)
            })

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 4: MERGE SUBTITLES AND RESTORE PUNCTUATION =====")
        print("=" * 50)
        print("--- [BEFORE] Fragmented Subtitle Chunks ---")
        for i, seg in enumerate(cleaned_segments[:4], 1):
            print(f"  Chunk {i} [{round(seg.get('start', 0), 1)}s]: \"{seg.get('text', '')}\"")

        print(f"\n--- [AFTER] Merged Complete Sentences (with Timestamps) ---")
        for i, s in enumerate(merged_sentences[:3], 1):
            print(f"  Sentence {i} [{s['start']}s - {s['end']}s]: \"{s['text']}\"")

        print(f"\n  Reduced {len(cleaned_segments)} chunks into {len(merged_sentences)} complete sentences.")
        print("=" * 50 + "\n")

    return merged_sentences


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript

    print("\n--- Testing Step 4 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_clean = clean_transcript(en_raw, language="en", verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language="en", verbose=True)

    print("\n--- Testing Step 4 with Hindi Sample ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_clean = clean_transcript(hi_raw, language="hi", verbose=False)
    hi_sentences = merge_subtitles_to_sentences(hi_clean, language="hi", verbose=True)


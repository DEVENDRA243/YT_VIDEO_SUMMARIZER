# ===== STEP 5: SENTENCE AND WORD TOKENIZATION =====

import re
import sys
from typing import List, Dict, Tuple
import nltk

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 5a: sentence tokenization
def tokenize_sentences(text: str, language: str = "en") -> List[str]:
    """
    Splits text into sentences.
    Handles Hindi Devanagari Purna Viram (।) as well as standard delimiters (.?!).
    """
    text = text.strip()
    if not text:
        return []

    if language == "hi":
        # Split on Hindi purna viram (।) or Western punctuation followed by space
        sentences = re.split(r'(?<=[।?!.])\s+', text)
    else:
        # Standard English sentence tokenizer
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.?!])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


# 5b: word tokenization
def tokenize_words(sentence: str, language: str = "en") -> List[str]:
    """
    Splits a single sentence into word tokens.
    For Hindi, extracts Devanagari word tokens, Latin loan words, and numbers.
    For English, uses NLTK's word_tokenize.
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    if language == "hi":
        # Extract Devanagari words (excluding Purna Viram \u0964 and Deergh Viram \u0965), English terms, and digits
        tokens = re.findall(r'[\u0900-\u0963\u0966-\u097Fa-zA-Z0-9_]+', sentence)
    else:
        try:
            # Filter to keep word tokens (strip pure punctuation tokens)
            raw_tokens = nltk.word_tokenize(sentence)
            tokens = [t for t in raw_tokens if re.search(r'\w', t)]
        except Exception:
            tokens = re.findall(r'\b\w+\b', sentence)

    return tokens


def tokenize_transcript(
    merged_sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 5 function.
    Given merged sentence objects from Step 4:
    - # 5a: performs sentence tokenization validation
    - # 5b: performs word tokenization on each sentence
    Adds 'words' and 'word_count' to each sentence dictionary.
    """
    tokenized_data = []
    total_words = 0

    for item in merged_sentences:
        sent_text = item.get("text", "")
        # # 5b: word tokenization
        words = tokenize_words(sent_text, language=language)
        total_words += len(words)

        tokenized_item = {
            "sentence_id": item.get("sentence_id", len(tokenized_data) + 1),
            "text": sent_text,
            "start": item.get("start", 0.0),
            "end": item.get("end", 0.0),
            "duration": item.get("duration", 0.0),
            "words": words,
            "word_count": len(words)
        }
        tokenized_data.append(tokenized_item)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 5: SENTENCE AND WORD TOKENIZATION =====")
        print("=" * 50)
        print("--- [BEFORE] Sentence Input Strings ---")
        for item in merged_sentences[:2]:
            print(f"  Sentence {item.get('sentence_id')}: \"{item.get('text')}\"")

        print(f"\n--- [AFTER] Tokenized Output ---")
        print(f"  Total Sentences : {len(tokenized_data)}")
        print(f"  Total Word Tokens: {total_words}")
        print("  Sample Tokenized Breakdown:")
        for item in tokenized_data[:2]:
            print(f"    Sentence {item['sentence_id']} ({item['word_count']} tokens):")
            print(f"      Tokens: {item['words'][:8]}...")
        print("=" * 50 + "\n")

    return tokenized_data


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences

    print("\n--- Testing Step 5 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=True)

    print("\n--- Testing Step 5 with Hindi Sample ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    hi_clean = clean_transcript(hi_raw, language=hi_lang, verbose=False)
    hi_sentences = merge_subtitles_to_sentences(hi_clean, language=hi_lang, verbose=False)
    hi_tokens = tokenize_transcript(hi_sentences, language=hi_lang, verbose=True)

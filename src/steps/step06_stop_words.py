# ===== STEP 6: STOP WORD REMOVAL =====

import sys
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Comprehensive Hindi Stop Words list
HINDI_STOPWORDS: Set[str] = {
    "का", "के", "की", "को", "में", "से", "पर", "और", "है", "हैं", "था", "थी", "थे",
    "होता", "होती", "होते", "किया", "किए", "गया", "गई", "गए", "भी", "ही", "एक",
    "यह", "वह", "इस", "उस", "इन", "उन", "हम", "आप", "तुम", "वे", "जो", "कर", "ने",
    "नहीं", "तो", "या", "एवं", "तथा", "द्वारा", "लेकिन", "किन्तु", "परन्तु", "लिए",
    "सकता", "सकती", "सकते", "अपना", "अपनी", "अपने", "सब", "कुछ", "कोई", "जब", "तब"
}


# 6a: load language stop words
def get_stopwords(language: str = "en") -> Set[str]:
    """
    Retrieves the stopword set for the specified language.
    """
    if language == "hi":
        return HINDI_STOPWORDS
    else:
        try:
            return set(stopwords.words("english"))
        except Exception:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))


# 6b: filter tokens
def filter_sentence_tokens(tokens: List[str], stopword_set: Set[str]) -> List[str]:
    """
    Filters out stop words and short symbols, retaining lowercased content tokens.
    """
    return [
        t for t in tokens
        if t.lower() not in stopword_set and len(t) > 1
    ]


def remove_stop_words(
    tokenized_sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 6 function.
    - # 6a: loads stopwords for the language
    - # 6b: filters tokens per sentence into 'content_words'
    Prints BEFORE/AFTER comparison of tokens.
    """
    stopword_set = get_stopwords(language=language)
    processed_sentences = []

    total_before = 0
    total_after = 0

    for item in tokenized_sentences:
        orig_words = item.get("words", [])
        total_before += len(orig_words)

        # 6b: filter tokens
        content_words = filter_sentence_tokens(orig_words, stopword_set)
        total_after += len(content_words)

        updated_item = dict(item)
        updated_item["content_words"] = content_words
        updated_item["content_word_count"] = len(content_words)
        processed_sentences.append(updated_item)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 6: STOP WORD REMOVAL =====")
        print("=" * 50)
        print("--- [BEFORE] All Tokens (Including Stop Words) ---")
        for item in tokenized_sentences[:2]:
            print(f"  Sentence {item.get('sentence_id')}: {item.get('words')}")

        print("\n--- [AFTER] Filtered Content Tokens ---")
        for item in processed_sentences[:2]:
            print(f"  Sentence {item.get('sentence_id')}: {item.get('content_words')}")

        reduction = round((1.0 - (total_after / max(total_before, 1))) * 100, 1)
        print(f"\n  Reduced {total_before} tokens to {total_after} content tokens ({reduction}% stop words removed).")
        print("=" * 50 + "\n")

    return processed_sentences


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences
    from step05_tokenize import tokenize_transcript

    print("\n--- Testing Step 6 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=True)

    print("\n--- Testing Step 6 with Hindi Sample ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    hi_clean = clean_transcript(hi_raw, language=hi_lang, verbose=False)
    hi_sentences = merge_subtitles_to_sentences(hi_clean, language=hi_lang, verbose=False)
    hi_tokens = tokenize_transcript(hi_sentences, language=hi_lang, verbose=False)
    hi_filtered = remove_stop_words(hi_tokens, language=hi_lang, verbose=True)

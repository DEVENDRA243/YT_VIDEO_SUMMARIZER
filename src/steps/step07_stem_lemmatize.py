# ===== STEP 7: STEMMING AND LEMMATIZATION =====

import sys
from typing import List, Dict, Tuple
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize English stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Common Hindi inflectional suffixes for Hindi rule-based stemming
HINDI_SUFFIXES = [
    "ात्मक", "ीयता", "कारी", "ीयाँ", "ियों", "ियां", "दार",
    "ाना", "ाकर", "ाता", "ाती", "ाते", "ेंगे", "ेंगी", "ूँगा",
    "ेगा", "ेगी", "ाया", "ायी", "ाये", "कर", "ना", "ता", "ती",
    "ते", "ों", "ें", "ी", "े", "ा"
]


# 7a: stemming
def stem_word(word: str, language: str = "en") -> str:
    """
    Stems a word by stripping grammatical inflection suffixes.
    Uses PorterStemmer for English and suffix-stripping for Hindi.
    """
    if language == "hi":
        w = word.strip()
        for suffix in HINDI_SUFFIXES:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                return w[:-len(suffix)]
        return w
    else:
        return stemmer.stem(word)


# 7b: lemmatization
def lemmatize_word(word: str, pos_hint: str = "v", language: str = "en") -> str:
    """
    Reduces word to its canonical dictionary base form (lemma).
    For English, uses WordNetLemmatizer with verb/noun POS hints.
    """
    if language == "hi":
        # In Hindi without full morphological dictionary, root extraction mirrors stemmer
        return stem_word(word, language="hi")
    else:
        # Check verb lemma first, then noun lemma
        lemma_v = lemmatizer.lemmatize(word.lower(), pos=wordnet.VERB)
        if lemma_v != word.lower():
            return lemma_v
        return lemmatizer.lemmatize(word.lower(), pos=wordnet.NOUN)


# 7c: side-by-side comparison
def build_comparison_table(sample_words: List[str], language: str = "en") -> List[Tuple[str, str, str]]:
    """
    Builds (original, stem, lemma) tuples for comparison.
    """
    results = []
    seen = set()
    for w in sample_words:
        w_lower = w.lower()
        if w_lower not in seen and len(w_lower) > 2:
            seen.add(w_lower)
            s = stem_word(w_lower, language=language)
            l = lemmatize_word(w_lower, language=language)
            results.append((w, s, l))
    return results


def stem_and_lemmatize(
    sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 7 function.
    Performs:
    - # 7a: stemming
    - # 7b: lemmatization
    - # 7c: side-by-side comparison display
    Appends 'lemmas' to each sentence for downstream vectorization.
    """
    all_content_words = []
    processed_sentences = []

    for item in sentences:
        words = item.get("content_words", item.get("words", []))
        all_content_words.extend(words)

        lemmas = [lemmatize_word(w, language=language) for w in words]
        stems = [stem_word(w, language=language) for w in words]

        updated_item = dict(item)
        updated_item["lemmas"] = lemmas
        updated_item["stems"] = stems
        processed_sentences.append(updated_item)

    # 7c: side-by-side comparison display
    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 7: STEMMING AND LEMMATIZATION =====")
        print("=" * 50)
        print("--- [BEFORE] Sample Content Words (Before Morphological Reduction) ---")
        preview_words = all_content_words[:8]
        print(f"  {preview_words}")

        print("\n--- [AFTER] Side-by-Side Comparison: Word vs Stem vs Lemma ---")
        comparison = build_comparison_table(all_content_words[:12], language=language)

        print(f"  {'Original Word':<20} | {'Stem (Rule Cut)':<20} | {'Lemma (True Root)':<20}")
        print("  " + "-" * 66)
        for orig, s, l in comparison[:8]:
            print(f"  {orig:<20} | {s:<20} | {l:<20}")
        print("=" * 50 + "\n")

    return processed_sentences


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences
    from step05_tokenize import tokenize_transcript
    from step06_stop_words import remove_stop_words

    print("\n--- Testing Step 7 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=False)
    stem_and_lemmatize(en_filtered, language=en_lang, verbose=True)

    print("\n--- Testing Step 7 with Hindi Sample ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    hi_clean = clean_transcript(hi_raw, language=hi_lang, verbose=False)
    hi_sentences = merge_subtitles_to_sentences(hi_clean, language=hi_lang, verbose=False)
    hi_tokens = tokenize_transcript(hi_sentences, language=hi_lang, verbose=False)
    hi_filtered = remove_stop_words(hi_tokens, language=hi_lang, verbose=False)
    stem_and_lemmatize(hi_filtered, language=hi_lang, verbose=True)

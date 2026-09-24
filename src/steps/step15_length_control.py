# ===== STEP 15: LENGTH CONTROL =====

import sys
from typing import List, Dict, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Length profiles: target sentence ratio and hard word cap
LENGTH_PROFILES = {
    "small": {
        "sentence_ratio": 0.15,
        "word_cap": 100,
        "min_sentences": 2
    },
    "medium": {
        "sentence_ratio": 0.25,
        "word_cap": 250,
        "min_sentences": 3
    },
    "long": {
        "sentence_ratio": 0.35,
        "word_cap": 500,
        "min_sentences": 5
    }
}


# 15a: calculate target sentence quota
def determine_sentence_quota(total_sentences: int, length_profile: str) -> int:
    """
    Computes target sentence count based on percentage ratio and profile bounds.
    """
    profile = LENGTH_PROFILES.get(length_profile.lower(), LENGTH_PROFILES["medium"])
    ratio = profile["sentence_ratio"]
    min_sents = profile["min_sentences"]

    target = int(round(total_sentences * ratio))
    # Ensure reasonable boundaries
    return max(min_sents, min(target, total_sentences))


# 15b: apply word cap enforcement
def enforce_word_cap(
    candidate_sentences: List[Dict],
    word_cap: int,
    max_sentences: int
) -> List[Dict]:
    """
    Selects top-ranked sentences up to max_sentences while strictly enforcing the word cap.
    """
    # Sort candidates by importance score first
    ranked = sorted(candidate_sentences, key=lambda s: s.get("composite_score", 0.0), reverse=True)

    chosen = []
    current_words = 0

    for s in ranked:
        if len(chosen) >= max_sentences:
            break

        words_in_sent = len(s.get("text", "").split())
        # Add if within word cap (or allow at least 1-2 sentences even if cap is tight)
        if current_words + words_in_sent <= word_cap or len(chosen) < 2:
            chosen.append(s)
            current_words += words_in_sent

    # Re-sort into chronological video timeline order
    return sorted(chosen, key=lambda s: s.get("start", 0.0))


def apply_length_control(
    ordered_sentences: List[Dict],
    length_option: str = "medium",
    verbose: bool = True
) -> Tuple[List[Dict], str]:
    """
    Main Step 15 function (Completes Part A: Extractive Summary).
    - # 15a: Calculates target sentence quota based on length mode (Small / Medium / Long)
    - # 15b: Enforces maximum word cap
    Returns: (summary_sentences_list, summary_text_string)
    """
    profile_key = length_option.lower()
    if profile_key not in LENGTH_PROFILES:
        profile_key = "medium"
    profile = LENGTH_PROFILES[profile_key]

    total_input_sents = len(ordered_sentences)
    total_input_words = sum(len(s.get("text", "").split()) for s in ordered_sentences)

    # 15a: sentence quota
    target_count = determine_sentence_quota(total_input_sents, profile_key)

    # 15b: word cap enforcement
    final_summary_sentences = enforce_word_cap(
        ordered_sentences,
        word_cap=profile["word_cap"],
        max_sentences=target_count
    )

    summary_text = " ".join(s.get("text", "") for s in final_summary_sentences)
    summary_words = len(summary_text.split())
    compression_ratio = round((1.0 - (summary_words / max(total_input_words, 1))) * 100, 1)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 15: LENGTH CONTROL =====")
        print("=" * 50)
        print("--- [BEFORE] Available Candidate Pool ---")
        print(f"  Total Sentences : {total_input_sents}")
        print(f"  Total Words     : {total_input_words}")
        print(f"  Chosen Profile  : {profile_key.upper()} (Cap: {profile['word_cap']} words, Target: ~{profile['sentence_ratio']*100:.0f}%)")

        print(f"\n--- [AFTER] Extractive Summary (Part A Final Result) ---")
        print(f"  Summary Sentences : {len(final_summary_sentences)} / {total_input_sents}")
        print(f"  Summary Word Count: {summary_words} words (Limit: {profile['word_cap']})")
        print(f"  Compression Ratio : {compression_ratio}% reduction")
        print("\n  Summary Text:")
        print(f"  \"{summary_text}\"")
        print("=" * 50 + "\n")

    return final_summary_sentences, summary_text


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences
    from step05_tokenize import tokenize_transcript
    from step06_stop_words import remove_stop_words
    from step07_stem_lemmatize import stem_and_lemmatize
    from step08_pos_tagging import tag_pos
    from step09_ner import extract_entities
    from step10_tfidf_keyphrases import extract_tfidf_features
    from step11_embeddings_similarity import compute_sentence_embeddings
    from step12_score_sentences import score_sentences
    from step13_detect_chapters import detect_chapters
    from step14_remove_redundancy import remove_redundancy_and_order

    print("\n--- Testing Step 15 with English Sample (Medium Length) ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=False)
    en_lemmatized = stem_and_lemmatize(en_filtered, language=en_lang, verbose=False)
    en_pos = tag_pos(en_lemmatized, language=en_lang, verbose=False)
    en_entities = extract_entities(en_pos, language=en_lang, verbose=False)
    en_tfidf, _ = extract_tfidf_features(en_entities, language=en_lang, verbose=False)
    en_embedded, sim_matrix = compute_sentence_embeddings(en_tfidf, language=en_lang, verbose=False)
    en_scored = score_sentences(en_embedded, sim_matrix, verbose=False)
    en_chapters = detect_chapters(en_scored, sim_matrix, verbose=False)
    en_ordered = remove_redundancy_and_order(en_scored, verbose=False)

    # Test all 3 length variations
    for mode in ["small", "medium", "long"]:
        print(f"\n--- Testing Length Mode: {mode.upper()} ---")
        apply_length_control(en_ordered, length_option=mode, verbose=True)

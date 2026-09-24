# ===== STEP 14: REDUNDANCY REMOVAL AND ORDER RESTORATION =====

import sys
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 14a: rank sentences by multi-signal score
def rank_candidates(sentences: List[Dict]) -> List[Dict]:
    """
    Ranks candidate sentences descending by their composite importance score.
    """
    return sorted(sentences, key=lambda s: s.get("composite_score", 0.0), reverse=True)


# 14b: filter redundant sentences using cosine similarity threshold
def filter_redundancy(
    ranked_candidates: List[Dict],
    similarity_threshold: float = 0.72
) -> Tuple[List[Dict], List[Dict]]:
    """
    Greedily selects top sentences while discarding any candidate that has
    cosine similarity > similarity_threshold with an already-selected sentence.
    """
    selected = []
    dropped_redundant = []

    for candidate in ranked_candidates:
        cand_emb = candidate.get("embedding")
        if cand_emb is None:
            selected.append(candidate)
            continue

        cand_vector = np.array(cand_emb).reshape(1, -1)
        is_duplicate = False

        for chosen in selected:
            chosen_vector = np.array(chosen["embedding"]).reshape(1, -1)
            sim = float(cosine_similarity(cand_vector, chosen_vector)[0][0])
            if sim >= similarity_threshold:
                is_duplicate = True
                dropped_redundant.append((candidate, chosen, round(sim, 3)))
                break

        if not is_duplicate:
            selected.append(candidate)

    return selected, dropped_redundant


# 14c: restore chronological video order
def restore_chronological_order(selected_sentences: List[Dict]) -> List[Dict]:
    """
    Re-sorts selected summary sentences by original video timeline (start timestamp).
    """
    return sorted(selected_sentences, key=lambda s: s.get("start", 0.0))


def remove_redundancy_and_order(
    scored_sentences: List[Dict],
    similarity_threshold: float = 0.72,
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 14 function.
    - # 14a: Ranks sentences by importance score
    - # 14b: Removes semantically redundant sentences (cosine sim > threshold)
    - # 14c: Restores chronological video order
    """
    if not scored_sentences:
        return []

    # 14a: rank
    ranked = rank_candidates(scored_sentences)

    # 14b: filter redundancy
    selected, dropped = filter_redundancy(ranked, similarity_threshold=similarity_threshold)

    # 14c: restore video narrative order
    ordered_sentences = restore_chronological_order(selected)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 14: REDUNDANCY REMOVAL AND ORDER RESTORATION =====")
        print("=" * 50)
        print("--- [BEFORE] Top Candidates Ranked by Raw Score ---")
        for i, s in enumerate(ranked[:3], 1):
            print(f"  {i}. [Score: {s.get('composite_score', 0):.3f}] S{s['sentence_id']}: \"{s['text'][:65]}...\"")

        print(f"\n--- [AFTER] Non-Redundant Sentences Restored to Video Order ---")
        if dropped:
            print(f"  Dropped {len(dropped)} redundant sentence(s) exceeding {similarity_threshold} similarity:")
            for cand, matched_with, sim in dropped[:2]:
                print(f"    - Dropped S{cand['sentence_id']} (Sim {sim:.2f} with S{matched_with['sentence_id']})")
        else:
            print("  All top candidate sentences were distinct and non-overlapping.")

        print(f"\n  Final Non-Redundant Pool: {len(ordered_sentences)} sentences (Chronological):")
        for s in ordered_sentences[:3]:
            print(f"    [{s['start']}s] S{s['sentence_id']}: \"{s['text']}\"")
        print("=" * 50 + "\n")

    return ordered_sentences


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

    print("\n--- Testing Step 14 with English Sample ---")
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
    remove_redundancy_and_order(en_scored, verbose=True)

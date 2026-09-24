# ===== STEP 13: CHAPTER DETECTION =====

import sys
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds into standard YouTube timestamp format (MM:SS).
    """
    total_sec = max(0, int(round(seconds)))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


# 13a: compute coherence window similarity
def compute_adjacent_coherence(sim_matrix: np.ndarray) -> List[float]:
    """
    Calculates semantic coherence between adjacent sentences.
    A sharp drop indicates a thematic topic shift.
    """
    n = sim_matrix.shape[0]
    if n <= 1:
        return []
    return [float(sim_matrix[i, i + 1]) for i in range(n - 1)]


# 13b: detect valley points as chapter boundaries
def find_chapter_boundaries(coherence_scores: List[float], min_gap: int = 3) -> List[int]:
    """
    Identifies valley points (local minima in coherence) that act as chapter start indices.
    """
    if not coherence_scores:
        return [0]

    boundaries = [0]
    mean_sim = float(np.mean(coherence_scores))
    std_sim = float(np.std(coherence_scores)) if len(coherence_scores) > 1 else 0.1
    threshold = max(mean_sim - (0.35 * std_sim), 0.20)

    last_boundary = 0
    for i in range(1, len(coherence_scores)):
        curr = coherence_scores[i]
        if (curr < threshold) and (i - last_boundary >= min_gap):
            boundaries.append(i + 1)
            last_boundary = i + 1

    return boundaries


# 13c: assign chapter titles and timestamps
def generate_chapter_title(chapter_sentences: List[Dict], chapter_idx: int, total_chapters: int) -> str:
    """
    Generates a concise, informative Title Case chapter title from top keywords.
    """
    if chapter_idx == 1:
        # Check if first chapter has distinct topic
        words = []
        for s in chapter_sentences:
            words.extend(s.get("content_words", []))
        counts = Counter(w.title() for w in words if len(w) > 3)
        top_terms = [w for w, _ in counts.most_common(2)]
        if top_terms:
            return f"Overview: {' & '.join(top_terms)}"
        return "Introduction & Overview"

    if chapter_idx == total_chapters:
        words = []
        for s in chapter_sentences:
            words.extend(s.get("content_words", []))
        counts = Counter(w.title() for w in words if len(w) > 3)
        top_terms = [w for w, _ in counts.most_common(2)]
        if top_terms:
            return f"Conclusion: {' & '.join(top_terms)}"
        return "Summary & Conclusion"

    words = []
    for s in chapter_sentences:
        words.extend(s.get("content_words", []))

    counts = Counter(w.title() for w in words if len(w) > 3 and w.lower() not in {"this", "video", "that", "also", "have", "with"})
    top_terms = [w for w, _ in counts.most_common(3)]

    if top_terms:
        return " & ".join(top_terms[:2])
    return f"Key Discussion {chapter_idx}"


def detect_chapters(
    sentences: List[Dict],
    sim_matrix: np.ndarray,
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 13 function.
    - Computes coherence and detects topic boundaries
    - Enforces duration rules:
      * Never show a single chapter for videos > 5 minutes
      * Videos > 10 minutes get >= 3 chapters
      * Last chapter starts in the last 25% of the video
      * Timestamps are strictly distinct and increasing
    """
    n = len(sentences)
    if n == 0:
        return []

    total_duration = sentences[-1].get("end", 0.0) if sentences else 0.0

    # Base semantic boundary detection
    coherence = compute_adjacent_coherence(sim_matrix) if sim_matrix.shape[0] == n else []
    min_gap = max(3, n // 15) if n >= 15 else 2
    boundary_indices = find_chapter_boundaries(coherence, min_gap=min_gap)

    # Target chapter count based on 3-5 minute intervals:
    # <= 4 mins: 1-2 chapters
    # 4 - 9 mins: 2-3 chapters
    # 9 - 16 mins (e.g. 11.8 min video): 3-4 chapters
    # 16 - 25 mins: 4-5 chapters
    if total_duration <= 240.0:
        target_num_chapters = 1 if total_duration <= 100 else 2
    elif total_duration <= 540.0:
        target_num_chapters = 2 if total_duration <= 360 else 3
    elif total_duration <= 1000.0:
        target_num_chapters = 3 if total_duration <= 750 else 4
    elif total_duration <= 1500.0:
        target_num_chapters = 4 if total_duration <= 1200 else 5
    else:
        target_num_chapters = min(7, max(5, int(round(total_duration / 300.0))))

    # For videos > 10 min, ensure >= 3 chapters
    if total_duration > 600.0:
        target_num_chapters = max(3, target_num_chapters)

    # Establish ideal temporal anchors
    interval = total_duration / target_num_chapters
    boundary_indices = [0]

    for k in range(1, target_num_chapters):
        ideal_time = k * interval
        # If last chapter, enforce that it starts in the last 25% of the video
        if k == target_num_chapters - 1 and total_duration > 120.0:
            ideal_time = max(ideal_time, 0.75 * total_duration)

        # Snap to nearest candidate sentence around ideal_time
        candidates = [
            idx for idx in range(boundary_indices[-1] + 2, n - (target_num_chapters - 1 - k))
        ]
        if candidates:
            best_idx = min(
                candidates,
                key=lambda idx: abs(sentences[idx].get("start", 0.0) - ideal_time)
            )
            boundary_indices.append(best_idx)

    # Deduplicate and sort
    boundary_indices = sorted(list(set(boundary_indices)))
    split_indices = boundary_indices + [n]
    total_chaps = len(boundary_indices)

    chapters = []
    last_assigned_time = -1.0

    for ch_num, (start_idx, end_idx) in enumerate(zip(split_indices[:-1], split_indices[1:]), 1):
        ch_sents = sentences[start_idx:end_idx]
        if not ch_sents:
            continue

        raw_start = ch_sents[0].get("start", 0.0)
        # Ensure strictly increasing start times
        if raw_start <= last_assigned_time:
            raw_start = last_assigned_time + 1.0
        last_assigned_time = raw_start

        ch_end_time = ch_sents[-1].get("end", raw_start)
        ch_title = generate_chapter_title(ch_sents, ch_num, total_chaps)

        chapters.append({
            "chapter_id": ch_num,
            "title": ch_title,
            "start_time": raw_start,
            "end_time": ch_end_time,
            "timestamp": format_timestamp(raw_start),
            "seconds": int(round(raw_start)),
            "sentence_ids": [s["sentence_id"] for s in ch_sents],
            "sentences": ch_sents
        })

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 13: CHAPTER DETECTION =====")
        print("=" * 50)
        print(f"  Duration: {total_duration:.1f}s across {n} sentences")
        print("\n--- [AFTER] Detected Chapters with Timestamps & Titles ---")
        for ch in chapters:
            print(f"  [{ch['timestamp']}] Chapter {ch['chapter_id']}: {ch['title']}")
            print(f"      (Sentences: {len(ch['sentences'])}, Window: {ch['start_time']}s - {ch['end_time']}s)")
        print("=" * 50 + "\n")

    return chapters



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

    print("\n--- Testing Step 13 with English Sample ---")
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
    detect_chapters(en_scored, sim_matrix, verbose=True)

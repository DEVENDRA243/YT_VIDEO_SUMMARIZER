# ===== STEP 12: MULTI-SIGNAL SENTENCE SCORING =====

import sys
import numpy as np
from typing import List, Dict, Tuple
import networkx as nx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Default signal weights (tuned on extractive summarization benchmarks)
DEFAULT_WEIGHTS = {
    "tfidf": 0.25,
    "textrank": 0.30,
    "position": 0.15,
    "entity": 0.15,
    "length": 0.15
}


# 12a: compute positional scores
def compute_position_scores(n_sentences: int) -> List[float]:
    """
    Computes inverted positional bias: sentences early in a discourse or
    at the very end typically contain thematic thesis/concluding statements.
    """
    if n_sentences == 0:
        return []
    scores = []
    for i in range(n_sentences):
        # Lead bias: first sentence gets 1.0, decaying linearly; final sentence gets slight boost
        pos_ratio = i / max(n_sentences - 1, 1)
        lead_score = 1.0 - (0.7 * pos_ratio)
        if i == n_sentences - 1:
            lead_score += 0.2  # Conclusion boost
        scores.append(round(min(lead_score, 1.0), 4))
    return scores


# 12b: compute entity and numerical scores
def compute_entity_num_scores(sentences: List[Dict]) -> List[float]:
    """
    Scores sentences based on presence of concrete facts: Named Entities and numbers.
    """
    scores = []
    for item in sentences:
        ent_count = item.get("entity_count", 0)
        has_num = any(char.isdigit() for char in item.get("text", ""))
        score = min((ent_count * 0.4) + (0.3 if has_num else 0.0), 1.0)
        scores.append(round(score, 4))
    return scores


# 12c: compute TextRank graph scores
def compute_textrank_scores(sim_matrix: np.ndarray, damping: float = 0.85) -> List[float]:
    """
    Constructs a graph where nodes are sentences and edges are cosine similarity weights.
    Applies Google PageRank (TextRank algorithm) to calculate centrality.
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Zero out self-loops
    adj_matrix = np.copy(sim_matrix)
    np.fill_diagonal(adj_matrix, 0)

    try:
        graph = nx.from_numpy_array(adj_matrix)
        pagerank_dict = nx.pagerank(graph, alpha=damping, max_iter=200, weight="weight")
        raw_scores = [pagerank_dict[i] for i in range(n)]
        # Min-max scale
        min_s, max_s = min(raw_scores), max(raw_scores)
        if max_s > min_s:
            return [round((s - min_s) / (max_s - min_s), 4) for s in raw_scores]
        return [1.0] * n
    except Exception:
        # Fallback to degree centrality if pagerank fails to converge
        row_sums = adj_matrix.sum(axis=1)
        max_sum = row_sums.max() if row_sums.max() > 0 else 1.0
        return [round(float(s / max_sum), 4) for s in row_sums]


def compute_length_scores(sentences: List[Dict]) -> List[float]:
    """
    Penalizes fragments (< 6 words) and run-on sentences (> 40 words).
    Optimal summary sentence length is 12-28 words.
    """
    scores = []
    for item in sentences:
        wc = item.get("word_count", len(item.get("text", "").split()))
        if wc < 6:
            score = 0.3
        elif 10 <= wc <= 32:
            score = 1.0
        elif wc <= 45:
            score = 0.7
        else:
            score = 0.4
        scores.append(score)
    return scores


# 12d: combine weighted multi-signal score
def score_sentences(
    sentences: List[Dict],
    sim_matrix: np.ndarray,
    weights: Dict[str, float] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 12 function.
    Combines 5 normalized linguistic signals:
    - # 12a: Position score
    - # 12b: Entity and numerical density
    - # 12c: TextRank graph centrality
    - TF-IDF topical score
    - Length optimality score
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    n = len(sentences)
    if n == 0:
        return []

    # Calculate individual signals
    pos_scores = compute_position_scores(n)
    ent_scores = compute_entity_num_scores(sentences)
    tr_scores = compute_textrank_scores(sim_matrix)
    len_scores = compute_length_scores(sentences)

    # Normalize TF-IDF scores
    raw_tfidf = [item.get("tfidf_score", 0.0) for item in sentences]
    max_tfidf = max(raw_tfidf) if raw_tfidf and max(raw_tfidf) > 0 else 1.0
    norm_tfidf = [round(s / max_tfidf, 4) for s in raw_tfidf]

    # Combine weighted scores
    scored_sentences = []
    for i in range(n):
        c_score = (
            weights["tfidf"] * norm_tfidf[i] +
            weights["textrank"] * tr_scores[i] +
            weights["position"] * pos_scores[i] +
            weights["entity"] * ent_scores[i] +
            weights["length"] * len_scores[i]
        )

        item = dict(sentences[i])
        item["signal_scores"] = {
            "tfidf": norm_tfidf[i],
            "textrank": tr_scores[i],
            "position": pos_scores[i],
            "entity": ent_scores[i],
            "length": len_scores[i]
        }
        item["composite_score"] = round(float(c_score), 4)
        scored_sentences.append(item)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 12: MULTI-SIGNAL SENTENCE SCORING =====")
        print("=" * 50)
        print("--- [BEFORE] Unranked Sentences ---")
        for s in scored_sentences[:2]:
            print(f"  Sentence {s['sentence_id']}: \"{s['text'][:70]}...\"")

        print("\n--- [AFTER] Multi-Signal Composite Scores (Top 3 Sentences) ---")
        ranked = sorted(scored_sentences, key=lambda x: x["composite_score"], reverse=True)
        for rank, s in enumerate(ranked[:3], 1):
            signals = s["signal_scores"]
            print(f"  Rank {rank} (Score: {s['composite_score']:.3f}) [S{s['sentence_id']}]: \"{s['text']}\"")
            print(f"    -> TFIDF={signals['tfidf']} | TextRank={signals['textrank']} | Pos={signals['position']} | Ent={signals['entity']}")
        print("=" * 50 + "\n")

    return scored_sentences


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

    print("\n--- Testing Step 12 with English Sample ---")
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
    score_sentences(en_embedded, sim_matrix, verbose=True)

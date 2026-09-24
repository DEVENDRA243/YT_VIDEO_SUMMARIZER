# ===== STEP 11: WORD AND SENTENCE EMBEDDINGS =====

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")

_embedding_model = None


def get_embedding_model():
    """
    Lazy loader for SentenceTransformer model (all-MiniLM-L6-v2).
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# 11a: compute sentence embeddings
def embed_sentences(sentences: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
    """
    Encodes each sentence into a 384-dimensional dense semantic vector in batches.
    """
    texts = [item.get("text", "") for item in sentences]
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

    for idx, item in enumerate(sentences):
        item["embedding"] = embeddings[idx]

    return sentences, embeddings


# 11b: compute pairwise cosine similarity matrix
def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes NxN pairwise cosine similarity between sentence embeddings.
    """
    return cosine_similarity(embeddings, embeddings)


# 11c: generate and save similarity heatmap
def save_similarity_heatmap(sim_matrix: np.ndarray, output_path: str) -> None:
    """
    Generates and saves a seaborn heatmap illustrating semantic similarity between sentences.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_sentences = sim_matrix.shape[0]
    # Downsample matrix for rendering if sentence count is very large
    render_matrix = sim_matrix[:20, :20] if n_sentences > 20 else sim_matrix
    n_sub = render_matrix.shape[0]

    plt.figure(figsize=(7, 5.5))
    labels = [f"S{i+1}" for i in range(n_sub)]

    ax = sns.heatmap(
        render_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlGnBu",
        annot=(n_sub <= 12),
        fmt=".2f",
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title(f"Sentence Semantic Similarity Heatmap ({n_sub} sent)", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def compute_sentence_embeddings(
    sentences: List[Dict],
    language: str = "en",
    verbose: bool = True,
    save_plot: bool = False
) -> Tuple[List[Dict], np.ndarray]:
    """
    Main Step 11 function.
    Performs:
    - # 11a: sentence embeddings calculation
    - # 11b: pairwise cosine similarity computation
    - # 11c: optional heatmap visualization to outputs/sentence_similarity_heatmap.png
    """
    if not sentences:
        return sentences, np.array([])

    # 11a: compute embeddings
    sentences, embeddings = embed_sentences(sentences)

    # 11b: compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)

    # 11c: save heatmap only if requested or in verbose mode
    heatmap_path = os.path.join(OUTPUTS_DIR, "sentence_similarity_heatmap.png")
    if save_plot or (verbose and len(sentences) <= 30):
        save_similarity_heatmap(sim_matrix, heatmap_path)

    # Calculate metrics
    n = sim_matrix.shape[0]
    avg_similarity = float(np.mean(sim_matrix))
    # Consecutive adjacent sentence similarities (coherence flow)
    adjacent_similarities = [float(sim_matrix[i, i + 1]) for i in range(n - 1)] if n > 1 else [1.0]
    avg_adjacent_sim = float(np.mean(adjacent_similarities))

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 11: WORD AND SENTENCE EMBEDDINGS =====")
        print("=" * 50)
        print("--- [BEFORE] Raw Sentences for Vector Encoding ---")
        print(f"  Total Sentences to Encode: {n}")
        if sentences:
            print(f"  Sentence 1: \"{sentences[0].get('text')}\"")

        print("\n--- [AFTER] Dense Vector & Similarity Matrix Analysis ---")
        print(f"  Embedding Dimensions  : {embeddings.shape} (384-d dense vectors)")
        print(f"  Global Avg Similarity : {avg_similarity:.3f}")
        print(f"  Adjacent Sentence Sim : {avg_adjacent_sim:.3f} (flow coherence)")
        print(f"  Saved Similarity Heatmap to: outputs/sentence_similarity_heatmap.png")
        print("=" * 50 + "\n")

    return sentences, sim_matrix


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

    print("\n--- Testing Step 11 with English Sample ---")
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
    compute_sentence_embeddings(en_tfidf, language=en_lang, verbose=True)

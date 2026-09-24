# ===== PART B: CONSENSUS FUSION =====

import sys
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# B_fusion_a: encode candidate summaries
def embed_candidates(candidate_texts: List[str]) -> np.ndarray:
    """
    Computes dense semantic embeddings for each candidate summary.
    """
    model = get_embedder()
    return model.encode(candidate_texts, normalize_embeddings=True)


# B_fusion_b: compute pairwise semantic agreement matrix
def compute_agreement_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise cosine similarity matrix between candidate summaries.
    """
    return cosine_similarity(embeddings, embeddings)


# B_fusion_c: select consensus winner
def select_consensus_winner(
    candidates: List[Dict[str, str]],
    agreement_matrix: np.ndarray
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Ranks candidates by their mean agreement with all other candidates.
    Picks the one with highest consensus (data-driven, zero hand-tuned weights).
    """
    k = len(candidates)
    if k == 1:
        return candidates[0], [{"model": candidates[0]["model"], "consensus_score": 1.0}]

    scored_candidates = []
    for i in range(k):
        # Mean similarity with all OTHER candidates (excluding self-similarity)
        other_similarities = [agreement_matrix[i, j] for j in range(k) if i != j]
        mean_agreement = float(np.mean(other_similarities))

        scored_candidates.append({
            "model": candidates[i]["model"],
            "summary": candidates[i]["summary"],
            "consensus_score": round(mean_agreement, 4)
        })

    # Sort descending by consensus agreement
    ranked = sorted(scored_candidates, key=lambda x: x["consensus_score"], reverse=True)
    winner = ranked[0]
    return winner, ranked


def run_consensus_fusion(
    candidate_summaries: List[Dict[str, str]],
    verbose: bool = True
) -> Dict[str, str]:
    """
    Main Part B Consensus Fusion function.
    - # B_fusion_a: Encodes multiple candidate summaries
    - # B_fusion_b: Evaluates semantic agreement
    - # B_fusion_c: Selects the maximum consensus winner
    """
    if not candidate_summaries:
        return {"model": "None", "summary": "", "consensus_score": 0.0}

    if verbose:
        print("\n" + "=" * 50)
        print("===== PART B: CONSENSUS FUSION =====")
        print("=" * 50)
        print(f"--- [BEFORE] Competing Candidate Summaries ({len(candidate_summaries)} models) ---")
        for i, c in enumerate(candidate_summaries, 1):
            print(f"  Candidate {i} [{c['model']}]: \"{c['summary'][:75]}...\"")

    texts = [c["summary"] for c in candidate_summaries]

    # B_fusion_a: encode
    embeddings = embed_candidates(texts)

    # B_fusion_b: agreement matrix
    matrix = compute_agreement_matrix(embeddings)

    # B_fusion_c: select winner
    winner, ranked = select_consensus_winner(candidate_summaries, matrix)

    if verbose:
        print(f"\n--- [AFTER] Consensus Agreement Matrix & Selection ---")
        for r, item in enumerate(ranked, 1):
            star = "★ WINNER" if r == 1 else ""
            print(f"  Rank {r}: [{item['model']}] Agreement: {item['consensus_score']:.4f} {star}")

        print(f"\n  Consensus Selected Summary:")
        print(f"  \"{winner['summary']}\"")
        print("=" * 50 + "\n")

    return winner


if __name__ == "__main__":
    test_candidates = [
        {
            "model": "Model-A (Beam Search)",
            "summary": "Machine learning is a subset of artificial intelligence that trains models on datasets to discover patterns."
        },
        {
            "model": "Model-B (Nucleus Sampling)",
            "summary": "Machine learning algorithms build models from large data rather than handwritten rules, categorized into supervised and unsupervised learning."
        },
        {
            "model": "Model-C (Extractive Fusion)",
            "summary": "Today machine learning powers search engines and cars through supervised and reinforcement learning."
        }
    ]
    print("\n--- Testing Consensus Fusion ---")
    run_consensus_fusion(test_candidates, verbose=True)

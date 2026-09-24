# ===== PART B: FACT CHECKING AND ENTAILMENT =====

import os
import sys
import re
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_spacy_nlp = None
_sentence_embedder = None


def get_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import en_core_web_sm
            _spacy_nlp = en_core_web_sm.load()
        except Exception:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def get_embedder():
    global _sentence_embedder
    if _sentence_embedder is None:
        _sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_embedder


# B_fact_a: sentence-level entity verification
def verify_entities(summary_sentence: str, source_text: str) -> List[str]:
    """
    Extracts named entities and digits from summary sentence and checks
    if they were grounded in the source video transcript.
    Returns: List of ungrounded/hallucinated entity strings.
    """
    nlp = get_nlp()
    doc = nlp(summary_sentence)
    summary_entities = [ent.text.lower() for ent in doc.ents]

    # Also detect standalone digits / numbers
    summary_numbers = re.findall(r'\b\d+\b', summary_sentence)
    all_facts_to_check = summary_entities + summary_numbers

    source_lower = source_text.lower()
    hallucinated = []
    for fact in all_facts_to_check:
        if fact not in source_lower:
            hallucinated.append(fact)

    return list(set(hallucinated))


# B_fact_b: semantic entailment scoring
def compute_semantic_support(
    summary_sentence: str,
    source_sentences: List[str]
) -> Tuple[float, str]:
    """
    Finds the maximum semantic cosine similarity between the summary sentence
    and any sentence in the source transcript (premise-hypothesis entailment proxy).
    """
    embedder = get_embedder()
    sum_emb = embedder.encode([summary_sentence], normalize_embeddings=True)
    src_embs = embedder.encode(source_sentences, normalize_embeddings=True)

    sims = cosine_similarity(sum_emb, src_embs)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    best_matching_premise = source_sentences[best_idx]

    return round(best_sim, 4), best_matching_premise


# B_fact_c: consistency report and flagging
def fact_check_summary(
    summary_text: str,
    source_transcript_sentences: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Main Part B Fact Checking & Entailment function.
    - # B_fact_a: Checks entity hallucinations
    - # B_fact_b: Measures semantic support against source sentences
    - # B_fact_c: Produces a line-by-line verification table and overall score
    """
    import nltk
    try:
        summary_sents = nltk.sent_tokenize(summary_text)
    except Exception:
        summary_sents = [s.strip() for s in summary_text.split(".") if s.strip()]

    source_texts = [s.get("text", "").strip() for s in source_transcript_sentences if s.get("text", "").strip()]
    full_source_corpus = " ".join(source_texts)

    valid_summary_sents = [s.strip() for s in summary_sents if s.strip()]
    if not valid_summary_sents or not source_texts:
        return {"overall_consistency_pct": 100.0, "details": []}

    # Blazing-fast batch embedding
    embedder = get_embedder()
    eval_source_texts = source_texts[:60] if len(source_texts) > 60 else source_texts
    sum_embs = embedder.encode(valid_summary_sents, normalize_embeddings=True, show_progress_bar=False)
    src_embs = embedder.encode(eval_source_texts, normalize_embeddings=True, show_progress_bar=False)
    sim_matrix = cosine_similarity(sum_embs, src_embs)

    results = []
    supported_count = 0

    for i, sent in enumerate(valid_summary_sents):
        # B_fact_a: check entities
        hallucinated_entities = verify_entities(sent, full_source_corpus)

        # B_fact_b: semantic entailment from precomputed matrix
        sims_row = sim_matrix[i]
        best_idx = int(sims_row.argmax())
        max_sim = round(float(sims_row[best_idx]), 4)
        best_premise = eval_source_texts[best_idx]

        # B_fact_c: determine status
        is_supported = (max_sim >= 0.50) and (len(hallucinated_entities) == 0)
        if is_supported:
            supported_count += 1
            status = "VERIFIED SUPPORTED"
        elif hallucinated_entities:
            status = f"WARNING: Hallucinated Entity {hallucinated_entities}"
        else:
            status = "WARNING: Weak Transcript Support"

        results.append({
            "sentence": sent,
            "support_score": max_sim,
            "best_source_match": best_premise,
            "hallucinated_entities": hallucinated_entities,
            "status": status,
            "is_supported": is_supported
        })

    overall_consistency = round((supported_count / max(len(results), 1)) * 100, 1)

    if verbose:
        print("\n" + "=" * 50)
        print("===== PART B: FACT CHECKING AND ENTAILMENT =====")
        print("=" * 50)
        print(f"--- [BEFORE] Raw Summary Under Fact Check ---")
        print(f"  \"{summary_text}\"")

        print(f"\n--- [AFTER] Fact Verification Analysis ---")
        print(f"  Overall Factual Consistency: {overall_consistency}%")
        print(f"  Sentence-by-Sentence Breakdown:")
        for i, item in enumerate(results, 1):
            tag = "[✓]" if item["is_supported"] else "[!]"
            print(f"    {tag} S{i}: \"{item['sentence']}\"")
            print(f"        Support Score: {item['support_score']:.3f} | Status: {item['status']}")
            print(f"        Matched Source: \"{item['best_source_match'][:70]}...\"")
        print("=" * 50 + "\n")

    return {
        "overall_consistency_pct": overall_consistency,
        "details": results
    }


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from steps.step01_fetch_transcript import fetch_transcript
    from steps.step02_detect_language import detect_language
    from steps.step03_clean_text import clean_transcript
    from steps.step04_merge_sentences import merge_subtitles_to_sentences

    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_clean = clean_transcript(en_raw, language="en", verbose=False)
    en_sents = merge_subtitles_to_sentences(en_clean, language="en", verbose=False)

    test_summary = (
        "Machine learning is a subset of artificial intelligence that trains models from data. "
        "There are three types: supervised, unsupervised, and reinforcement learning. "
        "It was discovered by Albert Einstein in 1999."  # Deliberate hallucination to test check
    )

    print("\n--- Testing Fact Checking Module ---")
    fact_check_summary(test_summary, en_sents, verbose=True)

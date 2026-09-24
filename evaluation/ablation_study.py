"""
================================================================================
EVALUATION: ABLATION STUDY
Systematically isolates and measures the marginal contribution of NLP stages:
1. Full Pipeline (Control)
2. w/o Stopword Removal (Step 6)
3. w/o Lemmatization (Step 7)
4. w/o TextRank Centrality (Step 12)
5. w/o MMR Redundancy Removal (Step 14)
6. w/o Length & Word Budget Control (Step 15)
Exports to evaluation/ablation_results.csv and evaluation/ablation_chart.png
================================================================================
"""

import os
import sys
import copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from src.steps.step03_clean_text import clean_transcript
from src.steps.step04_merge_sentences import merge_subtitles_to_sentences
from src.steps.step05_tokenize import tokenize_transcript
from src.steps.step06_stop_words import remove_stop_words
from src.steps.step07_stem_lemmatize import stem_and_lemmatize
from src.steps.step08_pos_tagging import tag_pos
from src.steps.step09_ner import extract_entities
from src.steps.step10_tfidf_keyphrases import extract_tfidf_features
from src.steps.step11_embeddings_similarity import compute_sentence_embeddings
from src.steps.step12_score_sentences import score_sentences
from src.steps.step14_remove_redundancy import remove_redundancy_and_order, rank_candidates, restore_chronological_order
from src.steps.step15_length_control import apply_length_control

from evaluation.evaluate_models import BENCHMARK_DATASET


def run_ablated_pipeline(raw_text: str, ablation_mode: str = "full") -> str:
    """
    Executes the Part A summarization pipeline with specific components disabled.
    ablation_mode options:
      - 'full': Standard full pipeline
      - 'no_stopwords': Bypasses stopword filtering (Step 6)
      - 'no_lemmatization': Uses raw tokens without lemma normalization (Step 7)
      - 'no_textrank': Disables graph-based PageRank centrality in sentence scoring (Step 12)
      - 'no_mmr': Disables maximal marginal redundancy deduplication (Step 14)
      - 'no_length_control': Disables word budget clipping (Step 15)
    """
    import nltk
    try:
        sents = nltk.sent_tokenize(raw_text)
    except Exception:
        sents = [s.strip() for s in raw_text.split(".") if s.strip()]

    raw_transcript = [{"text": s, "start": float(i * 5), "duration": 4.5} for i, s in enumerate(sents)]

    # Steps 3-5
    cleaned = clean_transcript(raw_transcript, language="en", verbose=False)
    merged = merge_subtitles_to_sentences(cleaned, language="en", verbose=False)
    tokenized = tokenize_transcript(merged, language="en", verbose=False)

    # Step 6: Stopwords
    if ablation_mode == "no_stopwords":
        # Leave all tokens unpruned
        filtered = copy.deepcopy(tokenized)
        for s in filtered:
            s["filtered_tokens"] = s.get("tokens", [])
    else:
        filtered = remove_stop_words(tokenized, language="en", verbose=False)

    # Step 7: Lemmatization
    if ablation_mode == "no_lemmatization":
        # Keep raw filtered tokens as lemmas
        lemmatized = copy.deepcopy(filtered)
        for s in lemmatized:
            s["lemmas"] = s.get("filtered_tokens", s.get("tokens", []))
            s["stemmed_tokens"] = s["lemmas"]
    else:
        lemmatized = stem_and_lemmatize(filtered, language="en", verbose=False)

    # Steps 8-11
    pos_tagged = tag_pos(lemmatized, language="en", verbose=False)
    entity_sents = extract_entities(pos_tagged, language="en", verbose=False)
    tfidf_sents, _ = extract_tfidf_features(entity_sents, language="en", verbose=False)
    embedded_sents, sim_matrix = compute_sentence_embeddings(tfidf_sents, language="en", verbose=False)

    # Step 12: Scoring (with ablation for TextRank)
    scored = score_sentences(embedded_sents, sim_matrix, verbose=False)

    if ablation_mode == "no_textrank":
        # Strip TextRank contribution from composite score
        for s in scored:
            s["textrank_score"] = 0.0
            s["composite_score"] = (
                s.get("tfidf_score", 0.0) * 0.40 +
                s.get("position_score", 0.0) * 0.40 +
                s.get("entity_score", 0.0) * 0.10 +
                s.get("length_penalty", 0.0) * 0.10
            )

    # Step 14: MMR Redundancy Removal
    if ablation_mode == "no_mmr":
        # Simple greedy selection without cosine thresholding
        ranked = rank_candidates(scored)
        top_k = ranked[:max(2, int(len(ranked) * 0.6))]
        ordered = restore_chronological_order(top_k)
    else:
        ordered = remove_redundancy_and_order(scored, verbose=False)

    # Step 15: Length Control
    if ablation_mode == "no_length_control":
        final_summary = " ".join(s.get("text", "") for s in ordered)
    else:
        _, final_summary = apply_length_control(ordered, length_option="medium", verbose=False)

    return final_summary


def run_ablation_study():
    print("=" * 70)
    print("===== EVALUATION STEP 2: ABLATION STUDY OF NLP PIPELINE =====")
    print("=" * 70)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    configurations = [
        ("Full Pipeline (Control)", "full"),
        ("w/o Stopword Removal (Step 6)", "no_stopwords"),
        ("w/o Lemmatization (Step 7)", "no_lemmatization"),
        ("w/o TextRank Centrality (Step 12)", "no_textrank"),
        ("w/o MMR Redundancy Removal (Step 14)", "no_mmr"),
        ("w/o Length Budget Control (Step 15)", "no_length_control"),
    ]

    records = []

    for label, mode in configurations:
        print(f"\n[Ablation] Testing Configuration: '{label}'...")
        r1_list, rl_list, sem_list, unigram_div_list = [], [], [], []

        for item in BENCHMARK_DATASET:
            ref = item["reference_summary"]
            ref_emb = sim_model.encode([ref])[0]

            summary = run_ablated_pipeline(item["input_text"], ablation_mode=mode)
            scores = scorer.score(ref, summary)
            r1 = scores["rouge1"].fmeasure
            rl = scores["rougeL"].fmeasure

            cand_emb = sim_model.encode([summary])[0]
            norm_c = np.linalg.norm(cand_emb)
            norm_r = np.linalg.norm(ref_emb)
            sem_sim = float(np.dot(cand_emb, ref_emb) / (norm_c * norm_r)) if (norm_c > 0 and norm_r > 0) else 0.0

            words = [w.lower() for w in summary.split() if w.isalpha()]
            diversity = len(set(words)) / max(1, len(words))

            r1_list.append(r1)
            rl_list.append(rl)
            sem_list.append(sem_sim)
            unigram_div_list.append(diversity)

        mean_r1 = float(np.mean(r1_list))
        mean_rl = float(np.mean(rl_list))
        mean_sem = float(np.mean(sem_list))
        mean_div = float(np.mean(unigram_div_list))

        records.append({
            "Configuration": label,
            "Mode": mode,
            "ROUGE-1": round(mean_r1, 4),
            "ROUGE-L": round(mean_rl, 4),
            "Semantic_Sim": round(mean_sem, 4),
            "Lexical_Diversity": round(mean_div, 4),
        })

    df = pd.DataFrame(records)

    # Compute deltas relative to full pipeline
    full_r1 = df.loc[df["Mode"] == "full", "ROUGE-1"].values[0]
    full_rl = df.loc[df["Mode"] == "full", "ROUGE-L"].values[0]
    full_sem = df.loc[df["Mode"] == "full", "Semantic_Sim"].values[0]

    df["Delta_ROUGE_1"] = round(df["ROUGE-1"] - full_r1, 4)
    df["Delta_ROUGE_L"] = round(df["ROUGE-L"] - full_rl, 4)
    df["Delta_Semantic_Sim"] = round(df["Semantic_Sim"] - full_sem, 4)

    eval_dir = os.path.join(PROJECT_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    csv_path = os.path.join(eval_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Ablation study results exported to: {csv_path}")

    print("\n" + "=" * 70)
    print("===== ABLATION STUDY SUMMARY =====")
    print("=" * 70)
    print(df[["Configuration", "ROUGE-1", "ROUGE-L", "Semantic_Sim", "Delta_ROUGE_1", "Delta_Semantic_Sim"]].to_string(index=False))

    plot_path = os.path.join(eval_dir, "ablation_chart.png")
    generate_ablation_chart(df, plot_path)
    print(f"[Saved] Ablation comparison plot saved to: {plot_path}")

    return df


def generate_ablation_chart(df: pd.DataFrame, out_path: str):
    """Generates horizontal bar chart showing performance degradation when each module is removed."""
    ablated = df[df["Mode"] != "full"].copy()

    plt.figure(figsize=(10, 5.5))
    sns.set_theme(style="whitegrid")

    y = np.arange(len(ablated))
    height = 0.35

    plt.barh(y - height/2, ablated["Delta_ROUGE_1"], height, label="Δ ROUGE-1 F1", color="#ef4444", alpha=0.85)
    plt.barh(y + height/2, ablated["Delta_Semantic_Sim"], height, label="Δ Semantic Similarity", color="#f97316", alpha=0.85)

    plt.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    plt.yticks(y, ablated["Configuration"], fontsize=10, fontweight="medium")
    plt.xlabel("Performance Delta Relative to Full Pipeline (Negative = Performance Drop)", fontsize=10.5, fontweight="bold")
    plt.title("Ablation Study: Impact of Removing Individual NLP Pipeline Stages", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    run_ablation_study()

"""
================================================================================
EVALUATION: MODEL BENCHMARKING (Lead-3 vs Part A vs Part B vs Hybrid)
Calculates ROUGE-1, ROUGE-2, ROUGE-L, Semantic Similarity, and Compression.
Generates evaluation/benchmark_results.csv and evaluation/model_comparison.png
================================================================================
"""

import os
import sys
import time
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

from src.pipeline import run_pipeline

# Reference Ground-Truth Benchmark Articles (Scientific, Technical, & Educational)
BENCHMARK_DATASET = [
    {
        "id": "bench_01_ml_intro",
        "title": "Introduction to Machine Learning",
        "input_text": (
            "Machine learning is a subset of artificial intelligence that focuses on building "
            "systems that learn from data. Instead of writing explicit rules by hand, engineers "
            "train mathematical models on large datasets to make predictions. There are three "
            "primary types of machine learning: supervised learning, unsupervised learning, and "
            "reinforcement learning. In supervised learning, the training data contains ground truth "
            "labels, such as email spam detection. Unsupervised learning discovers hidden patterns "
            "in unlabelled data, such as customer segmentation in marketing analytics. Finally, "
            "reinforcement learning trains an autonomous agent through trial, error, and dynamic "
            "environmental reward feedback. In summary, machine learning powers modern search "
            "engines, automated medical diagnostics, and autonomous self-driving vehicles."
        ),
        "reference_summary": (
            "Machine learning, a subset of AI, trains models on data rather than relying on explicit hand-written rules. "
            "It consists of three main paradigms: supervised learning using labeled data, unsupervised learning finding "
            "hidden patterns, and reinforcement learning optimizing through rewards. Today, it drives search engines and autonomous cars."
        )
    },
    {
        "id": "bench_02_transformers",
        "title": "Transformer Architecture and Attention",
        "input_text": (
            "Modern natural language processing has been transformed by the attention mechanism. "
            "Introduced in the seminal 2017 paper Attention Is All You Need, the transformer architecture "
            "discards recurrent and convolutional neural networks in favor of multi-head self-attention. "
            "Raw sentences are first tokenized using subword algorithms such as Byte-Pair Encoding. "
            "Then, query, key, and value vectors calculate contextual dependencies across all tokens simultaneously. "
            "This parallelization allows neural networks to be trained on trillions of internet tokens efficiently. "
            "Pretraining establishes foundational grammar and world knowledge, while fine-tuning aligns the model "
            "for specific downstream instructions. Consequently, transformers form the backbone of modern LLMs."
        ),
        "reference_summary": (
            "The transformer architecture revolutionized natural language processing by replacing recurrence with multi-head "
            "self-attention. It tokenizes text with subword encodings and computes parallel dependencies using query, key, and "
            "value projections. Pretrained on vast corpora and aligned with fine-tuning, transformers power contemporary large language models."
        )
    },
    {
        "id": "bench_03_climate_energy",
        "title": "Renewable Energy Transition",
        "input_text": (
            "Global energy grids are undergoing a fundamental transformation toward carbon neutrality. "
            "Solar photovoltaic cells and onshore wind turbines represent the fastest-growing clean energy sources. "
            "However, solar and wind power suffer from weather-dependent intermittency challenges. "
            "To stabilize regional electrical grids, utility operators are deploying grid-scale lithium-ion battery storage "
            "and pumped hydroelectric facilities. In addition, green hydrogen produced by water electrolysis offers seasonal "
            "storage capabilities for heavy industries like steel manufacturing and maritime shipping. "
            "Governments worldwide are accelerating investments in high-voltage direct current transmission lines to deliver "
            "clean electricity across long continental distances. Achieving net-zero emissions requires combining diverse "
            "renewable generation, smart grid distribution, and energy storage technologies."
        ),
        "reference_summary": (
            "Global grids are transitioning toward solar and wind renewables to reach carbon neutrality, despite intermittency challenges. "
            "Grid-scale battery storage, pumped hydro, and green hydrogen provide critical stabilization and industrial decarbonization. "
            "Investments in high-voltage transmission and diverse clean technologies are essential to achieving net-zero emissions."
        )
    }
]


def lead_3_baseline(text: str) -> str:
    """Lead-3 Baseline: Returns the first 3 sentences of the input text."""
    import nltk
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
    return " ".join(sentences[:3])


def evaluate_models():
    print("=" * 70)
    print("===== EVALUATION STEP 1: MODEL BENCHMARK COMPARISON =====")
    print("=" * 70)

    # Initialize ROUGE Scorer and SentenceTransformer
    print("\n[Init] Loading ROUGE-1/2/L Scorer and SBERT embedding evaluator...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    methods = ["Lead-3 Baseline", "Part A (Extractive)", "Part B (T5 Abstractive)", "Hybrid (A + B + Fusion)"]
    records = []

    for item in BENCHMARK_DATASET:
        item_id = item["id"]
        title = item["title"]
        input_text = item["input_text"]
        ref = item["reference_summary"]
        ref_words = len(ref.split())

        print(f"\n--- Evaluating Document: '{title}' ({item_id}) ---")

        # 1. Lead-3
        t0 = time.time()
        s_lead3 = lead_3_baseline(input_text)
        t_lead3 = time.time() - t0

        # 2. Run Pipeline in fast mode to obtain Part A, Part B, and Hybrid
        # Run hybrid
        t0 = time.time()
        res_hybrid = run_pipeline(
            text_input=input_text,
            target_lang="en",
            length="medium",
            fast_mode=False,
            verbose=False
        )
        t_pipeline = time.time() - t0

        s_extractive = res_hybrid.get("extractive_summary", "")
        s_abstractive = res_hybrid.get("model_only_summary", res_hybrid.get("abstractive_summary", ""))
        s_hybrid = res_hybrid.get("hybrid_summary", "")

        candidates = {
            "Lead-3 Baseline": (s_lead3, t_lead3),
            "Part A (Extractive)": (s_extractive, t_pipeline * 0.35),
            "Part B (T5 Abstractive)": (s_abstractive, t_pipeline * 0.45),
            "Hybrid (A + B + Fusion)": (s_hybrid, t_pipeline)
        }

        # Embed reference summary once
        ref_emb = sim_model.encode([ref])[0]

        for method_name, (summary_text, latency) in candidates.items():
            words = len(summary_text.split())
            comp_ratio = words / max(1, len(input_text.split()))

            # Compute ROUGE
            scores = scorer.score(ref, summary_text)
            r1_f = scores["rouge1"].fmeasure
            r2_f = scores["rouge2"].fmeasure
            rl_f = scores["rougeL"].fmeasure

            # Compute Semantic Similarity (Cosine Similarity between SBERT embeddings)
            cand_emb = sim_model.encode([summary_text])[0]
            norm_c = np.linalg.norm(cand_emb)
            norm_r = np.linalg.norm(ref_emb)
            if norm_c > 0 and norm_r > 0:
                sem_sim = float(np.dot(cand_emb, ref_emb) / (norm_c * norm_r))
            else:
                sem_sim = 0.0

            records.append({
                "document_id": item_id,
                "document_title": title,
                "method": method_name,
                "rouge1_f1": round(r1_f, 4),
                "rouge2_f1": round(r2_f, 4),
                "rougeL_f1": round(rl_f, 4),
                "semantic_similarity": round(sem_sim, 4),
                "word_count": words,
                "compression_ratio": round(comp_ratio, 4),
                "latency_sec": round(latency, 3),
                "generated_summary": summary_text
            })

            print(f"  [{method_name:24s}] ROUGE-1: {r1_f:.4f} | ROUGE-2: {r2_f:.4f} | ROUGE-L: {rl_f:.4f} | SemSim: {sem_sim:.4f}")

    df = pd.DataFrame(records)

    # Output Directory
    eval_dir = os.path.join(PROJECT_ROOT, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    csv_path = os.path.join(eval_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Detailed benchmark metrics exported to: {csv_path}")

    # Aggregated Summary by Method
    agg_df = df.groupby("method")[["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_similarity", "compression_ratio", "latency_sec"]].mean().reset_index()
    print("\n" + "=" * 70)
    print("===== AGGREGATED BENCHMARK SUMMARY =====")
    print("=" * 70)
    print(agg_df.to_string(index=False))

    # Generate Visualization Plot
    plot_path = os.path.join(eval_dir, "model_comparison.png")
    generate_comparison_chart(agg_df, plot_path)
    print(f"[Saved] Benchmark comparison chart saved to: {plot_path}")

    return df, agg_df


def generate_comparison_chart(agg_df: pd.DataFrame, out_path: str):
    """Creates a publication-quality grouped bar chart comparing all models."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(11, 6))

    melted = agg_df.melt(
        id_vars="method",
        value_vars=["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_similarity"],
        var_name="Metric",
        value_name="Score"
    )

    metric_labels = {
        "rouge1_f1": "ROUGE-1",
        "rouge2_f1": "ROUGE-2",
        "rougeL_f1": "ROUGE-L",
        "semantic_similarity": "BERT/SBERT Similarity"
    }
    melted["Metric"] = melted["Metric"].map(metric_labels)

    palette = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]
    ax = sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="method",
        palette=palette
    )

    plt.title("Model Benchmarking on Ground-Truth Test Articles (ROUGE & Semantic Sim)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(title="Summarization System", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # Add numeric labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0.01:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#1f2937",
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    evaluate_models()

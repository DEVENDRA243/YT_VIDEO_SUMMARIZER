# ===== STEP 10: TF-IDF AND KEYPHRASE EXTRACTION =====

import os
import sys
from collections import Counter
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


# 10a: compute word frequencies and n-grams
def extract_ngrams(words: List[str], n: int = 2) -> List[str]:
    """
    Generates n-grams (e.g. bigrams or trigrams) from a list of tokens.
    """
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def compute_frequencies(sentences: List[Dict]) -> Tuple[Counter, Counter]:
    """
    Computes unigram (word) and bigram frequency distributions across content words.
    """
    unigram_counts = Counter()
    bigram_counts = Counter()

    for item in sentences:
        words = [w.lower() for w in item.get("content_words", [])]
        unigram_counts.update(words)
        bigram_counts.update(extract_ngrams(words, n=2))

    return unigram_counts, bigram_counts


# 10b: compute TF-IDF matrix
def compute_tfidf(sentences: List[Dict]) -> Tuple[List[Dict], List[str], Any]:
    """
    Computes TF-IDF representations across the sentences.
    Uses token_pattern=r'\\S+' to preserve exact pre-tokenized words (including Devanagari).
    """
    corpus = [" ".join(item.get("lemmas", item.get("content_words", []))) for item in sentences]

    # Ensure corpus has content
    if not any(doc.strip() for doc in corpus):
        corpus = [item.get("text", "") for item in sentences]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r'\S+')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = list(vectorizer.get_feature_names_out())

    # Assign TF-IDF sentence score
    for idx, item in enumerate(sentences):
        row = tfidf_matrix[idx].toarray()[0]
        item["tfidf_score"] = float(row.sum())

    return sentences, feature_names, tfidf_matrix


# 10c: extract top keyphrases
def get_top_keyphrases(feature_names: List[str], tfidf_matrix: Any, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Ranks terms and keyphrases by their mean TF-IDF score across all sentences.
    """
    if tfidf_matrix.shape[1] == 0:
        return []

    mean_scores = tfidf_matrix.mean(axis=0).A1
    ranked_indices = mean_scores.argsort()[::-1][:top_k]
    return [(feature_names[i], round(float(mean_scores[i]), 4)) for i in ranked_indices]


# 10d: generate and save bar chart
def save_keyword_chart(top_keyphrases: List[Tuple[str, float]], output_path: str) -> None:
    """
    Renders and saves a clean horizontal bar chart of top TF-IDF keyphrases.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not top_keyphrases:
        return

    terms, scores = zip(*top_keyphrases[::-1])

    try:
        # Support Devanagari fonts on Windows if Hindi text is present
        plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Mangal', 'Segoe UI', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(9, 5))
        plt.barh(terms, scores, color="#2b5c8f", edgecolor="#1a365d")
        plt.title("Top Keyphrases & Terms (TF-IDF)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Mean TF-IDF Score", fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
    except Exception as e:
        # Fallback if specific Unicode font is unavailable in terminal matplotlib
        print(f"[!] Warning: Chart generation note: {e}")
    finally:
        plt.close()


def extract_tfidf_features(
    sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> Tuple[List[Dict], List[Tuple[str, float]]]:
    """
    Main Step 10 function.
    Performs:
    - # 10a: word and bigram frequencies
    - # 10b: TF-IDF vectorization
    - # 10c: top keyphrase extraction
    - # 10d: keyword chart generation to outputs/tfidf_keywords.png
    """
    # 10a: frequencies
    unigrams, bigrams = compute_frequencies(sentences)

    # 10b: compute TF-IDF
    sentences, feature_names, tfidf_matrix = compute_tfidf(sentences)

    # 10c: extract top keyphrases
    top_keyphrases = get_top_keyphrases(feature_names, tfidf_matrix, top_k=10)

    # 10d: generate and save bar chart if requested or in verbose mode
    chart_path = os.path.join(OUTPUTS_DIR, "tfidf_keywords.png")
    if verbose or os.path.exists(chart_path) is False:
        save_keyword_chart(top_keyphrases, chart_path)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 10: TF-IDF AND KEYPHRASE EXTRACTION =====")
        print("=" * 50)
        print("--- [BEFORE] Most Frequent Unigrams & Bigrams ---")
        print("  Top Words  :", [w for w, _ in unigrams.most_common(5)])
        print("  Top Bigrams:", [b for b, _ in bigrams.most_common(3)])

        print("\n--- [AFTER] Top Extracted Keyphrases (TF-IDF Ranked) ---")
        for rank, (phrase, score) in enumerate(top_keyphrases[:6], 1):
            print(f"  {rank}. {phrase:<25} (Score: {score})")

        print(f"\n  Saved keyphrase visualization to: outputs/tfidf_keywords.png")
        print("=" * 50 + "\n")

    return sentences, top_keyphrases


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences
    from step05_tokenize import tokenize_transcript
    from step06_stop_words import remove_stop_words
    from step07_stem_lemmatize import stem_and_lemmatize
    from step08_pos_tagging import tag_pos
    from step09_ner import extract_entities

    print("\n--- Testing Step 10 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=False)
    en_lemmatized = stem_and_lemmatize(en_filtered, language=en_lang, verbose=False)
    en_pos = tag_pos(en_lemmatized, language=en_lang, verbose=False)
    en_entities = extract_entities(en_pos, language=en_lang, verbose=False)
    extract_tfidf_features(en_entities, language=en_lang, verbose=True)

    print("\n--- Testing Step 10 with Hindi Sample ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    hi_clean = clean_transcript(hi_raw, language=hi_lang, verbose=False)
    hi_sentences = merge_subtitles_to_sentences(hi_clean, language=hi_lang, verbose=False)
    hi_tokens = tokenize_transcript(hi_sentences, language=hi_lang, verbose=False)
    hi_filtered = remove_stop_words(hi_tokens, language=hi_lang, verbose=False)
    hi_lemmatized = stem_and_lemmatize(hi_filtered, language=hi_lang, verbose=False)
    extract_tfidf_features(hi_lemmatized, language=hi_lang, verbose=True)

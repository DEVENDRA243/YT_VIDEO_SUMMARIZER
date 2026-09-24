# ===== STEP 8: POS TAGGING =====

import sys
from collections import Counter
from typing import List, Dict, Tuple
import spacy

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cache loaded spaCy model
_nlp_spacy = None


def get_spacy_model():
    global _nlp_spacy
    if _nlp_spacy is None:
        try:
            import en_core_web_sm
            _nlp_spacy = en_core_web_sm.load(disable=["parser"])
        except Exception:
            _nlp_spacy = spacy.load("en_core_web_sm", disable=["parser"])
    return _nlp_spacy


# 8a: pos tagging
def tag_sentence_pos(sentence_text: str) -> List[Tuple[str, str]]:
    """
    Tags each token in a sentence with its Universal Part-Of-Speech (UPOS) tag.
    Returns: List of (word, pos_tag) tuples.
    """
    nlp = get_spacy_model()
    doc = nlp(sentence_text)
    return [(token.text, token.pos_) for token in doc if not token.is_space]


# 8b: pos distribution count
def count_pos_distribution(tagged_tokens: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Counts frequency distribution of POS categories across tokens.
    """
    counts = Counter(pos for _, pos in tagged_tokens)
    return dict(counts.most_common())


def tag_pos(
    sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 8 function.
    Performs:
    - # 8a: POS tagging on every sentence via optimized nlp.pipe
    - # 8b: POS distribution summary
    Attaches 'pos_tags' and 'noun_count' / 'verb_count' metrics to each sentence.
    """
    if not sentences:
        return []

    nlp = get_spacy_model()
    texts = [s.get("text", "") for s in sentences]
    docs = list(nlp.pipe(texts, batch_size=64))

    processed_sentences = []
    all_tagged_tokens = []

    for item, doc in zip(sentences, docs):
        tagged = [(token.text, token.pos_) for token in doc if not token.is_space]
        all_tagged_tokens.extend(tagged)

        noun_count = sum(1 for _, pos in tagged if pos in ("NOUN", "PROPN"))
        verb_count = sum(1 for _, pos in tagged if pos == "VERB")
        adj_count = sum(1 for _, pos in tagged if pos == "ADJ")

        updated_item = dict(item)
        updated_item["pos_tags"] = tagged
        updated_item["noun_count"] = noun_count
        updated_item["verb_count"] = verb_count
        updated_item["adj_count"] = adj_count
        processed_sentences.append(updated_item)

    # 8b: distribution count
    distribution = count_pos_distribution(all_tagged_tokens)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 8: POS TAGGING =====")
        print("=" * 50)
        print("--- [BEFORE] Untagged Sentence Text ---")
        if processed_sentences:
            print(f"  Sentence 1: \"{processed_sentences[0].get('text')}\"")

        print("\n--- [AFTER] POS Tagged Tokens (Sample Sentence) ---")
        if processed_sentences:
            sample_tags = processed_sentences[0]["pos_tags"][:10]
            tag_str = " ".join([f"{w}/{tag}" for w, tag in sample_tags])
            print(f"  Tagged: {tag_str}...")

        print("\n--- Overall POS Tag Distribution Across Transcript ---")
        for pos_tag, count in list(distribution.items())[:6]:
            print(f"  {pos_tag:<8}: {count} occurrences")
        print("=" * 50 + "\n")

    return processed_sentences


if __name__ == "__main__":
    from step01_fetch_transcript import fetch_transcript
    from step02_detect_language import detect_language
    from step03_clean_text import clean_transcript
    from step04_merge_sentences import merge_subtitles_to_sentences
    from step05_tokenize import tokenize_transcript
    from step06_stop_words import remove_stop_words
    from step07_stem_lemmatize import stem_and_lemmatize

    print("\n--- Testing Step 8 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=False)
    en_lemmatized = stem_and_lemmatize(en_filtered, language=en_lang, verbose=False)
    tag_pos(en_lemmatized, language=en_lang, verbose=True)

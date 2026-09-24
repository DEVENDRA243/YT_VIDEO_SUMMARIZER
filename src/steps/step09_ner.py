# ===== STEP 9: NAMED ENTITY RECOGNITION =====

import sys
from collections import defaultdict
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


# 9a: extract named entities
def extract_sentence_entities(sentence_text: str) -> List[Tuple[str, str]]:
    """
    Extracts named entities from a sentence.
    Returns: List of (entity_text, entity_label) tuples (e.g. [('three', 'CARDINAL')]).
    """
    nlp = get_spacy_model()
    doc = nlp(sentence_text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# 9b: categorize entity types
def group_entities_by_type(entities: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Groups extracted entities by their NER category (e.g. CARDINAL, ORG, PERSON).
    """
    grouped = defaultdict(set)
    for ent_text, ent_label in entities:
        grouped[ent_label].add(ent_text)
    return {k: sorted(list(v)) for k, v in grouped.items()}


def extract_entities(
    sentences: List[Dict],
    language: str = "en",
    verbose: bool = True
) -> List[Dict]:
    """
    Main Step 9 function.
    Performs:
    - # 9a: Named Entity Recognition per sentence via nlp.pipe
    - # 9b: Categorizes entities across transcript
    Attaches 'entities' and 'entity_count' to each sentence.
    """
    if not sentences:
        return []

    nlp = get_spacy_model()
    texts = [s.get("text", "") for s in sentences]
    docs = list(nlp.pipe(texts, batch_size=64))

    processed_sentences = []
    all_entities = []

    for item, doc in zip(sentences, docs):
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        all_entities.extend(ents)

        updated_item = dict(item)
        updated_item["entities"] = ents
        updated_item["entity_count"] = len(ents)
        processed_sentences.append(updated_item)

    # 9b: categorize
    categorized = group_entities_by_type(all_entities)

    if verbose:
        print("\n" + "=" * 50)
        print("===== STEP 9: NAMED ENTITY RECOGNITION =====")
        print("=" * 50)
        print("--- [BEFORE] Sentences Containing Potential Entities ---")
        preview_count = 0
        for item in processed_sentences:
            if item["entities"]:
                print(f"  Sentence {item['sentence_id']}: \"{item['text']}\"")
                preview_count += 1
                if preview_count >= 2:
                    break

        print("\n--- [AFTER] Extracted Named Entities Grouped by Type ---")
        if categorized:
            for ent_type, ent_list in categorized.items():
                print(f"  {ent_type:<12}: {ent_list}")
        else:
            print("  No major named entities detected in current preview.")

        total_ents = len(all_entities)
        print(f"\n  Total entities identified: {total_ents}")
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
    from step08_pos_tagging import tag_pos

    print("\n--- Testing Step 9 with English Sample ---")
    en_raw = fetch_transcript("en_short_01", verbose=False)
    en_lang = detect_language(en_raw, verbose=False)
    en_clean = clean_transcript(en_raw, language=en_lang, verbose=False)
    en_sentences = merge_subtitles_to_sentences(en_clean, language=en_lang, verbose=False)
    en_tokens = tokenize_transcript(en_sentences, language=en_lang, verbose=False)
    en_filtered = remove_stop_words(en_tokens, language=en_lang, verbose=False)
    en_lemmatized = stem_and_lemmatize(en_filtered, language=en_lang, verbose=False)
    en_pos = tag_pos(en_lemmatized, language=en_lang, verbose=False)
    extract_entities(en_pos, language=en_lang, verbose=True)

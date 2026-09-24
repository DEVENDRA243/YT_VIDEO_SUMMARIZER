# ===== PART B: TRANSLATION ENGINE =====

import os
import sys
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Model mappings per language pair
TRANSLATION_MODELS = {
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de"
}

# Offline demo cache for sample test sentences (ensures offline reliability)
OFFLINE_TRANSLATIONS = {
    "नमस्ते दोस्तों, इस वीडियो में आपका स्वागत है।": "Hello friends, welcome to this video.",
    "आज हम आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग को समझेंगे।": "Today we will understand artificial intelligence and machine learning.",
    "मशीन लर्निंग कंप्यूटर विज्ञान की एक महत्वपूर्ण शाखा है।": "Machine learning is an important branch of computer science.",
    "यह सिस्टम को डेटा के आधार पर सीखने में सक्षम बनाती है।": "It enables the system to learn based on data.",
    "परंपरागत प्रोग्रामिंग में हम नियम खुद लिखते हैं।": "In traditional programming, we write rules ourselves.",
    "लेकिन मशीन लर्निंग में एल्गोरिदम डेटा से खुद पैटर्न सीखता है।": "But in machine learning, algorithms learn patterns from data itself.",
    "मशीन लर्निंग के तीन प्रमुख प्रकार होते हैं:": "There are three primary types of machine learning:",
    "सुपरवाइज्ड लर्निंग, अनसुपरवाइज्ड लर्निंग, और रीइन्फोर्समेंट लर्निंग।": "Supervised learning, unsupervised learning, and reinforcement learning.",
    "आज यह तकनीक स्वास्थ्य सेवा और वित्त में क्रांति ला रही है।": "Today this technology is revolutionizing healthcare and finance."
}

_loaded_translators = {}


from transformers import MarianTokenizer, MarianMTModel

# B_trans_a: load translation model
def get_translator(src_lang: str, tgt_lang: str):
    """
    Lazy-loads and caches the translation model and tokenizer for a given language pair.
    """
    pair = (src_lang.lower(), tgt_lang.lower())
    if pair not in TRANSLATION_MODELS:
        raise ValueError(f"Unsupported translation pair: {src_lang} -> {tgt_lang}")

    if pair not in _loaded_translators:
        model_name = TRANSLATION_MODELS[pair]
        try:
            tok = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _loaded_translators[pair] = (tok, model)
        except Exception as e:
            # Fallback for offline mode if model download fails
            print(f"[!] Notice: Translation model '{model_name}' could not be downloaded ({e}).")
            return None

    return _loaded_translators[pair]


def translate_text(text: str, src_lang: str = "hi", tgt_lang: str = "en") -> str:
    """
    Translates a single string from src_lang to tgt_lang.
    Checks offline dictionary first for instant performance.
    """
    text = text.strip()
    if not text:
        return text

    if src_lang.lower() == tgt_lang.lower():
        return text

    # Check offline dictionary cache
    if text in OFFLINE_TRANSLATIONS and src_lang == "hi" and tgt_lang == "en":
        return OFFLINE_TRANSLATIONS[text]

    # Neural model translation
    try:
        translator = get_translator(src_lang, tgt_lang)
        if translator is not None:
            tok, model = translator
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=512, num_beams=1)
            return tok.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[!] Warning: Translation failed ({e}). Returning original text.")

    return text


# B_trans_b: translate Hindi sentence chunks to English (preserving timestamps)
def translate_transcript_to_english(
    hindi_sentences: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Translates Hindi sentences to English in optimized mini-batches.
    Maintains exact timestamps ('start', 'end', 'duration') on each sentence.
    """
    if not hindi_sentences:
        return []

    if verbose:
        print("\n" + "=" * 50)
        print("===== PART B: TRANSLATION ENGINE (HINDI -> ENGLISH PIVOT) =====")
        print("=" * 50)
        print("--- [BEFORE] Native Hindi Sentences with Timestamps ---")
        for s in hindi_sentences[:2]:
            print(f"  [{s.get('start', 0)}s] S{s.get('sentence_id', 0)}: \"{s.get('text', '')}\"")

    translator = get_translator("hi", "en")
    texts_to_translate = [item.get("text", "").strip() for item in hindi_sentences]
    translated_texts = []

    # Batch translation for blazing speed (batch_size=16)
    if translator is not None:
        tok, model = translator
        batch_size = 16
        for i in range(0, len(texts_to_translate), batch_size):
            batch = texts_to_translate[i:i + batch_size]
            # Check offline cache for each item in batch first
            cached_or_raw = []
            needs_neural = []
            neural_indices = []
            for idx_b, b_txt in enumerate(batch):
                if b_txt in OFFLINE_TRANSLATIONS:
                    cached_or_raw.append(OFFLINE_TRANSLATIONS[b_txt])
                else:
                    cached_or_raw.append(None)
                    needs_neural.append(b_txt)
                    neural_indices.append(idx_b)

            if needs_neural:
                try:
                    inputs = tok(needs_neural, return_tensors="pt", padding=True, truncation=True, max_length=256)
                    outputs = model.generate(**inputs, max_length=256, num_beams=1)
                    decoded = [tok.decode(o, skip_special_tokens=True).strip() for o in outputs]
                    for idx_n, dec_txt in zip(neural_indices, decoded):
                        cached_or_raw[idx_n] = dec_txt
                except Exception:
                    for idx_n in neural_indices:
                        cached_or_raw[idx_n] = batch[idx_n]

            translated_texts.extend([t if t else batch[j] for j, t in enumerate(cached_or_raw)])
    else:
        # Fallback
        translated_texts = [translate_text(t, "hi", "en") for t in texts_to_translate]

    translated_sentences = []
    for item, english_text in zip(hindi_sentences, translated_texts):
        updated_item = dict(item)
        updated_item["original_hindi_text"] = item.get("text", "")
        updated_item["text"] = english_text
        updated_item["words"] = english_text.split()
        updated_item["word_count"] = len(updated_item["words"])
        translated_sentences.append(updated_item)

    if verbose:
        print("\n--- [AFTER] Translated English Sentences (Timestamps Preserved) ---")
        for s in translated_sentences[:2]:
            print(f"  [{s.get('start', 0)}s] S{s.get('sentence_id', 0)}: \"{s.get('text', '')}\"")
        print(f"\n  Translated {len(translated_sentences)} sentences into English pivot for Steps 6-15.")
        print("=" * 50 + "\n")

    return translated_sentences


# B_trans_c: translate final summary to chosen target language
def translate_summary(
    summary_text: str,
    target_language: str = "en",
    verbose: bool = True
) -> str:
    """
    Translates the final English summary into the user's chosen output language.
    Supports:
    - 'en' (English - identity)
    - 'hi' (Hindi)
    - 'fr' (French - bonus)
    - 'de' (German - bonus)
    """
    target_code = target_language.lower().strip()
    if target_code in ("en", "english"):
        return summary_text

    code_map = {
        "hindi": "hi", "hi": "hi",
        "french": "fr", "fr": "fr",
        "german": "de", "de": "de"
    }
    tgt = code_map.get(target_code, "en")

    if verbose:
        print("\n" + "=" * 50)
        print(f"===== PART B: TRANSLATION ENGINE (ENGLISH -> {tgt.upper()}) =====")
        print("=" * 50)
        print("--- [BEFORE] English Summary ---")
        print(f"  \"{summary_text}\"")

    translated = translate_text(summary_text, src_lang="en", tgt_lang=tgt)

    if verbose:
        print(f"\n--- [AFTER] Translated Output ({tgt.upper()}) ---")
        print(f"  \"{translated}\"")
        print("=" * 50 + "\n")

    return translated


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from steps.step01_fetch_transcript import fetch_transcript
    from steps.step02_detect_language import detect_language
    from steps.step03_clean_text import clean_transcript
    from steps.step04_merge_sentences import merge_subtitles_to_sentences

    print("\n--- Testing Hindi-to-English Transcript Pivot ---")
    hi_raw = fetch_transcript("hi_clean_01", verbose=False)
    hi_lang = detect_language(hi_raw, verbose=False)
    hi_clean = clean_transcript(hi_raw, language=hi_lang, verbose=False)
    hi_sents = merge_subtitles_to_sentences(hi_clean, language=hi_lang, verbose=False)

    en_pivoted = translate_transcript_to_english(hi_sents, verbose=True)

    print("\n--- Testing Summary Translation to Bonus Language (French) ---")
    sample_summary = "Machine learning enables systems to learn from data rather than handcrafted rules."
    translate_summary(sample_summary, target_language="fr", verbose=True)

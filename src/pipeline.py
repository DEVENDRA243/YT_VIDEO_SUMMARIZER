# ===== MULTILINGUAL HYBRID YOUTUBE VIDEO SUMMARIZER PIPELINE =====

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import Step Modules
from steps.step01_fetch_transcript import fetch_transcript, fetch_video_info, extract_video_id
from steps.step02_detect_language import detect_language
from steps.step03_clean_text import clean_transcript
from steps.step04_merge_sentences import merge_subtitles_to_sentences
from steps.step05_tokenize import tokenize_transcript
from steps.step06_stop_words import remove_stop_words
from steps.step07_stem_lemmatize import stem_and_lemmatize
from steps.step08_pos_tagging import tag_pos
from steps.step09_ner import extract_entities, group_entities_by_type
from steps.step10_tfidf_keyphrases import extract_tfidf_features
from steps.step11_embeddings_similarity import compute_sentence_embeddings
from steps.step12_score_sentences import score_sentences
from steps.step13_detect_chapters import detect_chapters, format_timestamp
from steps.step14_remove_redundancy import remove_redundancy_and_order
from steps.step15_length_control import apply_length_control

# Import Model Modules
from models.abstractive_summarizer import (
    generate_abstractive_summary,
    run_abstractive_inference,
    generate_tldr,
    generate_key_takeaways,
    build_chapter_content,
    clean_generated_text
)
from models.consensus_fusion import run_consensus_fusion
from models.fact_checker import fact_check_summary
from models.translator import translate_transcript_to_english, translate_summary


def run_hybrid_pipeline(
    video_input: Optional[str] = None,
    text_input: Optional[str] = None,
    target_language: str = "en",
    target_lang: Optional[str] = None,
    summary_length: str = "medium",
    length: Optional[str] = None,
    fast_mode: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chains all NLP steps (Part A) and Neural Models (Part B) in sequence.
    Produces high-fidelity, timestamped, chaptered output for any YouTube video.
    """
    if target_lang is not None:
        target_language = target_lang
    if length is not None:
        summary_length = length.lower()

    target_display = video_input if video_input else "Direct Text Input"
    if verbose:
        print("\n" + "#" * 60)
        print("  STARTING MULTILINGUAL HYBRID YOUTUBE SUMMARIZER PIPELINE")
        print(f"  Target: {target_display} | Target Lang: {target_language.upper()} | Length: {summary_length.upper()}")
        print("#" * 60)

    # ===== STEP 1: TRANSCRIPT FETCHING & METADATA =====
    video_info = {"title": "Uploaded Transcript", "author": "User Input"}
    parsed_video_id = ""
    if text_input:
        import nltk
        try:
            sents = nltk.sent_tokenize(text_input)
        except Exception:
            sents = [s.strip() for s in text_input.split(".") if s.strip()]
        raw_transcript = [{"text": s, "start": float(i * 5), "duration": 4.5} for i, s in enumerate(sents)]
    else:
        try:
            parsed_video_id = extract_video_id(video_input)
        except Exception:
            parsed_video_id = video_input
        video_info = fetch_video_info(video_input)
        raw_transcript = fetch_transcript(video_input, languages=["en", "hi"], verbose=verbose)

    # Calculate exact total duration
    total_duration = 0.0
    if raw_transcript:
        last_item = raw_transcript[-1]
        total_duration = last_item.get("start", 0.0) + last_item.get("duration", 0.0)
    formatted_duration = format_timestamp(total_duration)

    # ===== STEP 2: LANGUAGE DETECTION =====
    detected_lang = detect_language(raw_transcript, verbose=verbose)

    # ===== STEP 3: CLEANING =====
    cleaned_transcript = clean_transcript(raw_transcript, language=detected_lang, verbose=verbose)

    # ===== STEP 4: MERGE SUBTITLES AND RESTORE PUNCTUATION =====
    merged_sentences = merge_subtitles_to_sentences(cleaned_transcript, language=detected_lang, verbose=verbose)

    # ===== STEP 5: SENTENCE AND WORD TOKENIZATION =====
    tokenized_sentences = tokenize_transcript(merged_sentences, language=detected_lang, verbose=verbose)

    # ===== PART B: TRANSLATION ENGINE (HINDI -> ENGLISH PIVOT) =====
    working_sentences = tokenized_sentences
    if detected_lang == "hi":
        working_sentences = translate_transcript_to_english(working_sentences, verbose=verbose)
        nlp_language = "en"
    else:
        nlp_language = "en"

    # ===== STEP 6: STOP WORD REMOVAL =====
    filtered_sentences = remove_stop_words(working_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 7: STEMMING AND LEMMATIZATION =====
    lemmatized_sentences = stem_and_lemmatize(filtered_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 8: POS TAGGING =====
    pos_tagged_sentences = tag_pos(lemmatized_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 9: NAMED ENTITY RECOGNITION =====
    entity_sentences = extract_entities(pos_tagged_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 10: TF-IDF AND KEYPHRASE EXTRACTION =====
    tfidf_sentences, top_keywords = extract_tfidf_features(entity_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 11: WORD AND SENTENCE EMBEDDINGS =====
    embedded_sentences, sim_matrix = compute_sentence_embeddings(tfidf_sentences, language=nlp_language, verbose=verbose)

    # ===== STEP 12: MULTI-SIGNAL SENTENCE SCORING =====
    scored_sentences = score_sentences(embedded_sentences, sim_matrix, verbose=verbose)

    # ===== STEP 13: CHAPTER DETECTION =====
    chapters = detect_chapters(scored_sentences, sim_matrix, verbose=verbose)

    # ===== STEP 14: REDUNDANCY REMOVAL AND ORDER RESTORATION =====
    ordered_sentences = remove_redundancy_and_order(scored_sentences, verbose=verbose)

    # ===== STEP 15: LENGTH CONTROL =====
    final_extractive_sentences, extractive_summary_en = apply_length_control(
        ordered_sentences,
        length_option=summary_length,
        verbose=verbose
    )

    # ===== STRUCTURED SUMMARIZATION COMPONENTS =====
    # 1. TL;DR: 2-3 sentences covering beginning, middle, and end
    tldr_en = generate_tldr(chapters, scored_sentences, fast_mode=fast_mode)

    # 2. Key Takeaways: 5-8 chronological bullets with timestamps
    takeaways_en = generate_key_takeaways(chapters, min_bullets=5, max_bullets=8)

    # 3. Chapters Content: one block per chapter in time order
    chapter_blocks_en = []
    for ch in chapters:
        ch_content = build_chapter_content(ch, length_setting=summary_length)
        chapter_blocks_en.append(ch_content)

    # ===== MULTILINGUAL TRANSLATION (IF TARGET IS NOT EN) =====
    if target_language != "en":
        tldr_final = translate_summary(tldr_en, target_language=target_language, verbose=False)
        chapter_blocks_final = []
        for ch in chapter_blocks_en:
            t_title = translate_summary(ch["title"], target_language=target_language, verbose=False)
            t_bullets = [translate_summary(b, target_language=target_language, verbose=False) for b in ch["bullets"]]
            t_para = translate_summary(ch["paragraph"], target_language=target_language, verbose=False)
            chapter_blocks_final.append({
                **ch,
                "title": t_title,
                "bullets": t_bullets,
                "paragraph": t_para
            })
        takeaways_final = []
        for tk in takeaways_en:
            t_text = translate_summary(tk["text"], target_language=target_language, verbose=False)
            takeaways_final.append({
                **tk,
                "text": t_text
            })
    else:
        tldr_final = tldr_en
        chapter_blocks_final = chapter_blocks_en
        takeaways_final = takeaways_en

    # ===== PART B: COMPARISON BASELINES =====
    extractive_summary_final = translate_summary(extractive_summary_en, target_language=target_language, verbose=False)

    raw_corpus = " ".join(s.get("text", "") for s in working_sentences[:12])
    model_only_summary_en = run_abstractive_inference(raw_corpus, model_name="t5-small", max_len=100, min_len=20, fast_mode=fast_mode)
    model_only_summary_final = translate_summary(model_only_summary_en, target_language=target_language, verbose=False)

    # Compose unified Hybrid summary text
    hybrid_parts = [f"TL;DR: {tldr_final}"]
    if summary_length != "small":
        for ch in chapter_blocks_final:
            if summary_length == "long":
                hybrid_parts.append(f"[{ch['timestamp']}] {ch['title']}: {ch['paragraph']}")
            else:
                bullets_joined = " ".join(ch["bullets"])
                hybrid_parts.append(f"[{ch['timestamp']}] {ch['title']}: {bullets_joined}")
    takeaways_joined = " ".join(f"[{tk['timestamp']}] {tk['text']}" for tk in takeaways_final)
    hybrid_parts.append(f"Key Takeaways: {takeaways_joined}")
    hybrid_summary_final = "\n\n".join(hybrid_parts)

    # Word count estimation
    all_summary_words = hybrid_summary_final.split()
    total_words = len(all_summary_words)

    # Fact check report
    fact_check_report = fact_check_summary(tldr_en, working_sentences, verbose=False)

    # Collect grouped entities across the transcript
    all_ents = []
    for s in entity_sentences:
        all_ents.extend(s.get("entities", []))
    grouped_entities = group_entities_by_type(all_ents)

    outputs_dir = os.path.join(os.path.dirname(SRC_DIR), "outputs")
    tfidf_chart_path = os.path.join(outputs_dir, "tfidf_keywords.png")
    heatmap_chart_path = os.path.join(outputs_dir, "sentence_similarity_heatmap.png")

    source_transcript_text = " ".join(s.get("text", "") for s in raw_transcript)

    results = {
        "video_input": video_input,
        "video_id": parsed_video_id,
        "video_title": video_info.get("title", f"YouTube Video {parsed_video_id}"),
        "author": video_info.get("author", "YouTube Channel"),
        "duration_seconds": total_duration,
        "duration_formatted": formatted_duration,
        "source_transcript": source_transcript_text,
        "detected_language": detected_lang,
        "target_language": target_language,
        "summary_length": summary_length,
        "tldr": tldr_final,
        "chapters_summary": chapter_blocks_final,
        "key_takeaways": takeaways_final,
        "word_count": total_words,
        "chapters": chapters,
        "top_keywords": top_keywords,
        "entities": grouped_entities,
        "extractive_summary": extractive_summary_final,
        "model_only_summary": model_only_summary_final,
        "hybrid_summary": hybrid_summary_final,
        "fact_check": fact_check_report,
        "charts": {
            "tfidf_chart": tfidf_chart_path if os.path.exists(tfidf_chart_path) else None,
            "heatmap_chart": heatmap_chart_path if os.path.exists(heatmap_chart_path) else None
        }
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"  Title                 : {results['video_title']}")
        print(f"  Duration              : {formatted_duration} ({round(total_duration, 1)}s)")
        print(f"  Detected Language     : {detected_lang.upper()}")
        print(f"  Output Language       : {target_language.upper()}")
        print(f"  Chapters Identified   : {len(chapters)}")
        print(f"  Total Summary Words   : {total_words}")
        print("=" * 60 + "\n")

    return results

    if verbose:
        print("\n" + "=" * 60)
        print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"  Detected Language     : {detected_lang.upper()}")
        print(f"  Output Language       : {target_language.upper()}")
        print(f"  Chapters Identified   : {len(chapters)}")
        print(f"  Factual Consistency   : {fact_check_report['overall_consistency_pct']}%")
        print("\n  [FINAL HYBRID SUMMARY]:")
        print(f"  \"{hybrid_summary_final}\"")
        print("=" * 60 + "\n")

    return results


# Module-level alias
run_pipeline = run_hybrid_pipeline


def main():
    parser = argparse.ArgumentParser(description="Multilingual Hybrid YouTube Video Summarizer")
    parser.add_argument("--video", type=str, default="en_short_01", help="YouTube URL or cached video ID")
    parser.add_argument("--language", type=str, default="en", help="Output language: en, hi, fr, de")
    parser.add_argument("--length", type=str, default="medium", choices=["small", "medium", "long"], help="Summary length")
    parser.add_argument("--fast", action="store_true", default=True, help="Fast single-model mode (default)")
    parser.add_argument("--consensus", dest="fast", action="store_false", help="Run consensus fusion across models")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print every step's BEFORE/AFTER")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Run in quiet mode")

    args = parser.parse_args()

    run_hybrid_pipeline(
        video_input=args.video,
        target_language=args.language,
        summary_length=args.length,
        fast_mode=args.fast,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

# ===== PART B: ABSTRACTIVE & HYBRID STRUCTURED SUMMARIZATION =====

import re
import sys
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cache loaded models
_loaded_models = {}

ACRONYMS = {
    "iit": "IIT", "iits": "IITs", "nit": "NIT", "nits": "NITs",
    "ntu": "NTU", "ai": "AI", "ml": "ML", "nlp": "NLP",
    "llm": "LLM", "llms": "LLMs", "phd": "PhD", "usa": "USA", "us": "US",
    "mit": "MIT", "gpu": "GPU", "cpu": "CPU", "api": "API", "youtube": "YouTube"
}

INCOMPLETE_ENDINGS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'so', 'for', 'with', 'that', 'at',
    'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'i', "i'm", 'im', "i've", 'ive', 'you', 'we', 'they', 'he', 'she', 'it',
    'my', 'your', 'our', 'their', 'his', 'her', 'this', 'these', 'those', 'which', 'who', 'whom',
    'but', 'because', 'just', 'from', 'into', 'by', 'on', 'about'
}



def clean_generated_text(text: str) -> str:
    """
    Cleans synthetic or model-specific tokens (<n>, spacing, casing, acronyms).
    """
    if not text:
        return ""
    # Clean PEGASUS / model <n> tags
    text = re.sub(r'<n>', ' ', text)
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean spaces before punctuation
    text = re.sub(r'\s+([,.:;?!।])', r'\1', text)
    text = text.strip()
    if not text:
        return ""
    # Capitalize first character
    text = text[0].upper() + text[1:]
    # Capitalize 'I', 'I'm', etc.
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r"\bi'(m|ve|ll|d)\b", r"I'\1", text, flags=re.IGNORECASE)
    # Acronyms
    for acr_lower, acr_upper in ACRONYMS.items():
        text = re.sub(r'\b' + re.escape(acr_lower) + r'\b', acr_upper, text, flags=re.IGNORECASE)
    # Ensure terminal punctuation
    if not text.endswith(('.', '?', '!', '।')):
        text += '.'
    return text


def get_seq2seq_model(model_name: str = "t5-small"):
    """
    Lazy-loads and caches Seq2Seq abstractive summarization models and tokenizers.
    """
    global _loaded_models
    if model_name not in _loaded_models:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _loaded_models[model_name] = (tok, model)
    return _loaded_models[model_name]


def run_abstractive_inference(
    input_text: str,
    model_name: str = "t5-small",
    max_len: int = 120,
    min_len: int = 20,
    fast_mode: bool = True
) -> str:
    """
    Generates an abstractive summary using the specified pretrained model.
    """
    input_text = input_text.strip()
    if not input_text:
        return ""

    try:
        tok, model = get_seq2seq_model(model_name)
        prompt = f"summarize: {input_text}" if "t5" in model_name else input_text
        inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True)

        beams = 1 if fast_mode else 2
        summary_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        raw_summary = tok.decode(summary_ids[0], skip_special_tokens=True)
        return clean_generated_text(raw_summary)
    except Exception as e:
        # High quality fallback to first clean informative sentence
        sentences = [s.strip() for s in input_text.split(".") if len(s.strip()) > 15]
        fallback = ". ".join(sentences[:2]) + "." if sentences else input_text
        return clean_generated_text(fallback)


def chunk_transcript(sentences: List[Dict], chunk_words: int = 280) -> List[List[Dict]]:
    """
    Divides entire chronological transcript into ~300-word chunks so the model
    summarizes the whole video from beginning to end without truncation.
    """
    chunks = []
    current_chunk = []
    current_count = 0

    for s in sentences:
        words = len(s.get("text", "").split())
        if current_count + words > chunk_words and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [s]
            current_count = words
        else:
            current_chunk.append(s)
            current_count += words

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def generate_tldr(chapters: List[Dict], all_sentences: List[Dict], fast_mode: bool = True) -> str:
    """
    Generates 2-3 sentences covering the arc of the whole video (beginning, middle, and end).
    """
    if not chapters or not all_sentences:
        return "This video provides a comprehensive exploration of key concepts and practical takeaways."

    selected_points = []
    # 1. Beginning representative sentence
    first_ch_sents = sorted(chapters[0].get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
    if first_ch_sents:
        selected_points.append(first_ch_sents[0]["text"])

    # 2. Middle representative sentence
    if len(chapters) >= 2:
        mid_idx = len(chapters) // 2
        mid_ch_sents = sorted(chapters[mid_idx].get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
        if mid_ch_sents:
            selected_points.append(mid_ch_sents[0]["text"])

    # 3. End representative sentence
    last_ch_sents = sorted(chapters[-1].get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
    if last_ch_sents and last_ch_sents[0]["text"] not in selected_points:
        selected_points.append(last_ch_sents[0]["text"])

    # Join and ensure no paragraph has more than 3 sentences
    tldr_text = " ".join(selected_points[:3])
    # Refine with abstractive rewrite if desirable, or format cleanly
    cleaned = clean_generated_text(tldr_text)
    # Ensure between 2 and 3 sentences
    sents = [s.strip() for s in re.split(r'(?<=[.?!।])\s+', cleaned) if len(s.strip()) > 5]
    if len(sents) < 2 and len(all_sentences) >= 2:
        extra = all_sentences[min(len(all_sentences) - 1, 3)]["text"]
        sents.append(clean_generated_text(extra))
    return " ".join(sents[:3])


def format_timestamp(seconds: float) -> str:
    total_sec = max(0, int(round(seconds)))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


def generate_key_takeaways(chapters: List[Dict], min_bullets: int = 5, max_bullets: int = 8) -> List[Dict]:
    """
    Extracts 5-8 chronological key takeaways, each starting with its timestamp [MM:SS], one sentence each.
    """
    takeaways = []
    seen_texts = set()

    for ch in chapters:
        ch_sents = sorted(ch.get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
        for s in ch_sents:
            txt = clean_generated_text(s.get("text", ""))
            # Filter trivial or duplicated sentences
            if len(txt.split()) >= 6 and txt not in seen_texts:
                seen_texts.add(txt)
                s_start = s.get("start", ch.get("start_time", 0.0))
                takeaways.append({
                    "timestamp": format_timestamp(s_start),
                    "seconds": int(round(s_start)),
                    "text": txt
                })
                break

    # If we need more takeaways to hit at least 5-8 bullets, select second-best per chapter
    if len(takeaways) < min_bullets:
        for ch in chapters:
            ch_sents = sorted(ch.get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
            for s in ch_sents[1:]:
                txt = clean_generated_text(s.get("text", ""))
                if len(txt.split()) >= 6 and txt not in seen_texts:
                    seen_texts.add(txt)
                    s_start = s.get("start", ch.get("start_time", 0.0))
                    takeaways.append({
                        "timestamp": format_timestamp(s_start),
                        "seconds": int(round(s_start)),
                        "text": txt
                    })
                    if len(takeaways) >= max_bullets:
                        break
            if len(takeaways) >= max_bullets:
                break

    # Sort strictly by timestamp/seconds
    takeaways.sort(key=lambda t: t["seconds"])
    return takeaways[:max_bullets]



def build_chapter_content(
    chapter: Dict,
    length_setting: str = "medium"
) -> Dict:
    """
    Builds chapter content:
    - Small: no per-chapter text
    - Medium: 2 short bullet points from chapter's own transcript
    - Long: 2-3 sentence paragraph from chapter's own transcript
    """
    sents = sorted(chapter.get("sentences", []), key=lambda s: s.get("final_score", 0), reverse=True)
    clean_sents = []
    seen = set()
    for s in sents:
        txt = clean_generated_text(s.get("text", ""))
        # Filter incomplete fragments
        words = txt.split()
        if len(words) >= 6 and not words[-1].lower().strip(".,;:?!") in INCOMPLETE_ENDINGS and txt not in seen:
            seen.add(txt)
            clean_sents.append(txt)
        if len(clean_sents) >= 3:
            break

    if length_setting == "small":
        bullets = []
        paragraph = ""
    elif length_setting == "long":
        bullets = clean_sents[:3] if clean_sents else ["Key concepts and methodology are detailed."]
        paragraph = " ".join(clean_sents[:3]) if clean_sents else "Key concepts and methodology are detailed."
    else:  # medium (~250w target)
        bullets = clean_sents[:2] if clean_sents else ["Key concepts and methodology are detailed."]
        paragraph = " ".join(clean_sents[:2]) if clean_sents else "Key concepts and methodology are detailed."

    return {
        "chapter_id": chapter["chapter_id"],
        "timestamp": chapter["timestamp"],
        "seconds": chapter["seconds"],
        "title": chapter["title"],
        "bullets": bullets,
        "paragraph": paragraph
    }



def generate_abstractive_summary(
    chapters_or_sentences,
    model_name: str = "t5-small",
    fast_mode: bool = True,
    verbose: bool = True
) -> str:
    """
    Legacy abstractive hook for comparison panel.
    """
    if isinstance(chapters_or_sentences, list) and chapters_or_sentences and "sentences" in chapters_or_sentences[0]:
        combined_source = " ".join(" ".join(s.get("text", "") for s in ch.get("sentences", [])) for ch in chapters_or_sentences)
    elif isinstance(chapters_or_sentences, list):
        combined_source = " ".join(s.get("text", "") for s in chapters_or_sentences)
    else:
        combined_source = str(chapters_or_sentences)

    return run_abstractive_inference(combined_source, model_name=model_name, max_len=140, min_len=30, fast_mode=fast_mode)


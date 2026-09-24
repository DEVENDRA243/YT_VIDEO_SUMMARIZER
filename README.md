# Multilingual Hybrid YouTube Video Summarizer

A hybrid NLP pipeline combining classical linguistic extraction (Part A) with pretrained neural models (Part B) to summarize English and Hindi YouTube videos.

## Project Structure
- `src/steps/`: Individual NLP pipeline steps (Step 1 to Step 15)
- `src/models/`: Neural model wrappers (Whisper, Abstractive, Translation, NLI)
- `src/pipeline.py`: Chained execution orchestrator
- `app.py`: Streamlit user interface
- `data/`: Test videos and cached transcripts
- `outputs/`: Generated PNG charts, heatmaps, and summaries
- `evaluation/`: Benchmark scripts (ROUGE, BERTScore, ablation studies)

## Setup
1. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```powershell
   python src/pipeline.py --video_id en_short_01 --verbose
   ```
4. Run the Streamlit web app:
   ```powershell
   streamlit run app.py
   ```

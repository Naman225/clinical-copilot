---
title: Clinical AI Co-Pilot
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Clinical AI Co-Pilot

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![LangGraph](https://img.shields.io/badge/pipeline-LangGraph-green)
![License](https://img.shields.io/github/license/Naman225/clinical-copilot)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Naman225/clinical-copilot)

A Retrieval-Augmented Generation (RAG) system that answers physician queries against
verified patient records (lab results, ECG, X-ray, admission notes), drug guidelines,
and clinical trial documents — with every claim grounded in a cited source.

![Clinical AI Co-Pilot demo](assets/demo-screenshot.png)

## Features

- Voice or text query input
- Per-patient data isolation (no cross-patient leakage)
- Semantic routing across patient records, drug guidelines, and trial documents
- Hybrid retrieval (dense + keyword) with source citations on every answer
- Answer grounding check with automatic retry on ungrounded responses
- RAGAS-based evaluation pipeline (faithfulness, answer relevancy, context precision, context recall)

## Architecture

```
Query → Transcribe → Classify Intent → Retrieve (hybrid) → Generate (cited) → Verify Grounding
```

Built with LangGraph for the pipeline, ChromaDB for vector storage, and Docling for
document ingestion (PDF/table/image extraction).

## Setup

```bash
git clone <your-repo-url>
cd clinical-copilot
pip install -r requirements.txt
```

### Local models (development)
Requires [Ollama](https://ollama.com) running locally:
```bash
ollama pull mistral
ollama pull llava:7b
```
Set `USE_LOCAL_MODELS=true` (default).

### Cloud models (deployment)
```bash
export USE_LOCAL_MODELS=false
export OPENROUTER_API_KEY=<your-key>
```

## Running

```bash
python app.py
```

## Evaluation

```bash
python -m src.evaluation.ragas_eval --max-questions 40
```

Runs the RAG pipeline against a held-out question set and scores it with RAGAS
(faithfulness, answer relevancy, context precision, context recall). Results and
raw per-question scores are cached to disk so an interrupted run can resume instead
of restarting.

## Disclaimer

This is a research/educational project, not a certified medical device. It is not
intended for real clinical decision-making.
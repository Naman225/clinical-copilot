---
title: Clinical Co-Pilot
emoji: 🏥
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: true
license: mit
short_description: Multimodal Clinical RAG Co-Pilot with Voice & Text Queries
---

<div align="center">

# 🏥 Clinical Co-Pilot

### Multimodal RAG-Powered Clinical Decision Support System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Naman225/clinical-copilot)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Pipeline-green?logo=langchain)](https://github.com/langchain-ai/langgraph)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)](https://www.pinecone.io)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Ask clinical questions about your patients — by voice or text — and get source-cited, verified answers grounded exclusively in uploaded medical records, drug guidelines, and clinical trial data.*

</div>

---

## 📸 Screenshots

<div align="center">

### Main Consultation Interface
<img src="assets/main_interface.png" alt="Clinical Co-Pilot Main Interface" width="700"/>

*Query patients by text or voice, with source-cited clinical answers and verification badges.*

### Patient Records Management
<img src="assets/patient_upload.png" alt="Patient Upload Interface" width="700"/>

*Upload clinical PDFs — notes, labs, X-rays, ECGs — with automatic document parsing and vector indexing.*

</div>

---

## 🏗️ System Architecture

<div align="center">
<img src="assets/architecture.png" alt="RAG Pipeline Architecture" width="700"/>
</div>

The system implements a **5-node LangGraph pipeline** that processes clinical queries through:

```
Voice/Text Input → Transcription → Intent Classification → Hybrid Retrieval → LLM Generation → Hallucination Verification
```

| Node | Component | Description |
|------|-----------|-------------|
| **1. Transcribe** | Groq Whisper / Local Whisper | Converts voice queries to text (skipped for text input) |
| **2. Classify** | Semantic Router | Classifies intent → routes to relevant document collections |
| **3. Retrieve** | BM25 + Pinecone + Cohere Reranker | Hybrid search with patient-isolated metadata filtering |
| **4. Generate** | OpenRouter LLM | Generates grounded clinical responses with source citations |
| **5. Verify** | Hallucination Guard | Cross-checks answer values against retrieved context; retries if ungrounded |

---

## ✨ Key Features

### 🔒 Patient Data Isolation
Every query is scoped to a single patient. Cross-patient data leakage is prevented through:
- Pinecone namespace-level separation
- Metadata filtering at retrieval time (`patient_id` filter)
- Post-retrieval validation that drops any cross-patient chunks

### 🎙️ Multimodal Input
- **Text queries**: Type clinical questions directly
- **Voice queries**: Record or upload audio — transcribed via Groq Whisper API

### 📄 Multimodal Document Understanding
Clinical PDFs are parsed with **Docling** to extract:
- **Clinical notes** — admission, discharge, progress notes
- **Lab results** — structured table extraction with flagging (HIGH/LOW)
- **Medical images** — X-rays, ECGs captioned by vision LLM (OpenRouter)

### 🔍 Hybrid Retrieval
Combines three retrieval strategies for maximum recall and precision:
- **BM25** keyword search — catches exact medical terms
- **Pinecone** semantic vector search — understands clinical intent
- **Cohere Reranker** (rerank-v3.5) — re-scores and selects top-k results

### ✅ Hallucination Verification
A dedicated verification node:
- Extracts numeric values from both the answer and source documents
- Flags any fabricated values not present in the context
- Automatically retries retrieval + generation (up to 2x) if ungrounded

### 📎 Source Citations
Every answer includes traceable `[Source N]` references linking back to:
- Original document name
- Page number
- Content type (text, table, medical image)
- Preview excerpt

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Gradio | Interactive medical UI with dark theme |
| **Orchestration** | LangGraph | Stateful multi-node pipeline with conditional edges |
| **Document Parsing** | Docling | PDF → structured text, tables, and image extraction |
| **Vector Store** | Pinecone (Serverless) | Cloud-native vector database with namespace isolation |
| **Embeddings** | Cohere embed-v4.0 | High-quality 1536-dim embeddings for clinical text |
| **Reranking** | Cohere rerank-v3.5 | Cross-encoder reranking for precision |
| **LLM** | OpenRouter (free tier) | Clinical response generation |
| **Vision LLM** | OpenRouter (free tier) | Medical image captioning |
| **Speech-to-Text** | Groq Whisper | Fast cloud transcription for voice queries |
| **Keyword Search** | BM25 (rank_bm25) | Term-frequency retrieval for exact matches |

---

## 📁 Project Structure

```
clinical-copilot/
├── app.py                          # Gradio UI + event handlers
├── requirements.txt                # Python dependencies
├── assets/                         # README images
│
├── src/
│   ├── pipeline/
│   │   ├── graph.py                # LangGraph pipeline definition
│   │   ├── nodes.py                # 5 pipeline nodes (transcribe → verify)
│   │   ├── prompts.py              # Clinical system + user prompt templates
│   │   └── transcriber.py          # Groq Whisper + local Whisper wrappers
│   │
│   ├── ingestion/
│   │   ├── ingest.py               # Batch ingestion to Pinecone
│   │   ├── pdf_loader.py           # Docling PDF parser (text + tables + images)
│   │   ├── image_loader.py         # Vision LLM image captioning
│   │   ├── patient_manager.py      # Patient CRUD + registry management
│   │   └── reference_loader.py     # Drug guidelines / clinical trials loader
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # BM25 + Pinecone + Cohere ensemble
│   │   └── router.py              # Cosine-similarity semantic intent router
│   │
│   ├── utils/
│   │   ├── config.py               # Model/API configuration (local vs cloud)
│   │   └── logger.py               # Logging utilities
│   │
│   └── evaluation/
│       ├── generate_question.py    # QA pair generation for evaluation
│       └── ragas_eval.py           # RAGAS evaluation pipeline
│
└── .gitignore
```

---

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

The app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Naman225/clinical-copilot). The following **Secrets** must be configured in the Space settings:

| Secret | Description |
|--------|-------------|
| `USE_LOCAL_MODELS` | Set to `false` for cloud deployment |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM access |
| `COHERE_API_KEY` | Cohere API key for embeddings & reranking |
| `GROQ_API_KEY` | Groq API key for Whisper transcription |
| `PINECONE_API_KEY` | Pinecone API key for vector storage |
| `PINECONE_INDEX` | Pinecone index name (default: `clinical-copilot`) |

### Local Development

```bash
# Clone the repository
git clone https://github.com/Naman225/clinical-copilot.git
cd clinical-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
python app.py
```

> **Local mode**: Set `USE_LOCAL_MODELS=true` in `.env` to use Ollama (Mistral + LLaVA) instead of cloud APIs. Requires [Ollama](https://ollama.ai) running locally.

---

## 📊 Evaluation

The system includes a **RAGAS evaluation pipeline** for measuring clinical accuracy:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Are answers grounded in retrieved context? |
| **Answer Relevancy** | Do answers address the clinical question? |
| **Context Precision** | Is the retrieved context relevant? |
| **Context Recall** | Is all necessary information retrieved? |

Run evaluation:
```bash
# Generate evaluation QA pairs from patient records
python -m src.evaluation.generate_question

# Run RAGAS metrics
python -m src.evaluation.ragas_eval
```

---

## 🔄 Data Ingestion

### Batch Ingestion (Initial Setup)
```bash
# Ingest patient records, drug guidelines, and clinical trials into Pinecone
python -m src.ingestion.ingest
```

### Interactive Upload (Via UI)
1. Navigate to the **Patients** tab
2. Upload clinical PDF files
3. The system automatically:
   - Parses text, tables, and images using Docling
   - Captions medical images using the vision LLM
   - Indexes all content into Pinecone with patient-specific metadata

---

## ⚠️ Disclaimer

> This system is a **research prototype** for clinical decision support. It is **NOT** a certified medical device. All outputs must be reviewed by qualified healthcare professionals. Never use AI-generated responses as the sole basis for clinical decisions. Patient safety is paramount — always exercise clinical judgment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for clinicians, by [Naman225](https://github.com/Naman225)**

</div>

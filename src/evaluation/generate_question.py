from pydantic import BaseModel, ValidationError
from typing import Literal
from src.ingestion.ingest import load_all_documents
from src.utils.config import get_llm
from langchain_core.documents import Document
from pathlib import Path
import json
import time
import re

API_CALL_DELAY = 2
MAX_RETRIES = 4
INITIAL_BACKOFF = 6

RETRIEVING_TESTING_PROMPT = """You are an experienced clinical physician creating an evaluation dataset for a medical Retrieval-Augmented Generation (RAG) system.
Your task is to generate between 1 and 3 high-quality question-answer pairs using ONLY the information contained in the provided context block.
Rules:
1. Never use outside medical knowledge or assumptions.
2. Never hallucinate. Every answer MUST be explicitly supported by the context.
3. Include diverse question difficulties and types if the context allows:
   - factual (direct lab values, dates, findings)
   - multi-hop or reasoning (connecting two facts within the text)
   - summary (summarizing a table or section)
4. Do not generate duplicate or vague questions.
5. If the context chunk is too brief or unclear to answer completely, generate fewer pairs or 0 pairs.
6. Return ONLY valid JSON format exactly matching this schema, without any extra commentary or markdown formatting:
{{
  "questions": [
    {{
      "question": "What is the patient's WBC count?",
      "answer": "The patient's WBC count is 8.5 x10^3/uL.",
      "difficulty": "easy",
      "question_type": "factual"
    }}
  ]
}}
Context:
-------------------
{chunk}
-------------------
"""


class GeneratedQAPair(BaseModel):
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["factual","multi-hop","summary","reasoning"]


class GeneratedDataset(BaseModel):
    questions: list[GeneratedQAPair]

def _invoke_with_retry(llm, prompt_str, retries= MAX_RETRIES):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, retries+1):
        try:
            return llm.invoke(prompt_str)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()
            if is_rate_limit and attempt < retries:
                print(f" Rate-limited (attempt {attempt}/{retries}). Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            elif attempt == retries:
                raise
            else:
                time.sleep(2)

def _parse_json_response(raw_text : str) -> GeneratedDataset:
    raw_text = raw_text.strip()
    
    # Try direct JSON parse first
    try:
        data = json.loads(raw_text)
        return GeneratedDataset.model_validate(data)
    except Exception:
        pass
    # Fallback: extract JSON from markdown codeblock
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return GeneratedDataset.model_validate(data)
        except Exception:
            pass
            
    # Fallback: find outer braces
    braces_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if braces_match:
        try:
            data = json.loads(braces_match.group(1))
            return GeneratedDataset.model_validate(data)
        except Exception:
            pass
    raise ValueError(f"Could not parse valid JSON object from LLM response: {raw_text[:200]}...")


def generate_questions(doc: Document, collection_name: str, llm) -> list[dict]:
    prompt = RETRIEVING_TESTING_PROMPT.format(chunk=doc.page_content)
    
    response = _invoke_with_retry(llm, prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)
    try:
        result = _parse_json_response(raw_content)
    except Exception as e:
        print(f"    ⚠️ JSON parse error: {e}")
        return []
    dataset = []
    for qa in result.questions:
        dataset.append({
            "question": qa.question,
            "ground_truth": qa.answer,
            "difficulty": qa.difficulty,
            "question_type": qa.question_type,
            "reference_context": doc.page_content,
            "patient_id": doc.metadata.get("patient_id", "GLOBAL"),
            "page_number": doc.metadata.get("page_number", doc.metadata.get("page", 0)),
            "source_document": doc.metadata.get("source", ""),
            "file_name": doc.metadata.get("file_name", ""),
            "content_type": doc.metadata.get("content_type", "text"),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "collection": collection_name,
        })
    return dataset


def build_dataset(documents: list[Document], collection_name: str, llm, existing_data: list = None) -> list[dict]:
    dataset = existing_data if existing_data is not None else []
    total = len(documents)
    for idx, doc in enumerate(documents, start=1):
        pid = doc.metadata.get("patient_id", "N/A")
        words = doc.page_content.split()
        
        # Lowered threshold to 35 words so tables, dosage specs, and captions are included
        if len(words) < 35:
            continue
        print(f"[{collection_name}] {idx}/{total} | Patient: {pid} | Words: {len(words)}")
        try:
            if idx > 1:
                time.sleep(API_CALL_DELAY)
            qa_pairs = generate_questions(doc, collection_name, llm)
            if qa_pairs:
                dataset.extend(qa_pairs)
                print(f" Generated {len(qa_pairs)} QA pairs (Total: {len(dataset)})")
                
                # Incremental save to prevent data loss
                _save_json(dataset, Path("evaluation_dataset.json"))
        except Exception as e:
            print(f"  Failed on doc {idx} ({pid}): {e}")
    return dataset

def _save_json(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def generate_evaluation_dataset():

    llm = get_llm()

    print("Starting Medical RAG Evaluation Dataset Generation...\n")
    patient_docs, pharma_docs, trial_docs = load_all_documents()

    print(f"Loaded Documents -> Patient: {len(patient_docs)} | Pharma: {len(pharma_docs)} | Trials: {len(trial_docs)}")

    dataset = []
    dataset = build_dataset(patient_docs, "patient_records", llm, dataset)
    dataset = build_dataset(pharma_docs, "pharma_guidelines", llm, dataset)
    dataset = build_dataset(trial_docs, "clinical_trials", llm, dataset)

    output_path = Path("evaluation_dataset.json")
    _save_json(dataset, output_path)

    print(f"✅ Successfully generated {len(dataset)} total QA pairs!")
    print(f"Saved to: {output_path.absolute()}")

if __name__ == "__main__":
    generate_evaluation_dataset()
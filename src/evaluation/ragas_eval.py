import os
import json
from pathlib import Path
from datasets import Dataset
import sys
import types
try:
    from langchain_google_vertexai import ChatVertexAI, VertexAI
except ImportError:
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    except ImportError:
        class ChatVertexAI: pass
        class VertexAI: pass

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _mod = types.ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _mod
if "langchain_community.llms.vertexai" not in sys.modules:
    _mod_llms = types.ModuleType("langchain_community.llms.vertexai")
    _mod_llms.VertexAI = VertexAI
    sys.modules["langchain_community.llms.vertexai"] = _mod_llms

from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from src.pipeline.graph import run_pipeline_text, get_pipeline
from src.ingestion.patient_manager import load_registry
from src.retrieval.router import SemanticRouter
from src.retrieval.hybrid_retriever import build_retriever
import numpy as np

from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.utils import config
import logging
from ragas.run_config import RunConfig

def run_pipeline_with_context(question : str, patient_id : str) -> dict : 
    pipeline, embedder, vector_stores = get_pipeline()
    router = SemanticRouter(embedder)
    intent, collections = router.route(question)

    chunks = build_retriever(
        query=question,
        patient_id=patient_id,
        collections=collections,
        vector_stores=vector_stores
    )
    result = run_pipeline_text(text_query=question, patient_id=patient_id)

    return {
        "answer" : result["answer"],
        "contexts" : [c.page_content for c in chunks]
    }
    
def _cache_key(question: str, patient_id: str) -> str:
    return f"{patient_id}::{question}"


def _load_pipeline_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  Warning: cache file {cache_path} unreadable, starting fresh")
            return {}
    return {}


def _save_pipeline_cache(cache_path: str, cache: dict) -> None:
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, cache_path)


def stratified_sample(qa_pairs: list, max_questions: int) -> list:
    if not max_questions or max_questions >= len(qa_pairs):
        return qa_pairs

    groups = {}
    for qa in qa_pairs:
        groups.setdefault(qa.get("patient_id", "GLOBAL"), []).append(qa)

    keys = list(groups.keys())
    result = []
    idx = 0
    while len(result) < max_questions:
        key = keys[idx % len(keys)]
        bucket = groups[key]
        if bucket:
            result.append(bucket.pop(0))
        idx += 1
        if all(not v for v in groups.values()):
            break
    return result[:max_questions]


def build_ragas_dataset(
    eval_json_path: str = "evaluation_dataset.json",
    max_questions : int = None,
    patient_filter : int = None,
    cache_path: str = "pipeline_cache.json",
    use_cache: bool = True,
) -> Dataset:
    with open(eval_json_path, "r") as f:
        qa_pairs = json.load(f)
    
    if patient_filter:
        qa_pairs = [q for q in qa_pairs if q.get("patient_id") == patient_filter]
    if max_questions:
        qa_pairs = stratified_sample(qa_pairs, max_questions)
    
    print(f"Running {len(qa_pairs)} questions through pipeline...")

    cache = _load_pipeline_cache(cache_path) if use_cache else {}
    if cache:
        print(f"  Loaded {len(cache)} cached pipeline results from {cache_path}")

    questions, answers, contexts, ground_truths = [], [], [], []
    failed = 0
    newly_cached = 0

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        patient_id  = qa.get("patient_id", "GLOBAL")

        if patient_id == "GLOBAL":
            registry = load_registry()
            if not registry:
                continue
            patient_id = list(registry.keys())[0]

        key = _cache_key(question, patient_id)

        if use_cache and key in cache:
            print(f"[{i+1}/{len(qa_pairs)}] patient={patient_id} | {question[:60]}... (cached)")
            cached = cache[key]
            questions.append(question)
            answers.append(cached["answer"])
            contexts.append(cached["contexts"])
            ground_truths.append(ground_truth)
            continue

        print(f"[{i+1}/{len(qa_pairs)}] patient={patient_id} | {question[:60]}...")

        try:
            result = run_pipeline_with_context(question, patient_id)
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(ground_truth)

            if use_cache:
                cache[key] = {"answer": result["answer"], "contexts": result["contexts"]}
                newly_cached += 1
                # save every question, not just at the end - this is the
                # whole point: a crash on question 15 shouldn't lose 1-14
                _save_pipeline_cache(cache_path, cache)
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1
            continue
    print(f"\nCompleted: {len(questions)} successful, {failed} failed"
          f" ({newly_cached} newly cached, {len(questions) - newly_cached} from cache)")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

logging.getLogger("httpx").setLevel(logging.WARNING)
def run_ragas_evaluation(
    eval_json_path: str = "evaluation_dataset.json",
    output_path: str = "ragas_results.json",
    max_questions: int = 40,
    pipeline_cache_path: str = "pipeline_cache.json",
    raw_results_path: str = "ragas_raw_results.csv",
    use_cache: bool = True, 
    resume_scoring: bool = True,
) -> dict:
    workers = 1 if config.USE_LOCAL else 4
    run_config = RunConfig(max_workers=workers, timeout=600)
    answer_relevancy.strictness = 1  

    dataset = build_ragas_dataset(
        eval_json_path=eval_json_path,
        max_questions=max_questions,
        cache_path=pipeline_cache_path,
        use_cache=use_cache,
    )

    
    if resume_scoring and os.path.exists(raw_results_path):
        print(f"Found existing raw results at {raw_results_path} - "
              f"skipping evaluate() and resuming from there.\n"
              f"(delete this file, or pass resume_scoring=False, to force a fresh evaluate() run)")
        import pandas as pd
        df = pd.read_csv(raw_results_path)
    else:
        llm = config.get_llm()
        ragas_llm   = LangchainLLMWrapper(llm, run_config=run_config)
        lc_embed    = LangchainHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        ragas_embed = LangchainEmbeddingsWrapper(lc_embed, run_config=run_config)

        mode_str = "LOCAL (Ollama - Sequential)" if config.USE_LOCAL else f"CLOUD API (Parallel - {workers} workers)"
        print(f"\nRunning RAGAS metrics in {mode_str} mode...")
        print(f"Evaluating {len(dataset)} questions (answer_relevancy strictness=1)...\n")
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embed,
            run_config=run_config
        )

   
        df = results.to_pandas()

        
        df.to_csv(raw_results_path, index=False)
        print(f"Saved raw per-question scores to {raw_results_path}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    nan_counts = {col: int(df[col].isna().sum()) for col in metric_cols}
    if any(nan_counts.values()):
        print(f"Warning: NaN scores present (rows that failed/timed out): {nan_counts}")

    def safe_mean(col: str) -> float:
       
        return round(float(np.nanmean(df[col])), 4)

    scores = {
        "faithfulness":      safe_mean("faithfulness"),
        "answer_relevancy":  safe_mean("answer_relevancy"),
        "context_precision": safe_mean("context_precision"),
        "context_recall":    safe_mean("context_recall"),
        "num_questions":     len(dataset),
        "nan_counts":        nan_counts,
    }
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  RAGAS Evaluation Results ({scores['num_questions']} questions)")
    print(f"{'='*50}")
    print(f"  Faithfulness        {scores['faithfulness']:.3f}  (hallucination check)")
    print(f"  Answer Relevancy    {scores['answer_relevancy']:.3f}  (on-topic answers)")
    print(f"  Context Precision   {scores['context_precision']:.3f}  (retrieval quality)")
    print(f"  Context Recall      {scores['context_recall']:.3f}  (retrieval completeness)")
    print(f"{'='*50}\n")

    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS Evaluation Pipeline")
    parser.add_argument("--max-questions", "-m", type=int, default=40, help="Number of questions to evaluate")
    parser.add_argument("--cloud", action="store_true", help="Use cloud API models (OpenRouter) with parallel evaluation for 10x faster execution")
    parser.add_argument("--quick", action="store_true", help="Quick test run evaluating only 3 questions")
    parser.add_argument("--no-cache", action="store_true", help="Ignore pipeline_cache.json and re-run every pipeline call from scratch")
    parser.add_argument("--fresh", action="store_true", help="Ignore ragas_raw_results.csv and force a fresh evaluate() run even if a completed one exists")
    args = parser.parse_args()

    if args.cloud:
        os.environ["USE_LOCAL_MODELS"] = "false"
        config.USE_LOCAL = False

    max_q = 3 if args.quick else args.max_questions
    scores = run_ragas_evaluation(
        max_questions=max_q,
        use_cache=not args.no_cache,
        resume_scoring=not args.fresh,
    )
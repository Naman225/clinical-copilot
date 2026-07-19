from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from src.pipeline.nodes import (
    transcribe_node, classify_node, retrieve_node, generate_node, verify_node
)
from src.retrieval.hybrid_retriever import load_vector_stores
from src.utils import config

class ClinicalState(TypedDict):
    audio_path : str
    patient_id : str
    embedder : Any
    vector_stores : Dict

    transcribed_query : str 
    query_intent : str
    collections_to_search : List[str]
    retrieved_chunks :  List[Document]
    answer : str
    sources : List[str]
    is_grounded : bool
    retry_count : int

def should_retry(state) -> str:
    if not state["is_grounded"] and state["retry_count"] < 2 :
        return "retrieve"
    return END

def build_graph():
    graph = StateGraph(ClinicalState)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    ## Connect with edge
    graph.set_entry_point("transcribe")
    graph.add_edge("transcribe", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")

    graph.add_conditional_edges(
        "verify",
        should_retry,
        {
            "retrieve": "retrieve",
            END: END
        }
    )
    return graph.compile()

## For app.py

_embedder = None
_vector_stores = None
_pipeline = None

def get_pipeline():
    global _embedder, _vector_stores, _pipeline
    if _pipeline is None:
        print("Loading pipeline resources...")
        _embedder = config.get_embedder()
        _vector_stores = load_vector_stores("./db", _embedder)
        _pipeline = build_graph()
        print("Pipeline ready.")
    return _pipeline, _embedder, _vector_stores

def run_pipeline(audio_path: str, patient_id: str) -> dict:
    pipeline, embedder, vector_stores = get_pipeline()
    result = pipeline.invoke({
        "audio_path":             audio_path,
        "patient_id":             patient_id,
        "embedder":               embedder,
        "vector_stores":          vector_stores,
        "transcribed_query":      "",
        "query_intent":           "",
        "collections_to_search":  [],
        "retrieved_chunks":       [],
        "answer":                 "",
        "sources":                [],
        "is_grounded":            False,
        "retry_count":            0
    })
    return {
        "transcription": result["transcribed_query"],
        "intent":        result["query_intent"],
        "answer":        result["answer"],
        "sources":       result["sources"],
        "is_grounded":   result["is_grounded"]
    }

def run_pipeline_text(text_query: str, patient_id: str) -> dict:
    """Run the pipeline with a text query (skip Whisper transcription)."""
    pipeline, embedder, vector_stores = get_pipeline()
    result = pipeline.invoke({
        "audio_path":             "",
        "patient_id":             patient_id,
        "embedder":               embedder,
        "vector_stores":          vector_stores,
        "transcribed_query":      text_query,   # pre-filled, skips transcribe
        "query_intent":           "",
        "collections_to_search":  [],
        "retrieved_chunks":       [],
        "answer":                 "",
        "sources":                [],
        "is_grounded":            False,
        "retry_count":            0
    })
    return {
        "transcription": text_query,
        "intent":        result["query_intent"],
        "answer":        result["answer"],
        "sources":       result["sources"],
        "is_grounded":   result["is_grounded"]
    }
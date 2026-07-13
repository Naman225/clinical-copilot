from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.pipeline.nodes import (
    transcribe_node, classify_node, retrieve_node, generate_node, verify_node
)
from src.retrieval.hybrid_retriever import load_vector_stores

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


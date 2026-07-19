import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_LOCAL = os.getenv("USE_LOCAL_MODELS", "true").lower() == "true"

LOCAL_LLM       = "mistral:latest"
LOCAL_VISION    = "llava:7b"
CLOUD_LLM       = "openrouter/free"
CLOUD_VISION    = "openrouter/free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY", "")

COHERE_API_KEY   = os.getenv("COHERE_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "clinical-copilot")


def get_llm():
    if USE_LOCAL:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LOCAL_LLM, temperature=0.0, num_ctx=4096)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=CLOUD_LLM,
            base_url=OPENROUTER_BASE,
            api_key=OPENROUTER_KEY,
            temperature=0.0,
            max_tokens=1024
        )


def get_vision_llm():
    if USE_LOCAL:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LOCAL_VISION, temperature=0.0, num_ctx=4096)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=CLOUD_VISION,
            base_url=OPENROUTER_BASE,
            api_key=OPENROUTER_KEY,
            temperature=0.0,
            max_tokens=1024
        )


def get_embedder():
    if USE_LOCAL:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model="embed-v4.0", cohere_api_key=COHERE_API_KEY)


def get_reranker(top_n: int = 3):
    if USE_LOCAL:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        return FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)
    else:
        from langchain_cohere import CohereRerank
        return CohereRerank(model="rerank-v3.5", top_n=top_n, cohere_api_key=COHERE_API_KEY)


def get_stt():
    if USE_LOCAL:
        from src.pipeline.transcriber import MedicalTranscriber
        return MedicalTranscriber(model_size="base")
    else:
        from src.pipeline.transcriber import GroqWhisperTranscriber
        return GroqWhisperTranscriber(api_key=GROQ_API_KEY)


def get_store_count(store) -> int:
    if hasattr(store, "_collection"):
        return store._collection.count()
    elif hasattr(store, "_index"):
        try:
            stats = store._index.describe_index_stats()
            ns = getattr(store, "_namespace", None) or getattr(store, "namespace", None)
            if ns and ns in stats.namespaces:
                return stats.namespaces[ns].vector_count
            elif ns is None:
                return stats.total_vector_count
            return 0
        except Exception:
            return 0
    return 0
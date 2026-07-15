import os

USE_LOCAL = os.getenv("USE_LOCAL_MODELS", "true").lower() == "true"

LOCAL_LLM = "mistral:latest"
LOCAL_VISION      = "llava:7b"
CLOUD_LLM         = "openrouter/free"
CLOUD_VISION      = "openrouter/free"
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"
OPENROUTER_KEY    = os.getenv("OPENROUTER_API_KEY", "")

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
    """Return a vision-capable LLM for image captioning."""
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
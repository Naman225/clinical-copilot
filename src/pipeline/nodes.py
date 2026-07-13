import re
from src.pipeline.transcriber import MedicalTranscriber
from src.retrieval.router import SemanticRouter
from src.retrieval.hybrid_retriever import load_vector_stores, build_retriever
from src.pipeline.prompts import CLINICAL_SYSTEM_PROMPT
from langchain_ollama import ChatOllama

IMAGE_CAPTION_MAX_CHARS = 500

## Node - 1 --
def transcribe_node(state):
    transcriber = MedicalTranscriber(model_size= "base")
    result = transcriber.transcribe(state['audio_path'])
    print(f"[Node 1 - Transcribe] {result['text']}")
    return {**state ,"transcribed_query" : result["text"]}

## Node - 2 --
def classify_node(state):
    router = SemanticRouter(state["embedder"])
    intent, collections = router.route(state["transcribed_query"])
    print(f"[Node 2 - Classify] intent={intent}, collections={collections}")
    return {**state , "query_intent": intent, "collections_to_search": collections}

## Node - 3 --
def retrieve_node(state):
    chunks = build_retriever(
        query = state["transcribed_query"],
        patient_id= state["patient_id"],
        collections= state["collections_to_search"],
        vector_stores= state["vector_stores"]
    )
    print(f"[Node 3 - Retrieve] Got {len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks}


def reformat_chunk(doc) -> str:
    content = doc.page_content
    meta = doc.metadata

    if meta.get("content_type") != "table":
        return content

  
    try:
        lines = []
       
        entries = {}
        for part in content.split(". "):
            part = part.strip()
            if not part:
                continue
            if ", Result = " in part:
                name, val = part.split(", Result = ", 1)
                name = name.strip()
                if name not in entries:
                    entries[name] = {}
                entries[name]["result"] = val.strip()
            elif ", Unit = " in part:
                name, val = part.split(", Unit = ", 1)
                name = name.strip()
                if name not in entries:
                    entries[name] = {}
                entries[name]["unit"] = val.strip()
            elif ", Flag = " in part:
                name, val = part.split(", Flag = ", 1)
                name = name.strip()
                if name not in entries:
                    entries[name] = {}
                entries[name]["flag"] = val.strip()

        if entries:
            lines.append("Laboratory Results:")
            lines.append("-" * 40)
            for test_name, values in entries.items():
                result = values.get("result", "?")
                unit   = values.get("unit", "")
                flag   = values.get("flag", "")
                flag_marker = " ⚠️ CRITICAL" if flag in ["HIGH", "LOW"] else f" ({flag})"
                lines.append(f"  {test_name}: {result} {unit}{flag_marker}")
            return "\n".join(lines)
    except Exception:
        pass

    return content  


## Node - 4 --
def generate_node(state):
    context_blocks = []
    sources = []

    for i, doc in enumerate(state["retrieved_chunks"]):
        meta = doc.metadata
        content_type = meta.get("content_type", "?")

        formatted_content = reformat_chunk(doc)
        if content_type == "medical_image" and len(formatted_content) > IMAGE_CAPTION_MAX_CHARS:
            formatted_content = formatted_content[:IMAGE_CAPTION_MAX_CHARS] + "\n[... image description truncated]"

        context_blocks.append(
            f'Source {i+1} ({content_type} from '
            f'{meta.get("file_name","?")} page {meta.get("page_number","?")}):\n'
            f'{formatted_content}'
        )
        sources.append({
            "index": i+1,
            "file": meta.get("file_name", "?"),
            "page": meta.get("page_number", "?"),
            "type": content_type,
            "preview": doc.page_content[:100]
        })

    context = "\n\n---\n\n".join(context_blocks)

    # print full context this time — not truncated
    print("\n" + "="*60)
    print("FULL CONTEXT SENT TO MODEL:")
    print(context)
    print("="*60 + "\n")

    user_message = f"""Sources:
{context}

Question: {state["transcribed_query"]}

Using ONLY the exact values from the sources above, answer the question.
Copy numbers exactly as written. Do not add any values not in the sources.
Flag any CRITICAL values prominently."""

    llm = ChatOllama(model="mistral:latest", temperature=0.0, num_ctx=4096)
    response = llm.invoke([
        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ])

    print(f"[Node 4 - Generate] {len(response.content)} chars")
    return {**state, "answer": response.content, "sources": sources}


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values (int and float) from a text string."""
    return set(re.findall(r'\b\d+\.?\d*\b', text))


## Node - 5 --
def verify_node(state):

    not_found = [
        "do not contain",
        "cannot find",
        "not in the context",
        "no information"
    ]
    has_refusal = any(p in state['answer'].lower() for p in not_found)

  
    context_numbers = set()
    for doc in state.get("retrieved_chunks", []):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        context_numbers.update(_extract_numbers(content))

    # Extract numbers from the model's answer
    answer_numbers = _extract_numbers(state["answer"])

    # Filter out trivially common numbers (source indices, pagination, etc.)
    trivial = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0"}
    answer_numbers -= trivial
    context_numbers -= trivial

    # Numbers in the answer that are NOT from any retrieved chunk
    fabricated = answer_numbers - context_numbers
    if fabricated:
        print(f"[Node 5 - Verify] ⚠️  Potentially fabricated values: {fabricated}")

    grounded = (not has_refusal) and (len(fabricated) == 0)
    print(f"[Node 5 - Verify] grounded={grounded}, retries={state['retry_count']}")

    return{
        **state,
        "is_grounded" : grounded,
        "retry_count" : state["retry_count"]+1
    }
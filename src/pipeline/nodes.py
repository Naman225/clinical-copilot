import re
from src.retrieval.router import SemanticRouter
from src.retrieval.hybrid_retriever import load_vector_stores, build_retriever
from src.pipeline.prompts import CLINICAL_SYSTEM_PROMPT, CLINICAL_USER_TEMPLATE
from src.utils import config
from src.utils.config import get_llm

IMAGE_CAPTION_MAX_CHARS = 500

_LEAK_PATTERNS = [
    r"^\s*\d+\.\s*[A-Z][A-Za-z /&\-]{2,60}:?\s*$",
    r"^\s*Answer format:\s*$",
    r"^\s*CRITICAL INSTRUCTIONS:\s*$",
    r"^\s*Correct answer:\s*$",
    r"^\s*---\s*(EXAMPLE|END EXAMPLE)\s*---\s*$",
]
_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), re.MULTILINE)


def strip_leaked_instructions(text: str) -> str:
    cleaned = _LEAK_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

## Node - 1 --
def transcribe_node(state):
    if state.get("transcribed_query", "").strip():
        print(f"[Node 1 - Transcribe] Pre-filled: {state['transcribed_query']}")
        return state
    transcriber = config.get_stt()
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
    if not chunks:
        print(f"[Node 3 - Retrieve] ⚠️  No records found for patient_id='{state['patient_id']}'")
    else:
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
    
    if not state["retrieved_chunks"]:
        no_data_msg = (
            f"No medical records were found for patient ID '{state['patient_id']}'. "
            f"Please verify the patient ID and try again. "
            f"No clinical data can be provided without matching records."
        )
        print(f"[Node 4 - Generate] No data — returning safe message")
        return {**state, "answer": no_data_msg, "sources": []}

    target_pid = state["patient_id"]
    filtered_chunks = []
    for doc in state["retrieved_chunks"]:
        chunk_pid = doc.metadata.get("patient_id", "")
        # Allow patient_records that match, and GLOBAL reference docs
        if chunk_pid == target_pid or chunk_pid == "GLOBAL":
            filtered_chunks.append(doc)
        else:
            print(f"[Node 4 - Generate] ⚠️  Dropped cross-patient chunk "
                  f"(wanted={target_pid}, got={chunk_pid})")

    if not filtered_chunks:
        no_data_msg = (
            f"No medical records matched patient ID '{target_pid}' after filtering. "
            f"Please verify the patient ID and try again."
        )
        print(f"[Node 4 - Generate] All chunks filtered out — no matching patient data")
        return {**state, "answer": no_data_msg, "sources": []}

    context_blocks = []
    sources = []

    for i, doc in enumerate(filtered_chunks):
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
    print(f"FULL CONTEXT SENT TO MODEL (patient_id={target_pid}):")
    print(context)
    print("="*60 + "\n")

    user_message = CLINICAL_USER_TEMPLATE.format(
        context=context,
        query=state["transcribed_query"]
    )

    llm = get_llm()
    response = llm.invoke([
        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ])

    answer = strip_leaked_instructions(response.content)
    print(f"[Node 4 - Generate] {len(answer)} chars")
    return {**state, "answer": answer, "sources": sources}


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values (int and float) from a text string."""
    return set(re.findall(r'\b\d+\.?\d*\b', text))


## Node - 5 --
def verify_node(state):

    # If no chunks were retrieved (patient not found), the answer is a safe
    # system-generated message — mark as grounded and skip verification.
    if not state.get("retrieved_chunks"):
        print(f"[Node 5 - Verify] No data retrieved — marking as grounded (safe message)")
        return {**state, "is_grounded": True, "retry_count": state["retry_count"] + 1}

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
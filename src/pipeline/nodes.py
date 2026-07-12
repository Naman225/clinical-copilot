from src.pipeline.transcriber import MedicalTranscriber
from src.retrieval.router import SemanticRouter
from src.retrieval.hybrid_retriever import load_vector_stores, build_retriever
from src.pipeline.prompts import CLINICAL_SYSTEM_PROMPT, build_prompt
from langchain_ollama import ChatOllama


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
    return {**state , "query_intent": intent, "collection_to_search": collections}

## Node - 3 --
def retrieve_node(state):
    retriever = build_retriever(
        query = state["transcriber_query"],
        patient_id= state["patient_id"],
        collections= state["collections"],
        vector_stores= state["vector_stores"]
    )
    print(f"[Node 3 - Retrieve] Got {len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks}

## Node - 4 --
def generate_node(state):
    context_blocks = []
    sources = []
    for i, doc in enumerate(state["retrieved_chunks"]):
        meta = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc

        ## For LLM - formatted Context
        context_blocks.append(
            f'<source index ="{i+1}"'
            f'file="{meta.get("file_name","?")}" '
            f'page="{meta.get("page_number","?")}" '
            f'type="{meta.get("content_type","?")}">\n'
            f'{doc.page_content.strip()}\n'
            f'</source>'
        )

        ## for UI

        sources.append({
            "index": i+1,
            "file" : meta.get("file_name","?"),
            "page" : meta.get("page_number","?"),
            "type": meta.get("content_type", "?"),
            "preview": doc.page_content[:100]
        })

        context_string = "\n\n".join(context_blocks)

    llm = ChatOllama(model = "llama3:8b", temperature = 0.0,num_ctx= 4096)
    prompt = build_prompt(context_blocks, state["transcribed_query"])
    response = llm.invoke([
        {"role":"system", "content": CLINICAL_SYSTEM_PROMPT},
        {"role":"user", "content": prompt}
    ])
    print(f"[Node 4 - Generate] {len(response.content)} chars")
    return {**state, "answer": response.content, "sources":sources}


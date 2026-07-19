import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.documents import Document
from src.utils import config

COLLECTIONS = ["patient_records", "pharma_guidelines", "clinical_trials"]


def load_vector_stores(db_path: str, embedder) -> dict:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]
    if config.PINECONE_INDEX not in existing_indexes:
        print(f"Creating Pinecone index '{config.PINECONE_INDEX}'...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(config.PINECONE_INDEX)
    return {
        name: PineconeVectorStore(index=index, embedding=embedder, namespace=name)
        for name in COLLECTIONS
    }


def _all_docs(store, namespace: str, patient_id: str = None) -> list[Document]:
    """Fetch every document in a collection (optionally filtered to a patient_id).
    Used to build the BM25 keyword index alongside the semantic retriever."""
    index = store.index
    ids = [id_ for batch in index.list(namespace=namespace) for id_ in batch]
    docs = []
    for i in range(0, len(ids), 100):
        batch_ids = ids[i:i + 100]
        fetched = index.fetch(ids=batch_ids, namespace=namespace)
        for vec in fetched.vectors.values():
            meta = vec.metadata or {}
            if patient_id and meta.get("patient_id") != patient_id:
                continue
            docs.append(Document(page_content=meta.get("text", ""), metadata=meta))
    return docs


def get_patient_docs(store, patient_id: str) -> list[Document]:
    return _all_docs(store, namespace="patient_records", patient_id=patient_id)


def build_retriever(
    query: str,
    patient_id: str,
    collections: list[str],
    vector_stores: dict,
    k: int = 10,
    top_n_after_rerank: int = 3
) -> list[Document]:
    all_retrievers = []
    all_bm25_docs = []

    for col_name in collections:
        store = vector_stores[col_name]

        if col_name == "patient_records":
            semantic_retriever = store.as_retriever(
                search_kwargs={"k": k, "filter": {"patient_id": patient_id}}
            )
            bm25_docs = _all_docs(store, namespace=col_name, patient_id=patient_id)
        else:
            semantic_retriever = store.as_retriever(search_kwargs={"k": k})
            bm25_docs = _all_docs(store, namespace=col_name)

        all_retrievers.append(semantic_retriever)
        all_bm25_docs.extend(bm25_docs)

    if not all_bm25_docs:
        print(f"[Retriever] No documents found for patient_id='{patient_id}' — returning empty.")
        return []

    bm25_retriever = BM25Retriever.from_documents(all_bm25_docs)
    bm25_retriever.k = k * len(collections)

    all_retrievers.insert(0, bm25_retriever)
    weights = [0.4] + [0.6 / len(collections)] * len(collections)

    ensemble = EnsembleRetriever(retrievers=all_retrievers, weights=weights)
    compressor = config.get_reranker(top_n=top_n_after_rerank)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )

    return compression_retriever.invoke(query)
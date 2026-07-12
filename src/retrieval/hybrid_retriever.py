from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb

RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

def load_vector_stores(db_path: str, embedder: HuggingFaceEmbeddings) -> dict:
    return{
        "patient_records": Chroma(
            collection_name="patient_records",
            persist_directory=db_path,
            embedding_function= embedder
        ),
        "pharma_guidelines": Chroma(
            collection_name="pharma_guidelines",
            persist_directory=db_path,
            embedding_function= embedder
        ),
        "clinical_trials": Chroma(
            collection_name="clinical_trials",
            persist_directory=db_path,
            embedding_function=embedder
        )
    }

def get_patient_docs(vector_stores: Chroma, patient_id: str) -> list[Document]:
    results = vector_stores._collection.get(where={"patient_id": patient_id},
        include=["documents", "metadatas"]
    )
    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]

def build_retriever(
    query : str,
    patient_id : str,
    collections : list[str],
    vector_stores : str,
    k : int = 10,
    top_n_after_rerank : int = 3
) -> list[Document]:
    all_retrievers = []
    all_bm25_docs = []

    for col_name in collections:
        store = vector_stores[col_name]

        if col_name == "patient_records":
            semantic_retriever = store.as_retriever(
                search_kwargs = {
                    "k" : k,
                    "filter" : {"patient_id" : patient_id}
                }
            )
            bm25_docs = get_patient_docs(store, patient_id)
        else:
            semantic_retriever = store.as_retriever(
                search_kwargs= {
                    "k": k
                }
            )
            results = store._collection.get(include = ["documents", "metadatas"])
            bm25_docs = [
                Document(page_content=doc, metadata = meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]

        all_retrievers.append(semantic_retriever)
        all_bm25_docs.extend(bm25_docs)
    
    bm25_retrievers = BM25Retriever.from_documents(all_bm25_docs)
    bm25_retrievers.k = k * len(collections)

    all_retrievers.insert(0, bm25_retrievers)
    weights = [0.4] + [0.6 / len(collections)] * len(collections)

    ensemble = EnsembleRetriever(
        retrievers = all_retrievers,
        weights= weights
    )
    compressor = FlashrankRerank(
        model=RERANK_MODEL,
        top_n=top_n_after_rerank
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )

    return compression_retriever.invoke(query)       




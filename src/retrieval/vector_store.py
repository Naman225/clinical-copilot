import os 
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma

def load_vector_stores(self, embedder) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChromaDB path not found at: {db_path}")
    
    return{
        "patient_records": Chroma(
            db_path = db_path,
            collection_name = "patient_records",
            embedding_function= embedder
        ),
        "pharma_guidelines": Chroma(
            db_path = db_path,
            collection_name = "pharma_guidelines",
            embedding_function= embedder
        ),
        "clinical_trials": Chroma(
            db_path = db_path,
            collection_name = "clinical_trials",
            embedding_function= embedder
        )
    }
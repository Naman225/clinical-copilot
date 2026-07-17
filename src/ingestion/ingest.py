from pathlib import Path
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.image_loader import caption_images
from src.ingestion.reference_loader import load_reference_path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
def load_all_documents(
    DATA_DIR    = "./data_ingestion/multi_patient_db",
    PHARMA_DIR  = "./data_ingestion/pharma_guidelines",
    TRIALS_DIR  = "./data_ingestion/clinical_trials"
): 
    patient_docs = []

    for patient_dir in sorted(Path(DATA_DIR).iterdir()):
        if not patient_dir.is_dir():
           continue
        pdf_path = patient_dir / "patient_report.pdf"
        if not pdf_path.exists():
            continue
        patient_id = patient_dir.name
        text_docs, image_stubs = load_pdf(str(pdf_path), patient_id)
    
        image_docs = caption_images(image_stubs)
        patient_docs.extend(text_docs + image_docs)
    patient_docs = filter_complex_metadata(patient_docs)

    pharma_docs = load_reference_path(
        list(Path(PHARMA_DIR).glob("*.pdf")),
        "pharma_guideline"
    )
    trial_docs = load_reference_path(
        list(Path(TRIALS_DIR).glob("*.pdf")),
        "clinical_trial"
    )
    return patient_docs, pharma_docs, trial_docs

def build_vectorstores():
    patient_docs, pharma_docs, trial_docs = load_all_documents()
    print("\n Building Chroma")
    patient_store = Chroma.from_documents(
        patient_docs, embedder,
        collection_name="patient_records",
        persist_directory="./db"
    )
    pharma_store = Chroma.from_documents(
        pharma_docs, embedder,
        collection_name="pharma_guidelines",
        persist_directory="./db"
    )
    trial_store = Chroma.from_documents(
        trial_docs, embedder,
        collection_name="clinical_trials",
        persist_directory="./db"
    )

    print(f"\nDone.")
    print(f"  Patient chunks:  {patient_store._collection.count()}")
    print(f"  Pharma chunks:   {pharma_store._collection.count()}")
    print(f"  Trial chunks:    {trial_store._collection.count()}")

if __name__ == "__main__":
    build_vectorstores()
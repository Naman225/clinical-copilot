from pathlib import Path
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.image_loader import caption_images
from src.ingestion.reference_loader import load_reference_path
from langchain_community.vectorstores.utils import filter_complex_metadata
from src.utils import config

COLLECTIONS = ["patient_records", "pharma_guidelines", "clinical_trials"]


def load_all_documents(
    DATA_DIR="./data_ingestion/multi_patient_db",
    PHARMA_DIR="./data_ingestion/pharma_guidelines",
    TRIALS_DIR="./data_ingestion/clinical_trials"
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
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    embedder = config.get_embedder()
    patient_docs, pharma_docs, trial_docs = load_all_documents()
    print("\n Pushing to Pinecone")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    from pinecone import ServerlessSpec
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

    doc_sets = {
        "patient_records": patient_docs,
        "pharma_guidelines": pharma_docs,
        "clinical_trials": trial_docs,
    }
    for namespace, docs in doc_sets.items():
        if not docs:
            print(f"  {namespace}: no documents, skipping")
            continue
        PineconeVectorStore.from_documents(
            docs, embedder, index_name=config.PINECONE_INDEX, namespace=namespace
        )
        try:
            stats = index.describe_index_stats()
            count = stats.namespaces[namespace].vector_count
            print(f"  {namespace}: {count} vectors")
        except Exception:
            print(f"  {namespace}: uploaded {len(docs)} chunks")

    print("\nDone.")


if __name__ == "__main__":
    build_vectorstores()
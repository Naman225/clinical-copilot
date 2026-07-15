import json
from pathlib import Path
import chromadb
from langchain_community.vectorstores.utils import filter_complex_metadata
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.image_loader import caption_images

PATIENT_REGISTRY = Path("./db/patient_registry.json")

def load_registry() -> dict:
    if PATIENT_REGISTRY.exists():
        return json.loads(PATIENT_REGISTRY.read_text())
    return {}
def save_registry(registry: dict):
    PATIENT_REGISTRY.parent.mkdir(exist_ok=True)
    PATIENT_REGISTRY.write_text(json.dumps(registry, indent=2))

def get_next_patient_id(registry: dict) -> str:
    if not registry:
        return "1001"
    existing = [int(k) for k in registry.keys() if k.isdigit()]
    return str(max(existing) + 1)

def patient_exists(patient_id: str) -> bool:
    registry = load_registry()
    return patient_id in registry

def index_patient_pdf(
    pdf_path: str,
    patient_store,
    embedder,
    patient_id: str = None,
    replace: bool = False
):
    registry = load_registry()

    if patient_id is None:
        patient_id = get_next_patient_id(registry)
    elif patient_id in registry and not replace:
        return patient_id, f"Patient {patient_id} already exists. Use replace=True to overwrite."
    if replace and patient_id in registry:
        collection = patient_store._collection
        existing = collection.get(where = {"patient_id": patient_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"Deleted {len(existing['ids'])} existing chunks for patient {patient_id}")
        
    text_docs, image_stubs = load_pdf(
        pdf_path= pdf_path,
        patient_id=patient_id,
        image_output_dir=f"./extracted_images"
    )

    image_docs = caption_images(image_stubs)
    all_docs = text_docs + image_docs
    patient_store.add_documents(all_docs)

    # update registry
    registry[patient_id] = {
        "pdf_path": pdf_path,
        "chunk_count": len(all_docs),
        "file_name": Path(pdf_path).name
    }

    save_registry(registry)
    msg = f"Patient {patient_id} indexed: {len(all_docs)} chunks from {Path(pdf_path).name}"
    print(msg)
    return patient_id, msg

def get_patient_list() -> list[dict]:
    """Returns list of {id, file_name} for Gradio dropdown."""
    registry = load_registry()
    return [
        {"id": pid, "label": f"Patient {pid} — {info['file_name']}"}
        for pid, info in sorted(registry.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    ]
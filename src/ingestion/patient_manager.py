import json
from pathlib import Path
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
        try:
            patient_store.delete(filter={"patient_id": patient_id})
            print(f"Deleted existing chunks for patient {patient_id} from Pinecone.")
        except Exception as e:
            print(f"Warning: could not delete patient chunks: {e}")
        
    text_docs, image_stubs = load_pdf(
        pdf_path= pdf_path,
        patient_id=patient_id,
        image_output_dir=f"./extracted_images"
    )

    image_docs = caption_images(image_stubs)
    all_docs = filter_complex_metadata(text_docs + image_docs)
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


def remove_all_patients(patient_store=None):
    """Removes all indexed patient records from Pinecone database, extracted images, and patient registry."""
    import shutil
    from src.utils import config

    # 1. Clear Pinecone patient_records namespace
    if patient_store is not None and hasattr(patient_store, "delete"):
        try:
            patient_store.delete(delete_all=True, namespace="patient_records")
            print("Deleted all vectors from Pinecone database 'patient_records' namespace.")
        except Exception as e:
            print(f"Warning clearing Pinecone patient_records: {e}")
    else:
        try:
            from pinecone import Pinecone
            if config.PINECONE_API_KEY:
                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                index = pc.Index(config.PINECONE_INDEX)
                index.delete(delete_all=True, namespace="patient_records")
                print("Deleted all vectors from Pinecone database 'patient_records' namespace.")
        except Exception as e:
            pass

    # 2. Clear patient registry
    save_registry({})

    # 3. Clear extracted_images directory
    extracted_dir = Path("./extracted_images")
    if extracted_dir.exists():
        for item in extracted_dir.iterdir():
            if item.is_dir() and item.name != ".gitkeep":
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file() and item.name != ".gitkeep":
                item.unlink(missing_ok=True)
    print("All old patient data has been cleanly purged from Pinecone and local registries.")
import json
from pathlib import Path
from langchain_core.documents import Document

def load_sample_patient_chunks(json_path: str = "./src/evaluation/sample_patient_chunks.json") -> list[Document]:
    """
    Loads synthetic testing patient chunks (`patient_test_001`) into LangChain `Document` objects.
    Useful for offline evaluation, unit tests, and verifying RAG pipeline behavior without running
    Docling or OCR parsing on actual PDFs.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Testing chunk file not found: {path.absolute()}")
        
    with open(path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
        
    documents = []
    for chunk in raw_chunks:
        documents.append(Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"]
        ))
    return documents

if __name__ == "__main__":
    docs = load_sample_patient_chunks()
    print(f"✅ Successfully loaded {len(docs)} synthetic patient testing chunks across all multimodal types:")
    for i, d in enumerate(docs, 1):
        content_type = d.metadata.get("content_type", "unknown")
        section = d.metadata.get("section_headings", "Page " + str(d.metadata.get("page_number", "")))
        print(f"  [{i}] Type: {content_type.upper():<14} | Section: {section:<35} | Length: {len(d.page_content)} chars")

from pathlib import Path
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from src.ingestion.pdf_loader import ExtractedImage
import base64

def caption_images(
    image_stubs: list[ExtractedImage],
    model : str = 'llava:7b'
) -> list[Document]:
    if not image_stubs:
        return []
    print(f"  Captioning {len(image_stubs)} images with {model}...") 
    llm = ChatOllama(model = model, temperature = 0.0)
    documents =[]

    for stub in image_stubs:
        with open(stub.image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = f"""You are a medical imaging specialist.
        Document context: {stub.surrounding_text[:300]}
        Describe this medical image clinically. Include all visible findings,
        measurements, labels, and clinical significance."""

        response = llm.invoke([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }])
        documents.append(Document(
            page_content=f"Medical Image — Page {stub.page_number}:\n{response.content}",
            metadata={
                "patient_id": stub.patient_id,
                "source": stub.source,
                "file_name": Path(stub.source).name,
                "content_type": "medical_image",
                "page_number": stub.page_number,
                "image_path": str(stub.image_path)
            }
        ))
        print(f"  Captioned: {stub.image_path.name} "
              f"(page {stub.page_number}, patient {stub.patient_id})")

    return documents
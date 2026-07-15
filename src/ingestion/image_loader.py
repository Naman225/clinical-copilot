import time
import base64
from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.pdf_loader import ExtractedImage
from src.utils.config import get_vision_llm

# Rate-limit settings for free-tier cloud APIs
API_CALL_DELAY = 4          # seconds between successive vision calls
MAX_RETRIES = 5             # retry attempts on 429 / transient errors
INITIAL_BACKOFF = 10        # first retry waits this many seconds


def _invoke_with_retry(llm, messages, retries=MAX_RETRIES):
    """Invoke the LLM with exponential backoff on rate-limit (429) errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()
            if is_rate_limit and attempt < retries:
                print(f"    ⏳ Rate-limited (attempt {attempt}/{retries}). "
                      f"Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)   # cap at 2 minutes
            else:
                raise  # not a rate-limit error, or exhausted retries


def caption_images(
    image_stubs: list[ExtractedImage],
) -> list[Document]:
    """Caption medical images using the configured vision LLM (local or cloud)."""
    if not image_stubs:
        return []

    llm = get_vision_llm()
    model_name = getattr(llm, "model_name", getattr(llm, "model", "vision-llm"))
    print(f"  Captioning {len(image_stubs)} images with {model_name}...")

    documents = []

    for i, stub in enumerate(image_stubs):
        with open(stub.image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = f"""You are a medical imaging specialist.
        Document context: {stub.surrounding_text[:300]}
        Describe this medical image clinically. Include all visible findings,
        measurements, labels, and clinical significance."""

        # Throttle: wait between calls to stay under free-tier rate limits
        if i > 0:
            print(f"    ⏳ Waiting {API_CALL_DELAY}s (rate-limit throttle)...")
            time.sleep(API_CALL_DELAY)

        response = _invoke_with_retry(llm, [{
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
        print(f"  ✓ Captioned: {stub.image_path.name} "
              f"(page {stub.page_number}, patient {stub.patient_id})")

    return documents
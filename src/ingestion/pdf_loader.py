from pathlib import Path
from langchain_core.documents import Document
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions   
from docling.datamodel.base_models import InputFormat
from dataclasses import dataclass
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from langchain_community.vectorstores.utils import filter_complex_metadata

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
@dataclass
class ExtractedImage:
    """Holds image path + surrounding text context for LLaVA."""
    image_path: Path
    page_number: int
    surrounding_text: str
    patient_id: str
    source: str

def load_pdf(
    pdf_path: str,
    patient_id: str,
    image_output_dir: str = "./extracted_images"
) -> tuple[list[Document], list[ExtractedImage]]:

    img_dir = Path(image_output_dir) / patient_id
    img_dir.mkdir(parents=True, exist_ok=True)
    cpu_accelerator = AcceleratorOptions(
        num_threads=4,                     
        device=AcceleratorDevice.CPU        
    )
    options = PdfPipelineOptions()
    options.accelerator_options = cpu_accelerator
    options.generate_picture_images = True
    options.images_scale = 2.0
    options.do_table_structure = True
    options.do_ocr = False
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    base_meta = {
        "patient_id": patient_id,
        "source": pdf_path,
        "file_name": Path(pdf_path).name,
    }

    ## Text + tables extraction
    chunker = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=400)
    text_docs = []
    page_text_map = {}

    for i, chunk in enumerate(chunker.chunk(doc)):
        content = chunk.text.strip()
        
    
        if len(content) < 30:
            continue
            
        page_num = 0
        if (hasattr(chunk, "meta") and chunk.meta
                and hasattr(chunk.meta, "doc_items")
                and chunk.meta.doc_items):
            prov = getattr(chunk.meta.doc_items[0], "prov", None)
            if prov:
                page_num = prov[0].page_no

        page_text_map.setdefault(page_num, "")
        page_text_map[page_num] += f" {content}"

        headings = getattr(chunk.meta, "headings", []) if chunk.meta else []
        content_type = "table" if ("|" in content and "---" in content) else "text"

        text_docs.append(Document(
            page_content=content,
            metadata={
                **base_meta,
                "content_type": content_type,
                "page_number": page_num,
                "element_index": i,
                "section_headings": " > ".join(headings) if headings else ""
            }
        ))

    ## Images extraction loop
    image_stubs = []

    for i, picture in enumerate(doc.pictures):
        if not picture.image:
            continue
        
        img_path = img_dir / f"image_{i}.png"
        if not img_path.exists():
            pil_img = picture.get_image(doc) 
            if pil_img:
                pil_img.save(str(img_path))
        else:
            print(f"    ⏭️  Image already exists: {img_path.name}")

        img_page = 0
        if hasattr(picture, "prov") and picture.prov:
            img_page = picture.prov[0].page_no

        surrounding = " ".join([
            page_text_map.get(p, "")
            for p in [img_page - 1, img_page, img_page + 1]
        ]).strip()


        image_stubs.append(ExtractedImage(
            image_path=img_path,
            page_number=img_page,
            surrounding_text=surrounding,
            patient_id=patient_id,
            source=pdf_path
        ))
    
    text_docs = filter_complex_metadata(text_docs)
    return text_docs, image_stubs


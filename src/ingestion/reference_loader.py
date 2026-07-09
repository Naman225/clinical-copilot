from pathlib import Path
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOMAIN_CONFIG = {
    "pharma_guideline": {
        "collection_hint": "pharma_guidelines",
        "description": "pharmaceutical drug guidelines and dosage information"
    },
    "clinical_trial": {
        "collection_hint": "clinical_trials",
        "description": "clinical trial results and research findings"
    }
}


def _cpu_converter() -> DocumentConverter:
    """Build a Docling converter that runs entirely on CPU."""
    cpu_accel = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,
    )
    opts = PdfPipelineOptions()
    opts.accelerator_options = cpu_accel
    opts.do_table_structure = True
    opts.do_ocr = False
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        }
    )


def load_reference_path(
    pdf_paths : list,
    domain : str
) -> list[Document]:
    if not pdf_paths:
        print(f"  No PDFs found for domain: {domain}")
        return []
    str_paths = [str(p) for p in pdf_paths]
    print(f"\n loading {len(str_paths)} {domain} PDFs ...")
    for p in str_paths:
        print(f" ->{Path(p).name}")
    all_extracted_docs = []
    for path_str in str_paths:
        try:
            loader = DoclingLoader(
                file_path= path_str,
                export_type=ExportType.DOC_CHUNKS,
                chunker= HybridChunker(tokenizer = EMBED_MODEL, max_tokens= 400),
                converter=_cpu_converter(),
            )
        
            file_docs = loader.load()

            all_extracted_docs.extend(file_docs)
            all_extracted_docs = filter_complex_metadata(all_extracted_docs)
            
        except Exception as e:
            print(f" Error parsing file {Path(path_str).name}: {str(e)}")
            continue

    for doc in all_extracted_docs:
        doc.metadata["domain"] = domain
        doc.metadata["patient_id"] = "GLOBAL" 
        doc.metadata["content_type"] = (
            "table" if ("|" in doc.page_content and "---" in doc.page_content)
            else "text"
        )

        if "file_name" not in doc.metadata:
            source = doc.metadata.get("source", "")
            doc.metadata["file_name"] = Path(source).name

    final_filtered_docs = [d for d in all_extracted_docs if len(d.page_content.strip()) > 30]

    print(f"  → {len(final_filtered_docs)} chunks extracted from {domain} PDFs")
    return final_filtered_docs

import gradio as gr
import tempfile
import time
import os
from pathlib import Path
from src.pipeline.graph import run_pipeline, get_pipeline
from src.ingestion.patient_manager import (
    index_patient_pdf, get_patient_list,
    patient_exists, load_registry
)
import pandas as pd

# startup: load pipeline resources once
print("Initializing pipeline...")
pipeline, embedder, vector_stores = get_pipeline()
patient_store = vector_stores["patient_records"]

# helpers
def _build_choices():
    """Build (label, id) choices list from the registry."""
    patients = get_patient_list()
    return [(p["label"], p["id"]) for p in patients]

def refresh_patient_choices():
    """Return a Dropdown update with the latest patient list."""
    choices = _build_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)

# upload handler — supports multiple PDFs (up to 10)
def handle_upload(pdf_files, manual_id, replace_existing):
    if pdf_files is None or len(pdf_files) == 0:
        return "No files uploaded.", refresh_patient_choices(), refresh_patient_choices()

    base_id = manual_id.strip() if manual_id.strip() else None

    results = []
    for i, pdf_file in enumerate(pdf_files):
        # Determine patient_id for this file
        if base_id:
            # If user gave a base ID and there are multiple files,
            # append a suffix: 1001, 1001_2, 1001_3, ...
            if len(pdf_files) == 1:
                patient_id = base_id
            else:
                patient_id = base_id if i == 0 else f"{base_id}_{i + 1}"
        else:
            patient_id = None  # auto-assign

        # Check if patient already exists
        if patient_id and patient_exists(patient_id) and not replace_existing:
            results.append(
                f"⚠️ Patient {patient_id} ({Path(pdf_file.name).name}) — "
                f"already exists. Enable 'Replace' to overwrite."
            )
            continue

        try:
            # Pause between PDFs to respect free-tier rate limits
            if i > 0:
                print(f"  ⏳ Pausing 5s between PDFs (rate-limit cooldown)...")
                time.sleep(5)

            pid, msg = index_patient_pdf(
                pdf_path=pdf_file.name,
                patient_store=patient_store,
                embedder=embedder,
                patient_id=patient_id,
                replace=replace_existing
            )
            results.append(f"✓ {msg}")
        except Exception as e:
            fname = Path(pdf_file.name).name
            results.append(f"✗ Failed to index {fname}: {e}")

    status = "\n".join(results)
    # Return status + refresh BOTH dropdowns (management tab + query tab)
    return status, refresh_patient_choices(), refresh_patient_choices()

# query handler
def handle_query(audio_path, patient_id):
    if audio_path is None:
        return "No audio provided.", "", pd.DataFrame(), "No query"

    if not patient_id:
        return "No patient selected.", "", pd.DataFrame(), "No patient"

    result = run_pipeline(audio_path=audio_path, patient_id=str(patient_id))

    # build sources dataframe for display
    sources_df = pd.DataFrame(result["sources"]) if result["sources"] else pd.DataFrame()

    grounded_label = "✓ Grounded" if result["is_grounded"] else "⚠️ Unverified"

    return (
        result["answer"],
        f"**Transcription:** {result['transcription']}\n"
        f"**Intent:** {result['intent']} | **Status:** {grounded_label}",
        sources_df,
        result["transcription"]
    )

# UI
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    title="Clinical Intelligence Co-Pilot"
) as demo:

    gr.Markdown("""
    # 🏥 Clinical Intelligence & Diagnostics Co-Pilot
    Upload patient PDFs, then ask questions via voice or text.
    """)

    # ── Tab 1: Patient Management ──────────────────────────────────
    with gr.Tab("📋 Patient Management"):
        gr.Markdown("### Upload Patient Records")
        with gr.Row():
            with gr.Column():
                pdf_upload = gr.File(
                    label="Patient PDFs (upload up to 10 at once)",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                gr.Markdown(
                    "*You can select multiple PDF files at once. "
                    "Each file will be indexed as a separate patient.*"
                )
                manual_id = gr.Textbox(
                    label="Patient ID (leave blank to auto-assign all)",
                    placeholder="e.g. 1001 — for multiple files, IDs will be 1001, 1001_2, 1001_3..."
                )
                replace_checkbox = gr.Checkbox(
                    label="Replace existing patient data if ID already exists",
                    value=False
                )
                upload_btn = gr.Button("Index Patients", variant="primary")
            with gr.Column():
                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=8,
                    interactive=False
                )

        # Dropdown showing indexed patients (in this tab)
        init_choices = _build_choices()
        mgmt_dropdown = gr.Dropdown(
            choices=init_choices,
            value=init_choices[0][1] if init_choices else None,
            label="Select Patient"
        )

    # ── Tab 2: Clinical Query ──────────────────────────────────────
    with gr.Tab("🎤 Clinical Query"):
        gr.Markdown("### Ask About a Patient")
        with gr.Row():
            with gr.Column(scale=1):
                query_patient = gr.Dropdown(
                    choices=init_choices,
                    value=init_choices[0][1] if init_choices else None,
                    label="Select Patient"
                )
                refresh_btn = gr.Button("🔄 Refresh Patient List", size="sm")
                audio_input = gr.Audio(
                    type="filepath",
                    label="Doctor Query (speak or upload audio)"
                )
                query_btn = gr.Button("Run Clinical Query", variant="primary")

                gr.Markdown("---")
                gr.Markdown("**Or type your query:**")
                text_query = gr.Textbox(
                    label="Text Query (alternative to audio)",
                    placeholder="What are the lab results for this patient?"
                )
                text_btn = gr.Button("Submit Text Query")

            with gr.Column(scale=2):
                answer_output = gr.Textbox(
                    label="Clinical Response",
                    lines=12,
                    interactive=False
                )
                query_info = gr.Markdown("")
                sources_table = gr.DataFrame(
                    label="Sources Used",
                    headers=["index", "file", "page", "type", "preview"]
                )
                transcription_box = gr.Textbox(
                    label="Transcribed Query",
                    interactive=False
                )

    # ── Tab 3: System Info ─────────────────────────────────────────
    with gr.Tab("System Info"):
        registry = load_registry()
        gr.Markdown(f"""
        ### Current System Status
        - **Patients indexed:** {len(registry)}
        - **Mode:** {'Local (Ollama)' if os.getenv('USE_LOCAL_MODELS', 'true') == 'true' else 'Cloud (OpenRouter)'}
        - **Embedding model:** all-MiniLM-L6-v2
        - **Reranker:** ms-marco-MiniLM-L-12-v2
        """)

    # ── Wire up events ─────────────────────────────────────────────

    # Upload updates status + BOTH dropdowns
    upload_btn.click(
        fn=handle_upload,
        inputs=[pdf_upload, manual_id, replace_checkbox],
        outputs=[upload_status, mgmt_dropdown, query_patient]
    )

    # Refresh button on query tab
    refresh_btn.click(
        fn=refresh_patient_choices,
        inputs=[],
        outputs=[query_patient]
    )

    # Audio query
    query_btn.click(
        fn=handle_query,
        inputs=[audio_input, query_patient],
        outputs=[answer_output, query_info, sources_table, transcription_box]
    )

    # Text query
    def handle_text_query(text, patient_id):
        if not text.strip():
            return "No query provided.", "", pd.DataFrame(), ""
        result = run_pipeline_text(text, str(patient_id))
        sources_df = pd.DataFrame(result["sources"]) if result["sources"] else pd.DataFrame()
        return (
            result["answer"],
            f"**Query:** {text} | **Intent:** {result['intent']}",
            sources_df,
            text
        )

    text_btn.click(
        fn=handle_text_query,
        inputs=[text_query, query_patient],
        outputs=[answer_output, query_info, sources_table, transcription_box]
    )

if __name__ == "__main__":
    demo.launch(share=False)
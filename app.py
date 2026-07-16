import gradio as gr
import pandas as pd
import os
import base64
from pathlib import Path
from src.pipeline.graph import run_pipeline, run_pipeline_text, get_pipeline
from src.ingestion.patient_manager import (
    index_patient_pdf, get_patient_list,
    patient_exists, load_registry
)

# ── startup ──────────────────────────────────────────────────────────
print("Initializing Clinical Co-Pilot...")
pipeline, embedder, vector_stores = get_pipeline()
patient_store = vector_stores["patient_records"]
print("Ready.")

MODE = "Local" if os.getenv("USE_LOCAL_MODELS", "true").lower() == "true" else "Cloud (OpenRouter)"

# ── helpers ───────────────────────────────────────────────────────────
def _choices():
    patients = get_patient_list()
    if not patients:
        return []
    return [(p["label"], p["id"]) for p in patients]

def _make_dropdown(label="Select Patient"):
    choices = _choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None, label=label)

# ── handlers ──────────────────────────────────────────────────────────
def upload_patient(pdf_files):
    """Upload one or more patient PDFs. IDs are auto-assigned."""
    if not pdf_files:
        return (
            gr.update(value="⚠️ Please upload at least one PDF file."),
            _make_dropdown(),
            _make_dropdown()
        )

    lines = []
    for pdf_file in pdf_files:
        try:
            pid, msg = index_patient_pdf(
                pdf_path=pdf_file.name,
                patient_store=patient_store,
                embedder=embedder,
                patient_id=None,
                replace=False
            )
            lines.append(f"✅ Patient **{pid}** — {Path(pdf_file.name).name} indexed successfully ({msg.split(':')[-1].strip()})")
        except Exception as e:
            lines.append(f"❌ {Path(pdf_file.name).name} — failed: {e}")

    status = "\n\n".join(lines)
    return (
        gr.update(value=status),
        _make_dropdown(),
        _make_dropdown()
    )


def ask_audio(audio_path, patient_id):
    """Handle voice query."""
    if audio_path is None:
        return (
            "⚠️ Please record or upload an audio query.",
            "", pd.DataFrame(), ""
        )
    if not patient_id:
        return (
            "⚠️ Please select a patient first.",
            "", pd.DataFrame(), ""
        )
    return _run_query(audio_path=audio_path, text=None, patient_id=patient_id)


def ask_text(text, patient_id):
    """Handle typed query."""
    if not text or not text.strip():
        return (
            "⚠️ Please type a question.",
            "", pd.DataFrame(), ""
        )
    if not patient_id:
        return (
            "⚠️ Please select a patient first.",
            "", pd.DataFrame(), ""
        )
    return _run_query(audio_path=None, text=text.strip(), patient_id=patient_id)


def _run_query(audio_path, text, patient_id):
    """Shared logic for audio and text queries."""
    try:
        if audio_path:
            result = run_pipeline(audio_path=audio_path, patient_id=str(patient_id))
        else:
            result = run_pipeline_text(text_query=text, patient_id=str(patient_id))

        answer  = result["answer"]
        transcription = result["transcription"]
        intent  = result["intent"]
        sources = result["sources"]
        grounded = result["is_grounded"]

        intent_labels = {
            "patient_history": "🩺 Patient Records",
            "drug":            "💊 Drug Guidelines",
            "trial":           "🔬 Clinical Trials",
            "general":         "📚 All Sources"
        }
        intent_display = intent_labels.get(intent, intent)
        grounded_display = "✅ Answer verified against sources" if grounded else "⚠️ Answer could not be fully verified"

        info_md = (
            f"**Query:** {transcription}  \n"
            f"**Sources searched:** {intent_display}  \n"
            f"**Verification:** {grounded_display}"
        )

        sources_df = pd.DataFrame(sources)[["index", "type", "file", "page", "preview"]] if sources else pd.DataFrame()

        return answer, info_md, sources_df, transcription

    except Exception as e:
        return (
            f"❌ An error occurred: {e}\n\nPlease check that a patient is selected and try again.",
            "", pd.DataFrame(), ""
        )


def get_system_status():
    registry = load_registry()
    count = len(registry)
    lines = [
        f"**Patients in system:** {count}",
        f"**AI mode:** {MODE}",
        f"**Embedding:** all-MiniLM-L6-v2",
        f"**Reranker:** FlashRank ms-marco-MiniLM-L-12-v2",
        "",
    ]
    if registry:
        lines.append("**Indexed patients:**")
        for pid, info in sorted(registry.items()):
            lines.append(f"- Patient {pid} — {info['file_name']} ({info['chunk_count']} chunks)")
    else:
        lines.append("*No patients indexed yet. Go to 'Add Patient' to upload records.*")
    return "\n".join(lines)


# ── CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
body, .gradio-container { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1200px !important; margin: auto !important; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 0;
    margin-bottom: 16px;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-wrap img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    display: block;
    opacity: 0.6;
}
.hero-text {
    padding: 24px 32px 20px;
    text-align: center;
}
.hero-text h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.hero-text p {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
    line-height: 1.5;
}

/* ── Feature cards ── */
.features-row {
    display: flex;
    gap: 12px;
    padding: 0 32px 24px;
    justify-content: center;
}
.feat-card {
    flex: 1;
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.feat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.35);
}
.feat-icon { font-size: 28px; margin-bottom: 6px; }
.feat-title { color: #e2e8f0; font-weight: 600; font-size: 13px; margin: 4px 0 2px; }
.feat-desc  { color: #64748b; font-size: 11px; line-height: 1.4; }

/* ── Pipeline viz ── */
.pipeline-strip {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 12px 20px;
    background: rgba(15,23,42,0.5);
    border-radius: 12px;
    border: 1px solid rgba(56,189,248,0.08);
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.pip-step {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: #94a3b8;
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(100,116,139,0.15);
    white-space: nowrap;
}
.pip-arrow { color: #475569; font-size: 16px; margin: 0 2px; }

/* ── Status badges ── */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.loading-bar {
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.5), transparent);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    height: 3px;
    border-radius: 4px;
    margin-top: 6px;
}
.status-box {
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.3s ease;
}
.status-idle    { background: rgba(100,116,139,0.06); border: 1px solid rgba(100,116,139,0.12); color: #94a3b8; }
.status-loading { background: rgba(56,189,248,0.06); border: 1px solid rgba(56,189,248,0.2); color: #38bdf8; }
.status-done    { background: rgba(52,211,153,0.06); border: 1px solid rgba(52,211,153,0.2); color: #34d399; }
.status-error   { background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.2); color: #f87171; }

/* ── Section labels ── */
.section-label {
    font-size: 13px;
    font-weight: 600;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 8px 0 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
}

/* ── Misc polish ── */
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
}
"""

# ── HTML builders ─────────────────────────────────────────────────────
init_choices = _choices()

# Load hero banner as base64 for inline embedding
_hero_b64 = ""
try:
    with open("static/hero_banner.png", "rb") as _f:
        _hero_b64 = base64.b64encode(_f.read()).decode()
except FileNotFoundError:
    pass

HERO_HTML = f"""
<div class="hero-wrap">
    <img src="data:image/png;base64,{_hero_b64}" alt="Clinical AI" />
    <div class="hero-text">
        <h1>🏥 Clinical Intelligence & Diagnostics Co-Pilot</h1>
        <p>AI-powered multimodal medical assistant — reads patient PDFs (labs, X-rays, ECGs),<br>
        answers clinical questions via voice or text, with source-grounded verification.</p>
    </div>
    <div class="features-row">
        <div class="feat-card">
            <div class="feat-icon">📄</div>
            <div class="feat-title">Smart PDF Parsing</div>
            <div class="feat-desc">Docling-powered layout extraction of text, tables & medical images</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Multimodal RAG</div>
            <div class="feat-desc">Hybrid retrieval with BM25 + vector search, FlashRank reranking</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎙️</div>
            <div class="feat-title">Voice & Text</div>
            <div class="feat-desc">Whisper transcription with semantic intent routing</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">✅</div>
            <div class="feat-title">Verified Answers</div>
            <div class="feat-desc">Hallucination guard with numeric grounding checks</div>
        </div>
    </div>
</div>
"""

PIPELINE_HTML = """
<div class="pipeline-strip">
    <div class="pip-step">🎙️ Transcribe</div><span class="pip-arrow">→</span>
    <div class="pip-step">🧭 Classify Intent</div><span class="pip-arrow">→</span>
    <div class="pip-step">🔍 Retrieve</div><span class="pip-arrow">→</span>
    <div class="pip-step">🧠 Generate</div><span class="pip-arrow">→</span>
    <div class="pip-step">✅ Verify</div>
</div>
"""

def _status_html(msg, status="idle"):
    icons = {"idle": "💤", "loading": "⏳", "done": "✅", "error": "❌"}
    icon = icons.get(status, "")
    bar = '<div class="loading-bar"></div>' if status == "loading" else ""
    return f'<div class="status-box status-{status}">{icon} {msg}{bar}</div>'


# ── BUILD UI ──────────────────────────────────────────────────────────
with gr.Blocks(title="Clinical Co-Pilot") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tab("➕ Add Patient"):
        gr.HTML('<div class="section-label">Upload Patient Records</div>')
        gr.Markdown("Upload a patient PDF containing clinical notes, lab results, and imaging reports. A unique patient ID is assigned automatically.")

        pdf_upload = gr.File(
            label="Patient PDF files",
            file_types=[".pdf"],
            file_count="multiple"
        )
        upload_btn = gr.Button("📥  Add Patient(s) to System", variant="primary", size="lg")

        upload_loading = gr.HTML(value=_status_html("Upload a PDF to begin.", "idle"))
        upload_status = gr.Markdown(value="")

        gr.HTML('<div class="section-label">Patients in System</div>')
        mgmt_dropdown = gr.Dropdown(
            choices=init_choices,
            value=init_choices[0][1] if init_choices else None,
            label="Indexed patients",
            interactive=False
        )

    with gr.Tab("🔬 Clinical Query"):
        gr.HTML(PIPELINE_HTML)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="section-label">Patient & Input</div>')
                query_patient = gr.Dropdown(
                    choices=init_choices,
                    value=init_choices[0][1] if init_choices else None,
                    label="👤 Select Patient",
                    interactive=True
                )

                audio_input = gr.Audio(type="filepath", label="🎙️ Voice Query")
                audio_btn = gr.Button("🔍 Analyse Voice Query", variant="primary")

                text_input = gr.Textbox(
                    placeholder="e.g. What are the lab results? Is there anything concerning in the chest X-ray?",
                    label="⌨️ Text Query",
                    lines=2
                )
                text_btn = gr.Button("🔍 Analyse Text Query", variant="primary")

            with gr.Column(scale=2):
                query_loading = gr.HTML(value=_status_html("Submit a query to get started.", "idle"))

                gr.HTML('<div class="section-label">Clinical Response</div>')
                answer_box = gr.Textbox(
                    label="",
                    lines=12,
                    interactive=False,
                    placeholder="The AI response will appear here..."
                )
                query_info = gr.Markdown("")

                gr.HTML('<div class="section-label">Sources Referenced</div>')
                sources_table = gr.DataFrame(label="", wrap=True)
                transcription_box = gr.Textbox(label="📝 Transcribed query", interactive=False)

    with gr.Tab("⚙️ System Status"):
        gr.HTML('<div class="section-label">System Overview</div>')
        status_display = gr.Markdown(value=get_system_status())
        refresh_status_btn = gr.Button("🔄 Refresh Status")
        refresh_status_btn.click(fn=get_system_status, outputs=status_display)

    # ── events ──────────────────────────────────────────────────────
    upload_btn.click(
        fn=lambda: _status_html("Processing PDF — extracting text, tables & images…", "loading"),
        inputs=None, outputs=upload_loading
    ).then(
        fn=upload_patient,
        inputs=[pdf_upload],
        outputs=[upload_status, mgmt_dropdown, query_patient]
    ).then(
        fn=lambda: _status_html("Upload complete!", "done"),
        inputs=None, outputs=upload_loading
    )

    audio_btn.click(
        fn=lambda: (_status_html("Analysing… Transcribing → Classifying → Retrieving → Generating…", "loading"), "", "", pd.DataFrame(), ""),
        inputs=None, outputs=[query_loading, answer_box, query_info, sources_table, transcription_box]
    ).then(
        fn=ask_audio,
        inputs=[audio_input, query_patient],
        outputs=[answer_box, query_info, sources_table, transcription_box]
    ).then(
        fn=lambda: _status_html("Analysis complete.", "done"),
        inputs=None, outputs=query_loading
    )

    text_btn.click(
        fn=lambda: (_status_html("Analysing… Classifying → Retrieving → Generating…", "loading"), "", "", pd.DataFrame(), ""),
        inputs=None, outputs=[query_loading, answer_box, query_info, sources_table, transcription_box]
    ).then(
        fn=ask_text,
        inputs=[text_input, query_patient],
        outputs=[answer_box, query_info, sources_table, transcription_box]
    ).then(
        fn=lambda: _status_html("Analysis complete.", "done"),
        inputs=None, outputs=query_loading
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS
    )
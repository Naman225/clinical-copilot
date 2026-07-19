import gradio as gr
import pandas as pd
import os
import html
from pathlib import Path
from src.pipeline.graph import run_pipeline, run_pipeline_text, get_pipeline
from src.utils.config import get_store_count
from src.ingestion.patient_manager import (
    index_patient_pdf, get_patient_list, load_registry
)
from src.ingestion.reference_loader import load_reference_path
from langchain_community.vectorstores.utils import filter_complex_metadata

# Patch gradio_client boolean schema bug in get_api_info
try:
    import gradio_client.utils as client_utils
    _orig_json_schema = client_utils._json_schema_to_python_type
    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_json_schema(schema, defs)
    client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass

try:
    import spaces
except ImportError:
    class _SpacesMock:
        def GPU(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda func: func
    spaces = _SpacesMock()


# ── startup ──────────────────────────────────────────────────────────
print("Initializing Clinical Co-Pilot...")
pipeline, embedder, vector_stores = get_pipeline()
patient_store = vector_stores["patient_records"]
pharma_store  = vector_stores["pharma_guidelines"]
trial_store   = vector_stores["clinical_trials"]
print("Ready.")

MODE = "Local (Ollama)" if os.getenv("USE_LOCAL_MODELS", "true").lower() == "true" else "Cloud (OpenRouter)"

# ── display helpers ───────────────────────────────────────────────────
INTENT_LABELS = {
    "patient_history": "Patient records only",
    "drug":            "Drug guidelines + patient records",
    "trial":           "Clinical trials + patient records",
    "general":         "All available sources",
}

TYPE_LABELS = {
    "text":          "Clinical note",
    "table":         "Lab / table",
    "medical_image": "Imaging (X-ray, ECG, etc.)",
}

EXAMPLE_QUESTIONS = [
    "What are this patient's most recent lab results?",
    "Summarize the chest X-ray findings.",
    "Are there any abnormal ECG findings?",
    "What medications are mentioned in the record?",
    "Is there evidence of infection in the imaging?",
]

def _choices():
    patients = get_patient_list()
    return [(p["label"], p["id"]) for p in patients] if patients else []

def _refresh_patient_dropdown():
    ch = _choices()
    val = ch[0][1] if ch else None
    return gr.update(choices=ch, value=val)

def _patient_roster_df():
    registry = load_registry()
    if not registry:
        return pd.DataFrame(columns=["Patient ID", "File", "Sections indexed"])
    rows = []
    for pid, info in sorted(registry.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        rows.append({
            "Patient ID": pid,
            "File": info.get("file_name", "—"),
            "Sections indexed": info.get("chunk_count", 0),
        })
    return pd.DataFrame(rows)

def _stats_html():
    registry = load_registry()
    return f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="num">👤 {len(registry)}</div>
        <div class="lbl">Patients</div>
      </div>
      <div class="stat-card">
        <div class="num">💊 {get_store_count(pharma_store)}</div>
        <div class="lbl">Drug guidelines</div>
      </div>
      <div class="stat-card">
        <div class="num">🔬 {get_store_count(trial_store)}</div>
        <div class="lbl">Trial documents</div>
      </div>
    </div>
    """

def _status_banner(kind: str, title: str, detail: str = "") -> str:
    styles = {
        "success": ("#064e3b", "#059669", "#ecfdf5", "#10b981", "✓"),
        "warning": ("#78350f", "#d97706", "#fffbeb", "#f59e0b", "!"),
        "error":   ("#7f1d1d", "#dc2626", "#fef2f2", "#ef4444", "✕"),
        "info":    ("#1e293b", "#38bdf8", "#f8fafc", "#0284c7", "ℹ"),
    }
    bg, border, text, icon_bg, icon = styles.get(kind, styles["info"])
    detail_html = (
        f'<div style="font-size:0.86rem;color:#cbd5e1;margin-top:4px;line-height:1.5;">'
        f'{html.escape(detail)}</div>'
    ) if detail else ""
    return f"""
    <div style="display:flex;gap:14px;align-items:flex-start;
                background:{bg};border:1px solid {border};border-radius:10px;
                padding:16px 18px;margin-bottom:10px;color:{text};box-shadow:0 4px 6px -1px rgba(0,0,0,0.15);">
      <span style="background:{icon_bg};color:white;width:26px;height:26px;
                   border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
                   font-size:0.85rem;font-weight:700;flex-shrink:0;">{icon}</span>
      <div>
        <div style="font-weight:600;font-size:0.98rem;color:{text};">{html.escape(title)}</div>
        {detail_html}
      </div>
    </div>
    """

def _upload_result_card(kind: str, title: str, subtitle: str = "") -> str:
    styles = {
        "success": ("#064e3b", "#34d399", "✓"),
        "warning": ("#78350f", "#fbbf24", "!"),
        "error":   ("#7f1d1d", "#f87171", "✕"),
    }
    bg, text, icon = styles.get(kind, styles["success"])
    sub = (
        f'<div style="font-size:0.83rem;color:#cbd5e1;margin-top:2px;">{html.escape(subtitle)}</div>'
        if subtitle else ""
    )
    return f"""
    <div style="display:flex;gap:12px;background:{bg};border:1px solid rgba(255,255,255,0.15);border-radius:8px;
                padding:12px 16px;margin-bottom:8px;color:{text};">
      <span style="font-weight:700;font-size:1.1rem;">{icon}</span>
      <div>
        <div style="font-weight:600;color:#f8fafc;">{html.escape(title)}</div>
        {sub}
      </div>
    </div>
    """

def _build_upload_report(results: list[dict]) -> str:
    if not results:
        return _status_banner("info", "No files processed.")

    ok   = sum(1 for r in results if r["kind"] == "success")
    warn = sum(1 for r in results if r["kind"] == "warning")
    fail = sum(1 for r in results if r["kind"] == "error")
    total = len(results)

    if fail == 0 and warn == 0:
        summary_kind, summary_title = "success", f"All {total} file{'s' if total != 1 else ''} uploaded successfully"
    elif ok == 0:
        summary_kind, summary_title = "error", f"Upload failed for {fail + warn} of {total} file{'s' if total != 1 else ''}"
    else:
        summary_kind = "warning"
        summary_title = f"{ok} succeeded, {fail + warn} need attention ({total} total)"

    parts = [_status_banner(summary_kind, summary_title)]
    parts.extend(_upload_result_card(r["kind"], r["title"], r.get("subtitle", "")) for r in results)
    return "".join(parts)

def _file_path(f) -> str:
    return f.name if hasattr(f, "name") else str(f)

def _patient_loading():
    return """
    <div style="display:flex;gap:16px;align-items:center;background:#1e293b;border:1px solid #38bdf8;border-radius:10px;padding:18px 20px;color:#f8fafc;box-shadow:0 4px 6px -1px rgba(0,0,0,0.2);">
      <div style="width:28px;height:28px;border:3px solid rgba(56,189,248,0.2);border-top-color:#38bdf8;border-radius:50%;animation:spin 1s linear infinite;flex-shrink:0;"></div>
      <div>
        <div style="font-weight:600;font-size:1rem;color:#38bdf8;">⏳ Processing & Indexing Patient Records...</div>
        <div style="font-size:0.86rem;color:#cbd5e1;margin-top:4px;">Parsing notes, tables, and medical imaging. This may take a minute.</div>
      </div>
    </div>
    """

def _ref_loading():
    return """
    <div style="display:flex;gap:16px;align-items:center;background:#1e293b;border:1px solid #38bdf8;border-radius:10px;padding:18px 20px;color:#f8fafc;box-shadow:0 4px 6px -1px rgba(0,0,0,0.2);">
      <div style="width:28px;height:28px;border:3px solid rgba(56,189,248,0.2);border-top-color:#38bdf8;border-radius:50%;animation:spin 1s linear infinite;flex-shrink:0;"></div>
      <div>
        <div style="font-weight:600;font-size:1rem;color:#38bdf8;">⏳ Adding Documents to Knowledge Base...</div>
        <div style="font-size:0.86rem;color:#cbd5e1;margin-top:4px;">Extracting content and updating vector search index. Please wait.</div>
      </div>
    </div>
    """

# ── patient upload ────────────────────────────────────────────────────
@spaces.GPU(duration=25)
def upload_patient(pdf_files):
    if not pdf_files:
        return (
            _status_banner("warning", "No files selected", "Choose at least one patient PDF to upload."),
            _patient_roster_df(),
            _refresh_patient_dropdown(),
            _stats_html(),
        )

    results = []
    for f in pdf_files:
        path_str = _file_path(f)
        fname = Path(path_str).name
        try:
            pid, msg = index_patient_pdf(
                pdf_path=path_str,
                patient_store=patient_store,
                embedder=embedder,
                patient_id=None,
                replace=False,
            )
            registry = load_registry()
            sections = registry.get(pid, {}).get("chunk_count", "?")

            if "already exists" in msg.lower():
                results.append({"kind": "warning", "title": fname, "subtitle": msg})
            else:
                results.append({
                    "kind": "success",
                    "title": f"Patient {pid} added",
                    "subtitle": f"{fname} · {sections} sections indexed",
                })
        except Exception as e:
            results.append({"kind": "error", "title": fname, "subtitle": str(e)})

    return (
        _build_upload_report(results),
        _patient_roster_df(),
        _refresh_patient_dropdown(),
        _stats_html(),
    )

# ── reference upload ──────────────────────────────────────────────────
@spaces.GPU(duration=25)
def upload_reference(pdf_files, ref_type):
    stats = _stats_html()
    if not pdf_files:
        return _status_banner("warning", "No files selected", "Choose one or more PDF files."), stats

    domain = "pharma_guideline" if ref_type == "Drug Guidelines" else "clinical_trial"
    store  = pharma_store if domain == "pharma_guideline" else trial_store
    paths  = [Path(_file_path(f)) for f in pdf_files]
    label  = "Drug guidelines" if domain == "pharma_guideline" else "Clinical trials"

    results = []
    try:
        docs = load_reference_path(paths, domain)
        docs = filter_complex_metadata(docs)
        store.add_documents(docs)
        for p in paths:
            results.append({
                "kind": "success",
                "title": p.name,
                "subtitle": f"Added to {label.lower()}",
            })
        report = _build_upload_report(results)
        report += (
            f'<div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;padding-left:4px;">'
            f'{len(docs)} searchable sections added in total.</div>'
        )
        return report, _stats_html()
    except Exception as e:
        return _status_banner("error", f"Could not add {label.lower()}", str(e)), stats

# ── query ─────────────────────────────────────────────────────────────
def _format_sources(sources: list) -> pd.DataFrame:
    if not sources:
        return pd.DataFrame(columns=["#", "Type", "Document", "Page", "Excerpt"])
    rows = []
    for s in sources:
        rows.append({
            "#": s.get("index", ""),
            "Type": TYPE_LABELS.get(s.get("type", ""), s.get("type", "—")),
            "Document": s.get("file", "—"),
            "Page": s.get("page", "—"),
            "Excerpt": s.get("preview", ""),
        })
    return pd.DataFrame(rows)

def _format_query_meta(result: dict) -> str:
    intent = INTENT_LABELS.get(result["intent"], result["intent"])
    if result["is_grounded"]:
        badge_style = "background:#ecfdf5;color:#059669;border:1px solid #a7f3d0;"
        badge_text = "✓ Verified against source documents"
    else:
        badge_style = "background:#fffbeb;color:#d97706;border:1px solid #fde68a;"
        badge_text = "⚠ Review recommended — not fully verified"

    heard = html.escape(result.get("transcription") or "")
    return f"""
    <div style="margin-bottom:12px;">
      <span style="display:inline-block;padding:6px 14px;border-radius:999px;
                   font-size:0.8rem;font-weight:600;{badge_style}">{badge_text}</span>
      <div style="margin-top:10px;font-size:0.88rem;color:#cbd5e1;">
        <strong style="color:#94a3b8;">Question:</strong> {heard}
      </div>
      <div style="font-size:0.88rem;color:#cbd5e1;margin-top:4px;">
        <strong style="color:#94a3b8;">Sources searched:</strong> {html.escape(intent)}
      </div>
    </div>
    """

@spaces.GPU(duration=25)
def _run_voice_pipeline(audio_path, patient_id):
    return run_pipeline(audio_path=audio_path, patient_id=str(patient_id))

def _run(audio_path, text, patient_id):
    if not patient_id:
        return (
            "Please select a patient from the dropdown above.",
            _status_banner("warning", "No patient selected", "Upload a patient record first, then choose them here."),
            pd.DataFrame(),
            "",
        )
    try:
        if audio_path:
            result = _run_voice_pipeline(audio_path=audio_path, patient_id=patient_id)
        else:
            result = run_pipeline_text(text_query=text, patient_id=str(patient_id))

        return (
            result["answer"],
            _format_query_meta(result),
            _format_sources(result["sources"]),
            result.get("transcription") or "",
        )
    except Exception as e:
        return (
            f"Something went wrong while processing your question.\n\n{e}",
            _status_banner("error", "Query failed", str(e)),
            pd.DataFrame(),
            "",
        )

def system_status():
    registry = load_registry()
    lines = [
        "### System overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Patients indexed** | {len(registry)} |",
        f"| **AI mode** | {MODE} |",
        f"| **Embedding model** | {'embed-v4.0' if 'Cloud' in MODE else 'all-MiniLM-L6-v2'} |",
        f"| **Reranker** | {'rerank-v3.5' if 'Cloud' in MODE else 'ms-marco-MiniLM-L-12-v2'} |",
        f"| **Drug guideline sections** | {get_store_count(pharma_store)} |",
        f"| **Clinical trial sections** | {get_store_count(trial_store)} |",
        "",
        "> Answers are generated from indexed documents only. Always use clinical judgment.",
    ]
    if registry:
        lines += ["", "### Patient registry", ""]
        for pid, info in sorted(registry.items()):
            lines.append(f"- **Patient {pid}** — {info['file_name']} ({info['chunk_count']} sections)")
    else:
        lines += ["", "*No patients uploaded yet. Go to the **Patients** tab to add records.*"]
    return "\n".join(lines)

# ── CSS ───────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --blue-900: #0c4a6e;
    --blue-700: #0369a1;
    --radius: 12px;
}

.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

/* header */
.app-header {
    background: linear-gradient(135deg, #0c4a6e 0%, #075985 50%, #164e63 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 16px;
    display: flex;
    gap: 20px;
    align-items: center;
}
.app-header h1 { font-size: 1.65rem; font-weight: 700; margin: 0 0 6px; }
.app-header p  { margin: 0; opacity: 0.9; line-height: 1.55; font-size: 0.95rem; }
.header-icon { font-size: 2.8rem; line-height: 1; flex-shrink: 0; }
.trust-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.trust-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* stats */
.stats-row { display: flex; gap: 14px; margin: 0 0 18px; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 140px;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: var(--radius);
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.25);
}
.stat-card .num { font-size: 1.8rem; font-weight: 700; color: #38bdf8 !important; line-height: 1; }
.stat-card .lbl {
    font-size: 0.74rem; color: #cbd5e1 !important; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 6px; font-weight: 600;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* workflow hint */
.workflow-hint {
    background: #1e3a5f;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #e2e8f0;
    margin-bottom: 12px;
    line-height: 1.5;
}

/* upload hints — dark-theme friendly */
.upload-hint-dark {
    background: #1e293b;
    border: 1px dashed #475569;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.55;
    margin-bottom: 8px;
}

/* tighten upload column spacing */
.upload-col .block { padding: 4px 8px !important; gap: 4px !important; }
.upload-col .wrap { gap: 6px !important; }
.upload-col [data-testid="file-upload"] { min-height: unset !important; }
.upload-col .primary-btn { margin-top: 2px !important; }

/* status boxes — prevent collapse */
#patient-status-box, #ref-status-box { min-height: 56px; }

/* answer panel */
#answer-box textarea {
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

/* primary buttons */
.primary-btn button {
    background: var(--blue-900) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
}

.tab-nav button.selected {
    font-weight: 700 !important;
    color: var(--blue-700) !important;
}
"""

# ── BUILD UI ──────────────────────────────────────────────────────────
init_choices = _choices()
init_val     = init_choices[0][1] if init_choices else None
has_patients = bool(init_choices)

READY_STATUS = _status_banner("info", "Ready to upload", "Select PDF files, then click the upload button.")

with gr.Blocks(title="Clinical Co-Pilot") as demo:

    gr.HTML("""
    <div class="app-header">
      <div class="header-icon">🏥</div>
      <div style="flex:1;">
        <h1>Clinical Co-Pilot</h1>
        <p>Ask questions about your patients by voice or text.
           Answers are drawn from uploaded records, drug guidelines, and trial data — with source citations.</p>
        <div class="trust-badges">
          <span class="trust-badge">🔒 Patient-isolated</span>
          <span class="trust-badge">📎 Source-cited</span>
          <span class="trust-badge">✅ Verification check</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.22);padding:12px 18px;border-radius:14px;flex-shrink:0;">
        <span style="font-size:2.2rem;">⚡🩺</span>
        <div>
          <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:#7dd3fc;font-weight:700;">AI Engine</div>
          <div style="font-size:0.95rem;font-weight:600;color:white;">Active & Grounded</div>
        </div>
      </div>
    </div>
    """)

    stats_box = gr.HTML(value=_stats_html())

    if not has_patients:
        gr.HTML("""
        <div class="workflow-hint">
          <strong>Getting started:</strong>
          1) Go to <b>Patients</b> and upload a clinical PDF →
          2) Return here and select the patient →
          3) Ask your question by voice or text.
        </div>
        """)

    with gr.Tab("Consult"):
        with gr.Row():
            with gr.Column(scale=5):
                patient_dd = gr.Dropdown(
                    choices=init_choices,
                    value=init_val,
                    label="Select patient",
                    info="Each patient's data is kept separate",
                )

                gr.Markdown("#### Ask your question")
                text_in = gr.Textbox(
                    placeholder="e.g. What do the lab results show? Is the chest X-ray concerning?",
                    label="Type a question",
                    lines=3,
                )
                with gr.Row():
                    text_btn  = gr.Button("Get answer", variant="primary", elem_classes=["primary-btn"], scale=2)
                    voice_btn = gr.Button("Use voice instead", scale=1)

                with gr.Accordion("🎙️ Voice query", open=False, elem_id="voice-accordion") as voice_accordion:
                    audio_in = gr.Audio(type="filepath", label="Record or upload audio")
                    voice_submit = gr.Button("Analyse voice", variant="primary", elem_classes=["primary-btn"])

                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=text_in,
                    label="Example questions (click to fill)",
                )

            with gr.Column(scale=7):
                query_meta = gr.HTML("")
                answer_out = gr.Textbox(
                    label="Clinical answer",
                    lines=14,
                    interactive=False,
                    placeholder="Your answer will appear here with citations…",
                    elem_id="answer-box",
                )

                with gr.Group(visible=False) as sources_section:
                    gr.HTML('<div style="text-align: center; font-size: 1.15rem; font-weight: 700; margin: 18px 0 10px; color: #f8fafc;">📎 Sources referenced</div>')
                    sources_tbl = gr.DataFrame(
                        wrap=True,
                        column_widths=["6%", "16%", "24%", "8%", "46%"],
                    )

                transcript_box = gr.Textbox(
                    label="Transcribed question (voice only)",
                    interactive=False,
                    lines=1,
                    visible=False,
                )

    with gr.Tab("Patients"):
        with gr.Row():
            with gr.Column(elem_classes=["upload-col"]):
                gr.Markdown("### Upload patient records")
                gr.HTML("""
                <div class="upload-hint-dark">
                  📄 <strong>Patient PDF</strong> — notes, labs, X-ray &amp; ECG in one file<br>
                  🆔 A unique Patient ID is assigned automatically<br>
                  📁 Upload several patients at once
                </div>
                """)
                patient_pdf = gr.File(
                    label="Patient PDF files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                patient_upload_btn = gr.Button(
                    "📥 Upload patient record(s)",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

            with gr.Column():
                gr.Markdown("### Upload result")
                patient_status = gr.HTML(value=READY_STATUS, elem_id="patient-status-box")
                gr.Markdown("### Current patients")
                patient_roster = gr.DataFrame(
                    value=_patient_roster_df(),
                    interactive=False,
                    wrap=True,
                )

    with gr.Tab("Guidelines & Trials"):
        gr.Markdown(
            "Add drug prescribing information or clinical trial PDFs. "
            "These are searched automatically when you ask medication or evidence-based questions."
        )
        with gr.Row():
            with gr.Column(elem_classes=["upload-col"]):
                ref_type = gr.Radio(
                    choices=["Drug Guidelines", "Clinical Trials"],
                    value="Drug Guidelines",
                    label="Document type",
                )
                gr.HTML("""
                <div class="upload-hint-dark">
                  💊 <strong>Drug Guidelines</strong> — labels, dosing sheets, prescribing info<br>
                  🔬 <strong>Clinical Trials</strong> — research papers, outcomes, evidence summaries
                </div>
                """)
                ref_pdf = gr.File(
                    label="PDF files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                ref_upload_btn = gr.Button(
                    "📥 Add to knowledge base",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

            with gr.Column():
                gr.Markdown("### Upload result")
                ref_status = gr.HTML(value=READY_STATUS, elem_id="ref-status-box")

    with gr.Tab("About"):
        status_md = gr.Markdown(value=system_status())
        gr.Button("🔄 Refresh").click(fn=system_status, outputs=status_md)

    # ── events ────────────────────────────────────────────────────────
    QUERY_OUTPUTS = [answer_out, query_meta, sources_tbl, transcript_box, transcript_box, sources_section]

    def _ask_and_show_transcript(audio, text, patient_id):
        ans, meta, src, trans = _run(audio, text, patient_id)
        has_sources = isinstance(src, pd.DataFrame) and len(src) > 0
        return (
            ans,
            meta,
            src,
            trans,
            gr.update(value=trans, visible=bool(trans)),
            gr.update(visible=has_sources),
        )

    text_btn.click(
        fn=lambda t, p: _ask_and_show_transcript(None, t, p),
        inputs=[text_in, patient_dd],
        outputs=QUERY_OUTPUTS,
        show_progress="minimal",
        api_name=False,
    )
    text_in.submit(
        fn=lambda t, p: _ask_and_show_transcript(None, t, p),
        inputs=[text_in, patient_dd],
        outputs=QUERY_OUTPUTS,
        show_progress="minimal",
        api_name=False,
    )
    voice_submit.click(
        fn=lambda a, p: _ask_and_show_transcript(a, None, p),
        inputs=[audio_in, patient_dd],
        outputs=QUERY_OUTPUTS,
        show_progress="minimal",
        api_name=False,
    )
    voice_btn.click(
        fn=lambda: gr.Accordion(open=True),
        inputs=None,
        outputs=[voice_accordion],
        js="() => { const acc = document.querySelector('#voice-accordion'); if (acc) { acc.open = true; const summary = acc.querySelector('summary'); if (summary && !acc.hasAttribute('open')) summary.click(); } }",
        api_name=False,
    )

    patient_upload_btn.click(
        fn=_patient_loading,
        inputs=None,
        outputs=patient_status,
        show_progress="hidden",
        api_name=False,
    ).then(
        fn=upload_patient,
        inputs=[patient_pdf],
        outputs=[patient_status, patient_roster, patient_dd, stats_box],
        show_progress="minimal",
        api_name=False,
    )

    ref_upload_btn.click(
        fn=_ref_loading,
        inputs=None,
        outputs=ref_status,
        show_progress="hidden",
        api_name=False,
    ).then(
        fn=upload_reference,
        inputs=[ref_pdf, ref_type],
        outputs=[ref_status, stats_box],
        show_progress="minimal",
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="sky"), css=CSS)
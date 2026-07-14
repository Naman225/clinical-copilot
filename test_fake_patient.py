# test_fake_patient.py — Test with a patient ID that does NOT exist in the database
from src.pipeline.graph import run_pipeline

result = run_pipeline(
    audio_path="./data_ingestion/test_audio/query_1.mp3",
    patient_id="patient_99999999"  # This ID does NOT exist
)

print("\nTranscription:", result["transcription"])
print("Intent:", result["intent"])
print("Grounded:", result["is_grounded"])
print("\nAnswer:\n", result["answer"])
print("\nSources:")
if result["sources"]:
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['file']} p.{s['page']} — {s['preview'][:60]}")
else:
    print("  (no sources — patient not found)")

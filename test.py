# test_pipeline.py
from src.pipeline.graph import run_pipeline

result = run_pipeline(
    audio_path="./data_ingestion/test_audio/query_1.mp3",
    patient_id="patient_10001217"
)

print("\nTranscription:", result["transcription"])
print("Intent:", result["intent"])
print("Grounded:", result["is_grounded"])
print("\nAnswer:\n", result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"  [{s['index']}] {s['file']} p.{s['page']} — {s['preview'][:60]}")
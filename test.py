# test_transcriber.py
from src.pipeline.transcriber import MedicalTranscriber

transcriber = MedicalTranscriber(model_size="base")
result = transcriber.transcribe("./data_ingestion/test_audio/query_1.mp3")
print(f"Transcribed: {result['text']}")
import whisper
from pathlib import Path

class MedicalTranscriber:
    def __init__(self, model_size : str = "base"):
        print(f"Loading Whisper {model_size}...")
        self.model = whisper.load_model(model_size)
        print(f"Whisper Ready!")
    def transcribe(self, audio_path: str) -> dict:
        result = self.model.transcribe(
            audio_path,
            language = "en",
            fp16= False,
            verbose = False
        )
        return {
            "text" : result["text"].strip(),
            "language" : result["language"]
        }


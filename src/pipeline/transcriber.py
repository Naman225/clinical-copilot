from pathlib import Path

class GroqWhisperTranscriber:
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
 
    def transcribe(self, audio_path: str) -> dict:
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                file=f,
                model=self.model,
                language="en",
                response_format="verbose_json"
            )
        return {
            "text": result.text.strip(),
            "language": getattr(result, "language", "en")
        }
        
class MedicalTranscriber:
    def __init__(self, model_size: str = "base"):
        import whisper
        print(f"Loading Whisper {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Whisper ready!")
 
    def transcribe(self, audio_path: str) -> dict:
        result = self.model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            verbose=False
        )
        return {
            "text": result["text"].strip(),
            "language": result["language"]
        }

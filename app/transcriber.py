import os
import whisper

_model = None


def _get_model():
    global _model
    if _model is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        _model = whisper.load_model(model_name)
    return _model


def transcribe(file_path: str) -> str:
    """Transcribe an audio file and return the text."""
    model = _get_model()
    result = model.transcribe(file_path)
    return result["text"].strip()

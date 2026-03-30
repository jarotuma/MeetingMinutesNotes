import asyncio
import os
import whisper

_model = None


def _load_model():
    global _model
    if _model is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        _model = whisper.load_model(model_name)
    return _model


def _run_transcribe(file_path: str) -> str:
    model = _load_model()
    result = model.transcribe(file_path)
    return result["text"].strip()


async def transcribe(file_path: str) -> str:
    """Transcribe an audio file asynchronously.

    Whisper runs CPU-bound inference, so we offload it to a thread pool
    to avoid blocking the event loop during long recordings.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_transcribe, file_path)

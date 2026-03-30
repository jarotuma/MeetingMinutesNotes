import os
from groq import AsyncGroq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def transcribe(file_path: str) -> str:
    """Transcribe an audio file using Groq Whisper API."""
    client = _get_client()
    with open(file_path, "rb") as f:
        transcription = await client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="text",
        )
    return transcription.strip() if isinstance(transcription, str) else str(transcription)

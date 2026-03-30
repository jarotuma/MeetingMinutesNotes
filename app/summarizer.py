import json
import os
from groq import AsyncGroq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT = """Jsi asistent pro analýzu zápisů ze schůzek. \
Na základě přepisu schůzky vytvoř strukturované shrnutí v češtině.

Vrať POUZE validní JSON objekt s těmito klíči (bez markdown code bloků):
- "klic_body": seznam (array) hlavních témat a bodů schůzky (strings)
- "ukoly": seznam úkolů, každý jako objekt {"co": "...", "kdo": "..."} \
  (kdo = "neuvedeno" pokud není jasné)
- "rozhodnuti": seznam přijatých rozhodnutí (strings)"""


async def summarize(transcript: str) -> dict:
    """Summarize a meeting transcript using Groq API. Returns structured dict."""
    client = _get_client()
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Přepis schůzky:\n\n{transcript}"},
        ],
    )
    return json.loads(response.choices[0].message.content)

import json
import os
import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


SYSTEM_PROMPT = """Jsi asistent pro analýzu zápisů ze schůzek. \
Na základě přepisu schůzky vytvoř strukturované shrnutí v češtině.

Vrať POUZE validní JSON objekt s těmito klíči (bez markdown code bloků):
- "klic_body": seznam (array) hlavních témat a bodů schůzky (strings)
- "ukoly": seznam úkolů, každý jako objekt {"co": "...", "kdo": "..."} \
  (kdo = "neuvedeno" pokud není jasné)
- "rozhodnuti": seznam přijatých rozhodnutí (strings)"""


def summarize(transcript: str) -> dict:
    """Summarize a meeting transcript using Claude API. Returns structured dict."""
    client = _get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Přepis schůzky:\n\n{transcript}",
            }
        ],
    )
    raw = message.content[0].text.strip()
    return json.loads(raw)

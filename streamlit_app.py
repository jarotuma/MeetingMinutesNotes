import io
import json
import os

import streamlit as st
from groq import Groq
from pydub import AudioSegment

st.set_page_config(page_title="Meeting Minutes Notes", page_icon="🎙️", layout="centered")

st.title("🎙️ Meeting Minutes Notes")
st.write("Nahraj nahrávku ze schůzky – dostaneš přepis a strukturované shrnutí.")

SYSTEM_PROMPT = """Jsi asistent pro analýzu zápisů ze schůzek. \
Na základě přepisu schůzky vytvoř strukturované shrnutí v češtině.

Vrať POUZE validní JSON objekt s těmito klíči (bez markdown code bloků):
- "klic_body": seznam (array) hlavních témat a bodů schůzky (strings)
- "ukoly": seznam úkolů, každý jako objekt {"co": "...", "kdo": "..."} \
  (kdo = "neuvedeno" pokud není jasné)
- "rozhodnuti": seznam přijatých rozhodnutí (strings)"""

GROQ_LIMIT_BYTES = 24 * 1024 * 1024  # 24 MB


def get_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Chybí GROQ_API_KEY – nastav ho v Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)


def prepare_audio(file_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """Compress audio to mono MP3 64kbps if larger than Groq limit."""
    if len(file_bytes) <= GROQ_LIMIT_BYTES:
        return file_bytes, filename
    ext = filename.rsplit(".", 1)[-1].lower()
    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext if ext != "mp4" else "mp4")
    audio = audio.set_channels(1).set_frame_rate(16000)
    out = io.BytesIO()
    audio.export(out, format="mp3", bitrate="64k")
    compressed = out.getvalue()
    return compressed, "audio_compressed.mp3"


uploaded_file = st.file_uploader(
    "Vyber audio soubor",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"],
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("▶ Zpracovat nahrávku", type="primary"):
        client = get_client()

        # --- Příprava souboru ---
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / 1024 / 1024

        with st.spinner("Přepisuji nahrávku…"):
            if size_mb > 24:
                st.info(f"Soubor je {size_mb:.0f} MB – kompresuji před odesláním…")
            try:
                audio_bytes, audio_name = prepare_audio(file_bytes, uploaded_file.name)
                transcription = client.audio.transcriptions.create(
                    file=(audio_name, audio_bytes),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
                transcript = str(transcription).strip()
            except Exception as e:
                st.error(f"Chyba transkripce: {e}")
                st.stop()

        # --- Sumarizace ---
        with st.spinner("Sumarizuji obsah schůzky…"):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Přepis schůzky:\n\n{transcript}"},
                ],
            )
            summary = json.loads(response.choices[0].message.content)

        # --- Výsledky ---
        st.success("Hotovo!")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Klíčové body")
            for bod in summary.get("klic_body", []):
                st.write(f"• {bod}")

        with col2:
            st.subheader("✅ Rozhodnutí")
            rozhodnuti = summary.get("rozhodnuti", [])
            if rozhodnuti:
                for r in rozhodnuti:
                    st.write(f"• {r}")
            else:
                st.write("_Žádná rozhodnutí_")

        st.subheader("📋 Úkoly")
        ukoly = summary.get("ukoly", [])
        if ukoly:
            for ukol in ukoly:
                kdo = ukol.get("kdo", "neuvedeno")
                co = ukol.get("co", "")
                if kdo != "neuvedeno":
                    st.write(f"▸ **{kdo}**: {co}")
                else:
                    st.write(f"▸ {co}")
        else:
            st.write("_Žádné úkoly_")

        with st.expander("📄 Zobrazit přepis"):
            st.write(transcript)

import json
import os

import streamlit as st
from groq import Groq

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


def get_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Chybí GROQ_API_KEY – nastav ho v Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)


uploaded_file = st.file_uploader(
    "Vyber audio soubor",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"],
    help="Maximální velikost souboru: 25 MB",
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("▶ Zpracovat nahrávku", type="primary"):
        client = get_client()

        # --- Transkripce ---
        with st.spinner("Přepisuji nahrávku…"):
            transcription = client.audio.transcriptions.create(
                file=(uploaded_file.name, uploaded_file.getvalue()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
            transcript = str(transcription).strip()

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

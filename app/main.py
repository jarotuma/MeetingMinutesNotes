import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

load_dotenv()

from app import summarizer, transcriber  # noqa: E402 (after load_dotenv)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Meeting Minutes Notes")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Nepodporovaný formát souboru. Povolené: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        transcript = transcriber.transcribe(str(tmp_path))
        summary = summarizer.summarize(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "transcript": transcript,
        "summary": summary,
    }

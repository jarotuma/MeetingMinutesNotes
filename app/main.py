import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

load_dotenv()

from app import job_store, summarizer, transcriber  # noqa: E402

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Meeting Minutes Notes")
templates = Jinja2Templates(directory="app/templates")


async def _process_job(job_id: str, tmp_path: Path) -> None:
    try:
        await job_store.update_job(
            job_id, status="transcribing", progress_msg="Přepisuji nahrávku…"
        )
        transcript = await transcriber.transcribe(str(tmp_path))

        await job_store.update_job(
            job_id,
            status="summarizing",
            transcript=transcript,
            progress_msg="Sumarizuji obsah schůzky…",
        )
        summary = await summarizer.summarize(transcript)

        await job_store.update_job(
            job_id, status="done", summary=summary, progress_msg="Hotovo"
        )
    except Exception as e:
        await job_store.update_job(
            job_id, status="error", error=str(e), progress_msg="Chyba při zpracování"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Nepodporovaný formát. Povolené: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    tmp_path.write_bytes(await file.read())

    job = await job_store.create_job()
    background_tasks.add_task(_process_job, job.id, tmp_path)

    return {"job_id": job.id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen")
    return {
        "status": job.status,
        "progress_msg": job.progress_msg,
        "transcript": job.transcript,
        "summary": job.summary,
        "error": job.error,
    }

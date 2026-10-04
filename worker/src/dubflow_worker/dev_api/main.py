from __future__ import annotations

import shutil
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dubflow_worker.dev_api.runner import run_dubbing


ROOT = Path(__file__).resolve().parents[4]
JOBS_DIR = ROOT / "output" / "dev-api"
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()

app = FastAPI(title="DubFlow Dev API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _run_job(job_id: str, source: Path, output: Path, options: dict[str, str]) -> None:
    def update(stage: str, progress: int) -> None:
        with jobs_lock:
            jobs[job_id].update(status="PROCESSING", stage=stage, progress=progress)

    try:
        artifact = run_dubbing(source, output, progress=update, **options)
        with jobs_lock:
            jobs[job_id].update(
                status="COMPLETED", stage="COMPLETED", progress=100,
                artifactUrl=f"/api/v1/artifacts/{job_id}/video", artifact_path=str(artifact.resolve()),
            )
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="FAILED", error=f"{type(exc).__name__}: {exc}")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dubflow-dev-api"}


@app.post("/api/v1/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("vi"),
    voice: str | None = Form(None),
    device: str | None = Form(None),
    steps: int | None = Form(None, ge=1, le=32),
) -> dict[str, str]:
    filename = Path(video.filename or "video").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Choose an MP4, MOV, MKV, or WebM video.")
    job_id = uuid.uuid4().hex
    output = JOBS_DIR / job_id
    output.mkdir(parents=True, exist_ok=False)
    source = output / f"source{suffix}"
    try:
        uploaded = 0
        with source.open("wb") as target:
            while chunk := video.file.read(1024 * 1024):
                uploaded += len(chunk)
                if uploaded > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Videos must be 2 GB or smaller.")
                target.write(chunk)
    except HTTPException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(output, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not store uploaded video: {exc}") from exc
    if source.stat().st_size == 0:
        shutil.rmtree(output, ignore_errors=True)
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    with jobs_lock:
        jobs[job_id] = {
            "jobId": job_id, "status": "QUEUED", "stage": "UPLOAD", "progress": 0,
            "artifact_path": None,
        }
    background_tasks.add_task(
        _run_job, job_id, source, output,
        {
            "source_language": source_language,
            "target_language": target_language,
            "voice": voice or os.getenv("DUBFLOW_TTS_VOICE", "gia_bao"),
            "device": device or os.getenv("DUBFLOW_TTS_DEVICE", "gpu"),
            "steps": steps if steps is not None else int(os.getenv("DUBFLOW_TTS_STEPS", "32")),
        },
    )
    return {"jobId": job_id, "status": "QUEUED"}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {key: value for key, value in job.items() if key != "artifact_path"}


@app.get("/api/v1/artifacts/{job_id}/video")
def get_artifact(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        path_value = job.get("artifact_path") if job else None
    if not path_value:
        raise HTTPException(status_code=404, detail="A completed video is not available for this job.")
    path = Path(path_value).resolve()
    if not path.is_relative_to(JOBS_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path, media_type="video/mp4", filename="dubflow-dubbed.mp4")

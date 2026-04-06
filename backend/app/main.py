from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

from starlette.background import BackgroundTask
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.services.segmentation import SegmentationConfig, segment_wav_file

MAX_UPLOAD_SIZE_BYTES = 80 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oneshot-slicer")

app = FastAPI(title="One-shot WAV Slicer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_path / "index.html")


def _is_supported_wav(file: UploadFile) -> bool:
    suffix = Path(file.filename or "").suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


@app.post("/api/process")
async def process_wav(
    file: UploadFile = File(...),
    silence_threshold_db: float = Form(-40.0),
    min_silence_ms: int = Form(80),
    min_clip_ms: int = Form(25),
    padding_ms: int = Form(5),
    normalize: bool = Form(False),
    min_peak_db: float = Form(-60.0),
) -> FileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not _is_supported_wav(file):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max upload size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    config = SegmentationConfig(
        silence_threshold_db=silence_threshold_db,
        min_silence_ms=min_silence_ms,
        min_clip_ms=min_clip_ms,
        padding_ms=padding_ms,
        normalize=normalize,
        min_peak_db=min_peak_db,
    )

    logger.info("Processing uploaded file '%s' with config: %s", file.filename, config)

    temp_dir = tempfile.TemporaryDirectory(prefix="oneshot_slicer_")
    work_dir = Path(temp_dir.name)
    input_wav = work_dir / "input.wav"
    slices_dir = work_dir / "slices"
    output_zip = work_dir / "slices.zip"

    try:
        input_wav.write_bytes(contents)
        exported_files = segment_wav_file(input_wav, slices_dir, config)
        if not exported_files:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No slices detected. Try lowering silence threshold, reducing minimum silence, "
                    "or lowering minimum peak level."
                ),
            )

        with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for wav_file in exported_files:
                archive.write(wav_file, arcname=wav_file.name)

        headers = {
            "X-Detected-Slices": str(len(exported_files)),
            "X-Exported-Filenames": ",".join(path.name for path in exported_files),
        }

        response = FileResponse(
            path=output_zip,
            media_type="application/zip",
            filename="one_shots.zip",
            headers=headers,
            background=BackgroundTask(temp_dir.cleanup),
        )
        return response
    except HTTPException:
        temp_dir.cleanup()
        raise
    except Exception as exc:
        temp_dir.cleanup()
        logger.exception("Failed to process file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process WAV file.")

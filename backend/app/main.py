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

from app.services.segmentation import (
    LoopSegmentationConfig,
    SegmentationConfig,
    segment_loops_wav_file,
    segment_wav_file,
)

MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024
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
    mode: str = Form("oneshot"),
    silence_threshold_db: float = Form(-40.0),
    min_silence_ms: int = Form(80),
    min_clip_ms: int = Form(25),
    padding_ms: int = Form(5),
    normalize: bool = Form(False),
    min_peak_db: float = Form(-60.0),
    dedupe: bool = Form(False),
    dedupe_correlation_threshold: float = Form(0.87),
    dedupe_max_length_ratio: float = Form(1.5),
    dedupe_compare_points: int = Form(2048),
    dedupe_max_lag_ms: float = Form(10.0),
    dedupe_prefilter_threshold: float = Form(0.58),
    dedupe_spectral_threshold: float = Form(0.90),
    dedupe_spectral_min_waveform: float = Form(0.74),
    bpm: float = Form(120.0),
    steps_per_loop: int = Form(32),
    steps_per_beat: int = Form(4),
    auto_offset: bool = Form(True),
    max_offset_ms: float = Form(500.0),
    min_last_loop_ratio: float = Form(0.9),
) -> FileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not _is_supported_wav(file):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    if mode not in {"oneshot", "loop"}:
        raise HTTPException(status_code=400, detail="mode must be either 'oneshot' or 'loop'.")
    if not 0.5 < dedupe_correlation_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="dedupe_correlation_threshold must be between 0.5 and 1.0.",
        )
    if dedupe_max_length_ratio < 1.0:
        raise HTTPException(
            status_code=400,
            detail="dedupe_max_length_ratio must be at least 1.0.",
        )
    if dedupe_compare_points < 128 or dedupe_compare_points > 32768:
        raise HTTPException(
            status_code=400,
            detail="dedupe_compare_points must be between 128 and 32768.",
        )
    if dedupe_max_lag_ms < 0 or dedupe_max_lag_ms > 200:
        raise HTTPException(
            status_code=400,
            detail="dedupe_max_lag_ms must be between 0 and 200.",
        )
    if dedupe_prefilter_threshold < 0.35 or dedupe_prefilter_threshold > 0.95:
        raise HTTPException(
            status_code=400,
            detail="dedupe_prefilter_threshold must be between 0.35 and 0.95.",
        )
    if dedupe_spectral_threshold < 0.5 or dedupe_spectral_threshold > 1.0:
        raise HTTPException(
            status_code=400,
            detail="dedupe_spectral_threshold must be between 0.5 and 1.0.",
        )
    if dedupe_spectral_min_waveform < 0.5 or dedupe_spectral_min_waveform > 0.99:
        raise HTTPException(
            status_code=400,
            detail="dedupe_spectral_min_waveform must be between 0.5 and 0.99.",
        )
    if bpm <= 0:
        raise HTTPException(status_code=400, detail="bpm must be greater than 0.")
    if steps_per_loop <= 0 or steps_per_beat <= 0:
        raise HTTPException(status_code=400, detail="steps_per_loop and steps_per_beat must be positive.")
    if max_offset_ms < 0 or max_offset_ms > 5000:
        raise HTTPException(status_code=400, detail="max_offset_ms must be between 0 and 5000.")
    if min_last_loop_ratio < 0.5 or min_last_loop_ratio > 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_last_loop_ratio must be between 0.5 and 1.0.",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Max upload size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    logger.info("Processing uploaded file '%s' in mode=%s", file.filename, mode)

    temp_dir = tempfile.TemporaryDirectory(prefix="oneshot_slicer_")
    work_dir = Path(temp_dir.name)
    input_wav = work_dir / "input.wav"
    slices_dir = work_dir / "slices"
    output_zip = work_dir / "slices.zip"

    try:
        input_wav.write_bytes(contents)
        if mode == "loop":
            loop_config = LoopSegmentationConfig(
                bpm=bpm,
                steps_per_loop=steps_per_loop,
                steps_per_beat=steps_per_beat,
                auto_offset=auto_offset,
                max_offset_ms=max_offset_ms,
                min_last_loop_ratio=min_last_loop_ratio,
                normalize=normalize,
            )
            exported_files, used_offset_ms = segment_loops_wav_file(input_wav, slices_dir, loop_config)
            discarded_dupes = 0
            labels = ["loop"] * len(exported_files)
        else:
            config = SegmentationConfig(
                silence_threshold_db=silence_threshold_db,
                min_silence_ms=min_silence_ms,
                min_clip_ms=min_clip_ms,
                padding_ms=padding_ms,
                normalize=normalize,
                min_peak_db=min_peak_db,
                dedupe=dedupe,
                dedupe_correlation_threshold=dedupe_correlation_threshold,
                dedupe_max_length_ratio=dedupe_max_length_ratio,
                dedupe_compare_points=dedupe_compare_points,
                dedupe_max_lag_ms=dedupe_max_lag_ms,
                dedupe_prefilter_threshold=dedupe_prefilter_threshold,
                dedupe_spectral_threshold=dedupe_spectral_threshold,
                dedupe_spectral_min_waveform=dedupe_spectral_min_waveform,
            )
            exported_files, discarded_dupes, labels = segment_wav_file(input_wav, slices_dir, config)
            used_offset_ms = 0.0
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
            "X-Discarded-Duplicates": str(discarded_dupes),
            "X-Detected-Labels": ",".join(labels),
            "X-Mode": mode,
            "X-Loop-Offset-Ms": str(int(round(used_offset_ms))) if mode == "loop" else "0",
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

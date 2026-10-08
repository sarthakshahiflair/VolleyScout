from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, UploadFile, status

from app.core.config import settings
from app.schemas.analytics import (
    ModelInventoryResponse,
    ProcessedCsvDetail,
    ProcessedCsvListResponse,
    ProcessedVideoDetail,
    ProcessedVideosResponse,
    UploadAndRunResponse,
)
from app.services.analytics_service import analytics_service

router = APIRouter()


@router.get("/models", response_model=ModelInventoryResponse)
def list_models() -> ModelInventoryResponse:
    return analytics_service.list_models()


@router.get(
    "/processed/videos",
    response_model=ProcessedVideosResponse,
    summary="List successfully processed videos",
    description=(
        "Returns every video that completed the analytics pipeline successfully. "
        "A video is considered processed when both its ``_events.csv`` and "
        "``_analytics.mp4`` output files exist in the output/ folder."
    ),
)
def list_processed_videos() -> ProcessedVideosResponse:
    return analytics_service.list_processed_videos()


@router.get(
    "/processed/videos/csv",
    response_model=ProcessedCsvListResponse,
    summary="List all processed video CSVs",
    description=(
        "Returns metadata for every ``*_events.csv`` file in the output/ folder. "
        "Each entry corresponds to one successfully processed video."
    ),
)
def list_processed_csvs() -> ProcessedCsvListResponse:
    return analytics_service.list_processed_csvs()


@router.get(
    "/processed/videos/csv/{id}",
    response_model=ProcessedCsvDetail,
    summary="Get processed video CSV detail by ID",
    description=(
        "Returns full detail for a single events CSV by its 1-based ID, "
        "including all column names, total row count, and all event data rows."
    ),
)
def get_processed_csv_detail(
    id: Annotated[int, PathParam(ge=1, description="1-based ID matching the list endpoint")],
) -> ProcessedCsvDetail:
    return analytics_service.get_processed_csv_detail(id)


@router.get(
    "/processed/videos/{id}",
    response_model=ProcessedVideoDetail,
    summary="Get processed video detail by ID",
    description=(
        "Returns full detail for a single processed video by its 1-based ID, "
        "including player jersey number mappings, total events count, and full event records."
    ),
)
def get_processed_video_detail(
    id: Annotated[int, PathParam(ge=1, description="1-based ID matching the list endpoint")],
) -> ProcessedVideoDetail:
    return analytics_service.get_processed_video_detail(id)


@router.post(
    "/upload",
    response_model=UploadAndRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a video and run analytics",
    description=(
        "Upload a video file and immediately start the volleyball analytics pipeline.\n\n"
        "**`max_seconds`** controls how much of the video to process:\n"
        "- Leave it empty / pass `null` → **full video** (no limit)\n"
        "- Pass `0` → same as full video\n"
        "- Pass e.g. `60` → process only the first 60 seconds\n\n"
        "The endpoint blocks until processing finishes and returns the full analytics result."
    ),
)
async def upload_and_run_analytics(
    file: Annotated[UploadFile, File(description="Video file to analyse (.mp4, .avi, .mov, .mkv, .webm)")],
    max_seconds: Annotated[
        float | None,
        Form(
            ge=0,
            description=(
                "Seconds of video to process. "
                "Omit or send `null` / `0` to process the entire video."
            ),
        ),
    ] = None,
) -> UploadAndRunResponse:
    # ── Validate filename & extension ─────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video type '{suffix}'. "
                f"Allowed: {sorted(settings.VIDEO_EXTENSIONS)}"
            ),
        )

    # ── Save + process ────────────────────────────────────────────────────────
    return await analytics_service.upload_and_run(file, max_seconds)

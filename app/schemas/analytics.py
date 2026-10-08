from pathlib import Path
from typing import Any

from pydantic import Field

from app.schemas.base import APIModel


class AnalyticsRunRequest(APIModel):
    video_path: str = Field(..., description="Path to a video file on this machine.")
    max_seconds: float | None = Field(default=90.0, ge=0)
    split_label: str = "api"
    pretrained_only: bool = False
    force_finetuned: bool = False
    player_weights: str = "yolo11n.pt"
    ball_weights: str | None = None
    max_jersey_number: int = Field(default=25, ge=1, le=99)


class AnalyticsRunResponse(APIModel):
    video: str
    out_video: str
    out_csv: str
    frames: int
    n_events: int
    jerseys: dict[str, str] | dict[int, str]
    output_rows: list[dict[str, Any]]


class FileUploadResponse(APIModel):
    filename: str
    path: Path
    size_bytes: int


class UploadAndRunResponse(APIModel):
    """
    Combined response returned by POST /api/analytics/upload.
    Contains both the upload metadata and the full analytics result.
    """

    # ── Upload metadata ──────────────────────────────────────────────────────
    filename: str = Field(..., description="Saved filename inside the uploads/ folder.")
    size_bytes: int = Field(..., description="Size of the uploaded video in bytes.")

    # ── Analytics result ─────────────────────────────────────────────────────
    video: str = Field(..., description="Original video filename used for processing.")
    out_video: str = Field(..., description="Absolute path to the annotated output MP4.")
    out_csv: str = Field(..., description="Absolute path to the events CSV.")
    frames: int = Field(..., description="Total frames processed.")
    n_events: int = Field(..., description="Number of volleyball events detected.")
    jerseys: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of track-id → jersey number string.",
    )
    output_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full event rows as returned by the analytics pipeline.",
    )


# ── Processed Videos ─────────────────────────────────────────────────────────

class ProcessedVideoItem(APIModel):
    """One entry in the processed-videos list. Includes its list position as `id`."""

    id: int = Field(..., description="1-based position in the sorted list — use this as the detail endpoint ID.")
    filename: str = Field(..., description="Original video filename.")
    size_bytes: int = Field(..., description="File size of the output analytics MP4 in bytes.")
    processed_at: str = Field(..., description="Last-modified timestamp of the analytics MP4 in ISO-8601 format.")
    out_video: str = Field(..., description="Absolute path to the annotated analytics MP4.")
    out_csv: str = Field(..., description="Absolute path to the events CSV.")


class ProcessedVideosResponse(APIModel):
    total: int = Field(..., description="Total number of successfully processed videos.")
    videos: list[ProcessedVideoItem]


class ProcessedVideoDetail(APIModel):
    """Full detail for a single processed video, including jerseys and all event rows."""

    id: int = Field(..., description="1-based ID matching the list endpoint.")
    filename: str
    size_bytes: int
    processed_at: str
    out_video: str
    out_csv: str
    jerseys: dict[str, str] = Field(
        default_factory=dict,
        description="Track-id → jersey number mapping loaded from the _jerseys.json sidecar.",
    )
    n_events: int = Field(..., description="Number of event rows in the CSV.")
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full list of event rows from the events CSV.",
    )


# ── Processed CSVs ────────────────────────────────────────────────────────────

class ProcessedCsvItem(APIModel):
    """One entry in the processed-CSVs list. Includes its list position as `id`."""

    id: int = Field(..., description="1-based position in the sorted list — use this as the detail endpoint ID.")
    csv_filename: str = Field(..., description="CSV filename, e.g. match_events.csv.")
    csv_path: str = Field(..., description="Absolute path to the events CSV.")
    video_filename: str = Field(..., description="Original video name derived from the CSV stem.")
    size_bytes: int = Field(..., description="File size of the CSV in bytes.")
    created_at: str = Field(..., description="Last-modified timestamp of the CSV in ISO-8601 format.")


class ProcessedCsvListResponse(APIModel):
    total: int = Field(..., description="Total number of events CSV files found.")
    csvs: list[ProcessedCsvItem]


class ProcessedCsvDetail(APIModel):
    """Full detail for a single events CSV, including all rows and column names."""

    id: int = Field(..., description="1-based ID matching the list endpoint.")
    csv_filename: str
    csv_path: str
    video_filename: str
    size_bytes: int
    created_at: str
    columns: list[str] = Field(..., description="Column names present in the CSV.")
    row_count: int = Field(..., description="Total number of data rows.")
    rows: list[dict[str, Any]] = Field(..., description="All event rows as a list of dicts.")


# ── Models ────────────────────────────────────────────────────────────────────

class ModelFile(APIModel):
    name: str
    path: Path
    exists: bool


class ModelInventoryResponse(APIModel):
    models: list[ModelFile]

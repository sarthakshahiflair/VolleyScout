from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "VolleyScout"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    BACKEND_CORS_ORIGINS: str = "*"

    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./volleyscout.db"

    PROJECT_ROOT: Path = PROJECT_ROOT
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    MODEL_DIR: Path = PROJECT_ROOT / "models"

    DEFAULT_MAX_SECONDS: float | None = 90.0
    MAX_JERSEY_NUMBER: int = 25
    VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> List[str]:
        if self.BACKEND_CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()

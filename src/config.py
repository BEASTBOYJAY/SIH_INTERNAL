# src/config.py

"""Configuration for the ML Face Identifier Service."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Main service configuration for the Identifier."""

    # Device selection (cuda or cpu)
    device: str = Field(default="cuda:0", env="DEVICE")

    # Model Paths

    id_detection_model_path: Path = Field(
        default=Path("id_detector.pt"), env="ID_DETECTION_MODEL_PATH"
    )
    face_detection_model_path: Path = Field(
        default=Path("yolov8x-face.pt"), env="FACE_DETECTION_MODEL_PATH"
    )

    face_recognition_threshold: float = Field(
        default=0.7, env="FACE_RECOGNITION_THRESHOLD"
    )

    face_detection_confidence: float = Field(
        default=0.7, env="FACE_DETECTION_CONFIDENCE"
    )
    id_detection_confidence: float = Field(default=0.45, env="ID_DETECTION_CONFIDENCE")
    id_detection_iou: float = Field(default=0.4, env="ID_DETECTION_IOU")

    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()

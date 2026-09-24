import os
from pathlib import Path

DEFAULT_MODEL_PATH = "/Users/wennuan/.lmstudio/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/"

class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEFAULT_MODEL_NAME: str = "models/qwen3-embedding-4b"
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "8192"))

settings = Settings()

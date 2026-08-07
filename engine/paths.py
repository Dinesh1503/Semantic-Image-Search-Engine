import os
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ENGINE_DIR / "models"
DEFAULT_IMAGES_DIR = ENGINE_DIR / "test_data" / "a"

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".JPG")


class ModelSource(NamedTuple):
    source: str
    local_only: bool
    cache_dir: Path


def models_dir() -> Path:
    """Directory holding locally downloaded model weights (override with MODELS_PATH)."""
    load_dotenv()
    return Path(os.getenv("MODELS_PATH", DEFAULT_MODELS_DIR))


def images_dir() -> Path:
    """Directory of images served by the API (override with IMAGES_PATH)."""
    load_dotenv()
    return Path(os.getenv("IMAGES_PATH", DEFAULT_IMAGES_DIR))


def resolve_model_source(model_id: str, local_dir_name: str) -> ModelSource:
    """Prefer a local copy of a model over downloading it from the hub.

    `source` is what to load from, `local_only` tells the loader to stay offline and
    `cache_dir` is where a hub download should be stored.
    """
    local_path = models_dir() / local_dir_name
    if local_path.exists():
        return ModelSource(str(local_path), True, local_path)
    return ModelSource(model_id, False, local_path)

import os

import torch
from dotenv import load_dotenv


def torch_device() -> str:
    """Best available accelerator, overridable with the DEVICE environment variable."""
    load_dotenv()

    override = os.getenv("DEVICE")
    if override:
        return override

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

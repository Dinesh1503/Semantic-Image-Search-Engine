"""Shared test fixtures and stubs.

The ``engine`` package pulls in Apple-only / heavy runtime dependencies
(``mlx``, ``mlx_embeddings``, ``mlx_vlm``, ``psycopg``) and several modules run
imperative demo code at import time.  To exercise the pure Python logic in unit
tests on any platform we install light-weight stand-ins into ``sys.modules`` and
provide the environment variables the modules expect *before* they are imported.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np


def _install_fake_mlx():
    """Expose numpy under the ``mlx.core`` name.

    ``EmbeddingModel`` only uses ``array``, ``linalg.norm``, ``maximum`` and
    ``float32`` from ``mlx.core`` -- all of which numpy provides with matching
    semantics -- so numpy is a faithful drop-in for the tested code paths.
    """
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.__path__ = []  # mark as a package so ``import mlx.core`` works
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = np.array
    mlx_core.linalg = np.linalg
    mlx_core.maximum = np.maximum
    mlx_core.float32 = np.float32
    mlx_pkg.core = mlx_core
    sys.modules["mlx"] = mlx_pkg
    sys.modules["mlx.core"] = mlx_core


def _default_embedding_backend():
    """A (model, tokenizer) pair usable by ``EmbeddingModel``."""
    tokenizer = MagicMock(name="tokenizer")
    tokenizer.encode.return_value = [1, 2, 3]

    hidden = types.SimpleNamespace(last_hidden_state=np.ones((1, 3, 4)))
    model = MagicMock(name="model", return_value=hidden)
    return model, tokenizer


def _install_fake_mlx_embeddings():
    pkg = types.ModuleType("mlx_embeddings")
    pkg.__path__ = []
    utils = types.ModuleType("mlx_embeddings.utils")
    utils.load = MagicMock(name="load", return_value=_default_embedding_backend())
    pkg.utils = utils
    sys.modules["mlx_embeddings"] = pkg
    sys.modules["mlx_embeddings.utils"] = utils


def _install_fake_mlx_vlm():
    vlm = types.ModuleType("mlx_vlm")
    vlm.load = MagicMock(name="load", return_value=("vlm-model", "vlm-processor"))
    vlm.generate = MagicMock(
        name="generate",
        return_value=types.SimpleNamespace(text="a factual caption"),
    )
    sys.modules["mlx_vlm"] = vlm


def _install_fake_psycopg():
    psycopg = types.ModuleType("psycopg")

    def _connect(*_args, **_kwargs):
        conn = MagicMock(name="connection")
        cursor = conn.cursor.return_value
        cursor.fetchone.return_value = (0,)
        cursor.fetchall.return_value = []
        return conn

    psycopg.connect = MagicMock(name="connect", side_effect=_connect)
    sys.modules["psycopg"] = psycopg


def _install_fake_dotenv():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = MagicMock(name="load_dotenv")
    sys.modules["dotenv"] = dotenv


# ``db.py`` builds its connection string from these at import time.
os.environ.setdefault("DB", "test_db")
os.environ.setdefault("USERNAME", "test_user")
os.environ.setdefault("PORT", "5432")
os.environ.setdefault("PASSWORD", "secret")

_install_fake_mlx()
_install_fake_mlx_embeddings()
_install_fake_mlx_vlm()
_install_fake_psycopg()
_install_fake_dotenv()

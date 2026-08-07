"""Unit tests for ``engine/models/embedding_model.py``."""

import types

import numpy as np
import pytest

from models.embedding_model import EmbeddingModel


def test_init_loads_named_model():
    model = EmbeddingModel()
    assert model.model_name == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    # ``load`` returns a (model, tokenizer) pair that gets stored on the instance.
    assert model.model is not None
    assert model.tokenizer is not None


def _embedding_model_with_backend(tokens, hidden_state):
    model = EmbeddingModel()
    model.tokenizer.encode.return_value = tokens
    model.model.return_value = types.SimpleNamespace(last_hidden_state=hidden_state)
    return model


def test_get_vector_embeddings_returns_normalised_list():
    hidden = np.array([[[3.0, 4.0], [30.0, 40.0]]])  # last token is (30, 40)
    model = _embedding_model_with_backend([1, 2], hidden)

    vector = model.get_vector_embeddings("a red car")

    assert isinstance(vector, list)
    assert all(isinstance(v, float) for v in vector)
    # (30, 40) normalised by its L2 norm (50) -> (0.6, 0.8).
    assert vector == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_get_vector_embeddings_encodes_the_input_text():
    model = _embedding_model_with_backend([7, 8, 9], np.ones((1, 3, 2)))
    model.tokenizer.encode.reset_mock()

    model.get_vector_embeddings("hello world")

    model.tokenizer.encode.assert_called_once_with("hello world")


def test_get_vector_embeddings_handles_zero_vector_without_dividing_by_zero():
    hidden = np.zeros((1, 4, 3))
    model = _embedding_model_with_backend([0], hidden)

    vector = model.get_vector_embeddings("blank")

    assert vector == [0.0, 0.0, 0.0]

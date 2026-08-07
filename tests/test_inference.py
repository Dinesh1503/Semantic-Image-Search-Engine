"""Unit tests for ``engine/inference.py``."""

import types

from inference import Inference


def _make_files(base, names):
    for rel in names:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_get_index_collects_supported_images_recursively(tmp_path):
    _make_files(
        tmp_path,
        [
            "a.jpg",
            "b.png",
            "c.jpeg",
            "notes.txt",
            "archive.zip",
            "nested/d.jpg",
        ],
    )

    result = Inference().get_index(str(tmp_path))

    filenames = sorted(item["image_filename"] for item in result)
    assert filenames == ["a.jpg", "b.png", "c.jpeg", "d.jpg"]
    # Each entry exposes both the filename and its full path.
    for item in result:
        assert set(item) == {"image_path", "image_filename"}
        assert item["image_filename"] in item["image_path"]


def test_get_index_returns_empty_when_no_images(tmp_path):
    _make_files(tmp_path, ["readme.md", "data.csv"])
    assert Inference().get_index(str(tmp_path)) == []


def test_get_image_captions_returns_generated_text():
    calls = []

    class FakeModel:
        def generate(self, image_path):
            calls.append(image_path)
            return types.SimpleNamespace(text=f"caption for {image_path}")

    image_indexes = [
        {"image_path": "/imgs/a.jpg", "image_filename": "a.jpg"},
        {"image_path": "/imgs/b.jpg", "image_filename": "b.jpg"},
    ]

    captions = Inference().get_image_captions(FakeModel(), image_indexes)

    # NOTE: the current implementation ``break``s after the first image.
    assert captions == ["caption for /imgs/a.jpg"]
    assert calls == ["/imgs/a.jpg"]


def test_get_image_captions_empty_input():
    class FakeModel:
        def generate(self, image_path):  # pragma: no cover - should not be called
            raise AssertionError("generate should not be called")

    assert Inference().get_image_captions(FakeModel(), []) == []


def test_vectorise_captions_delegates_to_embedding_model():
    class FakeEmbeddingModel:
        def __init__(self):
            self.seen = None

        def get_vector_embeddings(self, caption):
            self.seen = caption
            return [0.1, 0.2, 0.3]

    embedding_model = FakeEmbeddingModel()
    vector = Inference().vectorise_captions(embedding_model, "a sunny beach")

    assert vector == [0.1, 0.2, 0.3]
    assert embedding_model.seen == "a sunny beach"

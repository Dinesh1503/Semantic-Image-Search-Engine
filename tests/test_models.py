"""Unit tests for ``engine/models/models.py``."""

import inspect

import pytest

from models.models import MlxModel, VlmModel


def test_vlmmodel_is_abstract():
    assert inspect.isabstract(VlmModel)
    with pytest.raises(TypeError):
        VlmModel()  # cannot instantiate an ABC with an abstract method


def test_mlxmodel_is_a_vlmmodel_subclass():
    assert issubclass(MlxModel, VlmModel)


def test_mlxmodel_init_loads_model_and_builds_prompt():
    model = MlxModel()

    assert model.model_name == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert model.model == "vlm-model"
    assert model.processor == "vlm-processor"

    # The chat template must wrap the instruction with the expected VLM tokens.
    assert model.system_prompt.startswith("<|im_start|>user\n")
    assert model.system_prompt.rstrip().endswith("<|im_start|>assistant")
    assert "<|vision_start|><|image_pad|><|vision_end|>" in model.system_prompt
    assert "MAIN SUBJECTS" in model.system_prompt


def test_mlxmodel_generate_delegates_to_backend():
    model = MlxModel()

    response = model.generate("/tmp/cat.jpg")

    assert response.text == "a factual caption"
    model.generate_captions_func.assert_called_once()
    _, kwargs = model.generate_captions_func.call_args
    assert kwargs["image"] == ["/tmp/cat.jpg"]
    assert kwargs["prompt"] == model.system_prompt
    assert kwargs["temp"] == 0
    assert kwargs["max_tokens"] == 600

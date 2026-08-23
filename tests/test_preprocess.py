"""Preprocessing contract: whatever comes in, a [1, 3, 224, 224] float tensor comes out."""

from __future__ import annotations

import io

import pytest
import torch
from fastapi import HTTPException
from PIL import Image

from app.utils.preprocess import preprocess_image

EXPECTED_SHAPE = (1, 3, 224, 224)


def test_output_shape_from_real_sample(sample_image_bytes: bytes) -> None:
    """The shape the model requires, from an actual test_samples image."""
    tensor = preprocess_image(sample_image_bytes)

    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == EXPECTED_SHAPE
    assert tensor.dtype == torch.float32


@pytest.mark.parametrize(
    "size,mode",
    [
        ((64, 64), "RGB"),        # smaller than 224, upscaled
        ((1024, 300), "RGB"),     # non-square, aspect ratio not preserved by Resize((224,224))
        ((224, 224), "L"),        # greyscale, must be converted to 3 channels
        ((256, 256), "RGBA"),     # alpha channel, must be dropped
    ],
)
def test_output_shape_is_independent_of_input(size, mode) -> None:
    """Resize + convert("RGB") must normalise every input to the same tensor shape."""
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")

    tensor = preprocess_image(buffer.getvalue())

    assert tuple(tensor.shape) == EXPECTED_SHAPE


def test_imagenet_normalization_is_applied(sample_image_bytes: bytes) -> None:
    """Normalised pixels leave the [0, 1] range that ToTensor produces."""
    tensor = preprocess_image(sample_image_bytes)

    assert tensor.min() < 0.0, "expected ImageNet mean subtraction to push values negative"


def test_invalid_bytes_raise_http_400() -> None:
    """Undecodable bytes surface as a 400, not an unhandled PIL error."""
    with pytest.raises(HTTPException) as excinfo:
        preprocess_image(b"this is not an image")

    assert excinfo.value.status_code == 400

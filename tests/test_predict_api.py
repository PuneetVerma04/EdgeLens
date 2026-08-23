"""API-level tests for /api/edgelens/predict and the health endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings

PREDICT_URL = "/api/edgelens/predict"
HEALTH_URL = "/api/edgelens/"


# ---------------------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------------------


def test_health_endpoint(client) -> None:
    response = client.get(HEALTH_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "message" in body


def test_health_response_carries_process_time_header(client) -> None:
    """The timing middleware in main.py is wired up."""
    response = client.get(HEALTH_URL)

    assert "X-Process-Time" in response.headers
    assert float(response.headers["X-Process-Time"]) >= 0.0


# ---------------------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------------------


def test_predict_happy_path(client, stub_model, sample_image_path: Path) -> None:
    """A real test_samples image through the full route: validate, preprocess, infer, format."""
    with sample_image_path.open("rb") as handle:
        response = client.post(
            PREDICT_URL,
            files={"file": (sample_image_path.name, handle, "image/jpeg")},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == {"label", "confidence", "inference_time"}
    assert body["label"] in {"OK", "Defective"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["inference_time"] >= 0.0
    assert stub_model.call_count == 1, "the route should invoke the model exactly once"


def test_predict_uses_the_model_output(client, sample_image_path: Path) -> None:
    """The response reflects the model's logits rather than a fixed default.

    StubModel returns logits favouring class 1, which postprocess_output maps to "OK".
    """
    with sample_image_path.open("rb") as handle:
        response = client.post(
            PREDICT_URL,
            files={"file": (sample_image_path.name, handle, "image/jpeg")},
        )

    body = response.json()
    assert body["label"] == "OK"
    assert body["confidence"] > 0.9


# ---------------------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------------------


def test_oversized_file_is_rejected(client) -> None:
    """Payloads above settings.max_file_size_mb get a 413 before any decoding happens."""
    limit_mb = get_settings().max_file_size_mb
    oversized = b"\xff" * (limit_mb * 1024 * 1024 + 1)

    response = client.post(
        PREDICT_URL,
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )

    assert response.status_code == 413
    assert str(limit_mb) in response.json()["detail"]


def test_size_limit_follows_config_not_a_hardcoded_constant(client, monkeypatch) -> None:
    """Lowering max_file_size_mb must change what the route rejects.

    Regression guard for the limit being read from config.py rather than a module constant.
    The same 2 MB payload is posted twice: rejected for size at a 1 MB limit, and not
    rejected for size at the configured limit. Deliberately independent of what that
    configured limit actually is, since a developer's .env can set it to anything.
    """
    settings = get_settings()
    configured_mb = settings.max_file_size_mb
    if configured_mb <= 2:
        pytest.skip(f"configured limit is {configured_mb} MB; this test needs headroom above 2 MB")

    payload = b"\xff" * (2 * 1024 * 1024)

    # get_settings() is lru_cached, so every caller shares this one object. Mutating the
    # field retargets the route; monkeypatch restores it on teardown.
    monkeypatch.setattr(settings, "max_file_size_mb", 1, raising=False)
    tightened = client.post(PREDICT_URL, files={"file": ("mid.jpg", payload, "image/jpeg")})
    assert tightened.status_code == 413
    assert "1 MB" in tightened.json()["detail"]

    monkeypatch.setattr(settings, "max_file_size_mb", configured_mb, raising=False)
    relaxed = client.post(PREDICT_URL, files={"file": ("mid.jpg", payload, "image/jpeg")})
    # 400 from preprocessing: 2 MB of 0xff passes the size and content-type gates, then
    # fails to decode as an image. The point is that it is no longer a 413.
    assert relaxed.status_code == 400


def test_wrong_content_type_is_rejected(client, sample_image_bytes: bytes) -> None:
    """A non-image content type gets a 400 even when the bytes are a valid image."""
    response = client.post(
        PREDICT_URL,
        files={"file": ("sample.txt", sample_image_bytes, "text/plain")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_missing_file_is_rejected(client) -> None:
    """No multipart file at all is a FastAPI validation error."""
    response = client.post(PREDICT_URL)

    assert response.status_code == 422

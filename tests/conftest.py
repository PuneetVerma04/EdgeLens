"""Shared pytest fixtures for the EdgeLens backend tests.

The app under test is exercised through FastAPI's TestClient, which runs the real lifespan.
Two startup dependencies are stubbed out so the suite runs anywhere, including CI:

  * `load_model` -- the real ResNet50 checkpoint is ~94 MB and gitignored, so it does not
    exist on a fresh clone or on a CI runner. A stub model stands in for it.
  * `connect_db` / `close_db` -- no MongoDB in CI. `log_inference` already no-ops when the
    collection is None, so prediction logging degrades exactly as it does in production.

Everything else -- routing, validation, preprocessing, postprocessing, response schemas --
is the real code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SAMPLES = REPO_ROOT / "test_samples"

# The backend is laid out so that `app` is importable from backend/, which is what the
# Dockerfile and the uvicorn command both assume.
sys.path.insert(0, str(REPO_ROOT / "backend"))


class StubModel:
    """Stands in for the trained ResNet50.

    Returns a fixed [1, 2] logit tensor, matching what the current binary OK/Defective
    classifier emits. `predict.py` checks the output is 2-D with batch size 1, and
    `postprocess_output` softmaxes over dim=1 with label_map {0: Defective, 1: OK}.

    NOTE: when core/model.py and utils/postprocess.py are rewritten for YOLO box output
    (see NEU_DET_YOLO_Plan.md), this stub's output shape must change with them.
    """

    #: Logits favouring class 1 -> "OK" after softmax.
    LOGITS = [[-1.0, 3.0]]

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        # Assert the contract the real model relies on, so a preprocessing regression fails
        # here rather than silently producing a plausible-looking prediction.
        assert tensor.shape == (1, 3, 224, 224), f"unexpected input shape {tuple(tensor.shape)}"
        return torch.tensor(self.LOGITS)


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """A real image from test_samples/, used instead of a synthetic one."""
    candidate = TEST_SAMPLES / "cast_ok_0_119.jpeg"
    if not candidate.is_file():
        pytest.skip(f"missing fixture image {candidate}")
    return candidate


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path: Path) -> bytes:
    return sample_image_path.read_bytes()


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def client(monkeypatch, stub_model):
    """TestClient over the real app, with model loading and MongoDB stubbed out."""
    from fastapi.testclient import TestClient

    import app.main as main

    async def _noop() -> None:
        return None

    # main.py does `from ... import load_model`, so the name to patch lives in main's
    # namespace, not in the module it was defined in.
    monkeypatch.setattr(main, "load_model", lambda: stub_model)
    monkeypatch.setattr(main, "connect_db", _noop)
    monkeypatch.setattr(main, "close_db", _noop)

    with TestClient(main.app) as test_client:
        yield test_client

"""Smoke-test fixtures (guide section 5.4). CPU only, tiny subsets."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SMOKE_N = 32          # guide section 5.4: tiny subset of 32 images
SMOKE_SEED = 101      # a tuning seed, so nothing novel is introduced


@pytest.fixture(scope="session", autouse=True)
def deterministic():
    """Full op determinism for the smoke tests.

    Guide section 5.4 check 4: if determinism proves impossible on GPU it
    must be logged; on this CPU environment it is enabled outright.
    """
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    yield


@pytest.fixture(scope="session")
def tiny():
    """(images, script_ids, labels) for 32 training images."""
    import data
    return data.as_arrays("train", "primary", limit=SMOKE_N)

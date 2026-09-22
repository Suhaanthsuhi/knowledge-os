from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.generate import build_all


@pytest.fixture(scope="session")
def fixtures() -> dict[str, Path]:
    """Regenerate the synthetic fixture files once per test session."""
    return build_all()

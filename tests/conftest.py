from __future__ import annotations

from copy import deepcopy

import pytest

from src.defaults import DEFAULTS
from src.schema import migrate_assumptions


@pytest.fixture
def base_inputs() -> dict:
    inputs, _, _ = migrate_assumptions(deepcopy(DEFAULTS))
    return inputs


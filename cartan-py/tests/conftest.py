import numpy as np
import pytest
from hypothesis import settings

# Tolerance for floating-point comparisons
RTOL = 1e-10
ATOL = 1e-12

# Relaxed tolerance for operations near cut locus or high dimension
RTOL_RELAXED = 1e-6
ATOL_RELAXED = 1e-8

def assert_allclose(actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

@pytest.fixture
def rng():
    return np.random.default_rng(42)

# Hypothesis CI settings: fewer examples, no deadline
settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("default", max_examples=200, deadline=None)

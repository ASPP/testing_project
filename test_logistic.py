from numpy.testing import assert_allclose
import pytest
from logistic import logistic_step

# Add here your test for the logistic map


def test_logistic_step_corner_cases():
    # Test cases are (x, r, expected)
    cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    for x, r, expected in cases:
        result = logistic_step(x, r)
        assert_allclose(result, expected)

@pytest.mark.parametrize('x, r, expected', [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5),
    ]
    )
def test_logistic_step_generic_cases(x, r, expected):
    # Test cases are (x, r, expected)
    result = logistic_step(x, r)
    assert_allclose(result, expected)
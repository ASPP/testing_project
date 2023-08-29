from numpy.testing import assert_allclose
import pytest
from logistic import logistic_step


@pytest.mark.parametrize('x, r, expected', [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    )
def test_logistic_step_corner_cases(x, r, expected):
    result = logistic_step(x, r)
    assert_allclose(result, expected)

@pytest.mark.parametrize('x, r, expected', [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5),
    ]
    )
def test_logistic_step_generic_cases(x, r, expected):
    result = logistic_step(x, r)
    assert_allclose(result, expected)
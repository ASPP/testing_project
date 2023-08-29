import pytest
from numpy.testing import assert_allclose

from logistic import f

cases = [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2.0, 0.5)
    ]
@pytest.mark.parametrize('x, r, expected', cases)
def test_generic_cases(x, r, expected):
    result = f(x, r)
    assert_allclose(result, expected) 


cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
@pytest.mark.parametrize('x, r, expected', cases)
def test_f_corner_cases(x, r, expected):
    result = f(x, r)
    assert_allclose(result, expected)

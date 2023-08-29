import pytest
from numpy.testing import assert_allclose

from logistic import f, iterate_f

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


cases = [
        (1, 0.1, 2.2, [0.1, 0.198]),
        (4, 0.2, 3.4, [0.2, 0.544, 0.843418, 0.449019, 0.841163]),
        (3, 0.5, 2, [0.5, 0.5, 0.5, 0.5])
    ]
@pytest.mark.parametrize('it, x, r, expected', cases)
def test_iterate_f(it, x, r, expected):
    result = iterate_f(it, x, r)
    assert_allclose(result, expected, atol=0.000001)
import pytest
import numpy as np
from numpy.testing import assert_allclose

from logistic import f, iterate_f
from logistic_fit import fit_r

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


cases = [
        (23, 0.3, 3.421),
        (50, 0.6, 2.57),
        (40, 0.8, 0),
        (37, 0.1, 2.56)
    ]
@pytest.mark.parametrize('it, x0, r', cases)
def test_fit_r(it, x0, r):
    xs = iterate_f(it, x0, r)
    result = fit_r(xs)
    assert np.isclose(result, r, atol=0.0001)
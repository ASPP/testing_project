import pytest
import numpy as np
from numpy.testing import assert_allclose
from logistic import f
# from logistic import f, iterate_f
from logistic_fit import fit_r
from logistic import iterate_f

@pytest.mark.parametrize('x, r, expected', [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
)
def test_f_corner_cases(x, r, expected):
    result = f(x, r)
    assert_allclose(result, expected)

@pytest.mark.parametrize(
    'x, r, expected',
    [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5),
    ]
)
def test_f_generic_cases(x, r, expected):
    result = f(x, r)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    'x, r, it, expected', 
    [
        (0.1, 2.2, 1, [0.1, 0.198]),
        (0.2, 3.4, 4, [0.2, 0.544, 0.843418, 0.449019, 0.841163]),
        (0.5, 2, 3, [0.5, 0.5, 0.5, 0.5]),
    ]
)
def test_iterate_f(x, r, it, expected):
    result = iterate_f(x, r, it)
    assert_allclose(result, expected, rtol=5e-07)

@pytest.mark.parametrize(
    'x, r, it',
    [
        (0.3, 3.421, 23)
    ]
)
def test_fit_r(x, r, it):
    traj = iterate_f(x, r, it)
    r_fitted = fit_r(traj)
    assert_allclose(r, r_fitted)

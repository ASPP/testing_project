from numpy.testing import assert_allclose
import pytest
from logistic import logistic_step, run_iterations
from logistic_fit import fit_r
from itertools import product
import numpy as np

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

@pytest.mark.parametrize('x, r, n_iter, expected', [
        (0.1, 2.2, 1, [0.1, 0.198]),
        (0.2, 3.4, 4, [0.2, 0.544, 0.843418, 0.449019, 0.841163]),
        (0.5, 2, 3, [0.5, 0.5, 0.5, 0.5]),
    ]
    )
def test_logistic_iterations(x, r, n_iter, expected):
    result = run_iterations(x, r, n_iter)
    assert_allclose(result, expected, rtol=1e-4)




@pytest.mark.parametrize('x, r, n_iter', [(i/10,f/10,20) for i,f in product(range(1,10),range(10,30,2))]
    )
def test_logistic_fit_r(x, r, n_iter):
    result = fit_r(run_iterations(x, r, n_iter))
    assert_allclose(result, r, rtol=1e-3)


SEED = 5
@pytest.mark.parametrize('x', [np.random.RandomState(SEED).uniform(0.0001, 0.9999) for _ in range(50)])
def test_logistic_convergence(x):
    r = 1.5
    n_iter = 100
    expected = 1/3
    result = run_iterations(x, r, n_iter)
    assert_allclose(result[-1], expected, rtol=1e-4)

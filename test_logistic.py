import pytest
import numpy as np
from numpy.testing import assert_allclose
from logistic import f
# from logistic import f, iterate_f
from logistic_fit import fit_r
from logistic import iterate_f

SEED = np.random.randint(0, 2**31)

@pytest.fixture
def random_state():
    print(f'Using seed {SEED}')
    random_state = np.random.RandomState(SEED)
    return random_state

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

def test_random_converge(random_state, r=1.5, it=100, expected=1/3, low=0.0001, high=0.9999):
    for _ in range(100):
        x0 = random_state.uniform(low, high)
        traj = iterate_f(x0, r, it)
        assert_allclose(traj[-1], expected, atol=1e-3)
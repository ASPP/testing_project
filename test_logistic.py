from numpy.testing import assert_allclose
from logistic import f, iterate_f
import pytest
from math import isclose
import numpy as np

SEED = np.random.randint(0, 2**31)
# Add here your test for the logistic map

# def test_f_general():
#     cases = [
#         (0.1, 2.2, 0.198),
#         (0.2, 3.4, 0.544),
#         (0.5, 2, 0.5),
#     ]
#     for x, r, expected in cases:
#         result = f(x, r)
#         assert_allclose(result, expected)


# def test_f_corner_cases():
#     # Test cases are (x, r, expected)
#     cases = [
#         (0, 1.1, 0),
#         (1, 3.7, 0),
#     ]
#     for x, r, expected in cases:
#         result = f(x, r)
#         assert_allclose(result, expected)


# @pytest.mark.parametrize('x,r,expected',[(0.1,2.2,0.198),(0.2,3.4,0.544),(0.5,2,0.5),(0,1.1,0),(1,3.7,0)])
# def test_f(x,r,expected):
#     result = f(x,r)
#     assert isclose(result,expected)



# @pytest.mark.parametrize('x,r,it,expected',[(0.1,2.2,1,[0.1,0.198]),(0.2,3.4,4,[0.2,0.544,0.843418,0.449019,0.841163]),(0.5,2,3,[0.5,0.5,0.5,0.5])])
# def test_iterate_f(x,r,it,expected):
#     result = iterate_f(x,r,it)
#     assert_allclose(result,expected,atol=0.001)

@pytest.fixture
def random_state():
    print(f'\nUsing Seed: {SEED}')
    random_state = np.random.RandomState(SEED)
    return random_state

def test_random_starting_point_convergence(random_state):
    r = 1.5
    n = 100
    it = 30
    n_convergence_datapoints = 3
    for _ in range(n):
        x0 = random_state.uniform(0.0001, 0.9999)
        xs = iterate_f(x0,r,it)
        for i in range(1, n_convergence_datapoints+1):
            assert np.isclose(xs[-i], 1/3, atol=0.001)
        
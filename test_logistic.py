from numpy.testing import assert_allclose
from logistic import f, iterate_f
import pytest
from math import isclose
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



@pytest.mark.parametrize('x,r,it,expected',[(0.1,2.2,1,[0.1,0.198]),(0.2,3.4,4,[0.2,0.544,0.843418,0.449019,0.841163]),(0.5,2,3,[0.5,0.5,0.5,0.5])])
def test_iterate_f(x,r,it,expected):
    result = iterate_f(x,r,it)
    assert_allclose(result,expected,atol=0.001)
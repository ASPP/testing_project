from numpy.testing import assert_allclose
import pytest
from logistic import f
from logistic import iterate_f

# Add here your test for the logistic map

@pytest.mark.parametrize('x,r,expected', [(0, 1.1, 0),
        (1, 3.7, 0),])
def test_f_corner_cases(x,r,expected):
    result = f(x, r)
    assert_allclose(result, expected)

@pytest.mark.parametrize('x,r,expected', [(0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5)])
def test_f_generic_cases(x,r,expected):
    result = f(x, r)
    assert_allclose(result, expected)


@pytest.mark.parametrize('it, x, r, expected', [
    (1, 0.1, 2.2, [0.1, 0.198]),
    (4, 0.2, 3.4, [0.2, 0.544, 0.843418, 0.449019, 0.841163]),
    (3, 0.5, 2, [0.5, 0.5, 0.5, 0.5]),
])
def test_iterate_f(it,x,r,expected):
    result = iterate_f(it, x, r)
    assert_allclose(result, expected, atol=0.00001)

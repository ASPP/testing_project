from numpy.testing import assert_allclose
import pytest
from logistic import f

# Add here your test for the logistic map


def test_f_corner_cases():
    # Test cases are (x, r, expected)
    cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)



@pytest.mark.parametrize('x, r, expected', [ 
        (0.1,2.2,0.198),
        (0.2,3.4,0.544),
        (0.5,2,0.5)
    ])
def test_f_generic_values(x,r,expected):
    result = f(x, r)
    assert_allclose(result, expected)
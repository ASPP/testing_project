from numpy.testing import assert_allclose

from logistic import f


def test_f():
    # Test cases are (x, r, expected)
    cases = [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5),
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)


def test_f_special_x_values():
    # Test cases are (x, r, expected)
    cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)

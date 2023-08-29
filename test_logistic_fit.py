import math
import pytest

from logistic import iterate_f
from logistic_fit import fit_r


@pytest.mark.parametrize("x0, r, it", [
    (0.3, 3.421, 23),
    (0.1, 3.421, 23),
    (0.3, 1.0, 23),
    (0.3, 3.421, 5),
])
def test_logistic_fit_recover_r(x0, r, it):
    trajectory = iterate_f(it, x0, r)
    fitted_r = fit_r(trajectory)
    assert math.isclose(fitted_r, r)

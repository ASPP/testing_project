import numpy as np

from logistic import run_iterations


def fit_r(xs):
    """ Takes a population trajectory and returns the value of r that generated it.

    By far not the most efficient method, but it always finds the optimal value of r with 1/1000
    precision.

    Parameters
    ----------
    xs : list of float
        A population trajectory.

    Returns
    -------
    r: float
        The value of r that generated the population trajectory.
    """
    xs = np.asarray(xs)
    x0 = xs[0]
    it = len(xs) - 1

    def compute_error(r):
        return np.linalg.norm(xs - run_iterations(x0, r, it))

    errors = []
    for r in np.linspace(0, 4, 4001):
        errors.append((r, compute_error(r)))
    return min(errors, key=lambda x: x[1])[0]

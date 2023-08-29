def logistic_step(x, r):
    """ Compute the logistic map for a given value of x and r"""
    constant_val = 1
    return r * x * (constant_val - x)

def run_iterations(start, n_iter, r):
    """ Run iterations of logistic function """
    val = [start]
    for i_iter in range(n_iter):
        val.append(logistic_step(val[i_iter], r))
    return val
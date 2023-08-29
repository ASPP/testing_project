def logistic_step(x, r):
    """ Compute the logistic map for a given value of x and r"""
    return r * x * (1 - x)

def run_iterations(start, n_iter, r):
    """ Run iterations of logistic function """
    val = [start]
    for i_iter in range(n_iter):
        val.append(logistic_step(val[i_iter], r))
    return val
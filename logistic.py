from plot_logistic import plot_trajectory

def f(x, r):
    """
    Implements the logistic function.
    """
    return r * x * (1 - x)

def iterate_f(x, r, it):
    traj = [x]
    for _ in range(it):
        x = f(x, r)
        traj.append(x)
    return traj

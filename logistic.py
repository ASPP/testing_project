from matplotlib import pyplot as plt


def logistic_step(x, r):
    """ Compute the logistic map for a given value of x and r"""
    constant_val = 1
    return r * x * (constant_val - x)

def run_iterations(start, r, n_iter):
    """ Run iterations of logistic function """
    val = [start]
    for i_iter in range(n_iter):
        val.append(logistic_step(val[i_iter], r))
    return val

if __name__=='__main__':
    start_vals = [i/10 for i in range(1, 6)]
    r = 1.5
    n_iter = 10
    fig_name = "test_trajectory"

    fig, ax = plt.subplots(figsize=(10, 5))
    for i_start_val in start_vals:
        trajectory = run_iterations(i_start_val, r, n_iter)
        ax.plot(list(range(n_iter+1)), trajectory, label=str(i_start_val))
    fig.suptitle(f'Logistic function: r={r}, n_iter={n_iter}')

    fig.savefig(fig_name)
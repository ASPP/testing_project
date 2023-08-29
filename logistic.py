# Your code goes here
def f(x,r):
    """
    compute logistic map
    """
    return r*x*(1-x)


def iterate_f(it, x, r):
    """
    run f for it iterations
    """
    res=[x]
    for i in range(it):
        x=f(x,r)
        res.append(x)
    return res
# Your code goes here
def f(x,r):
    return r*x*(1-x)

def iterate_f(it,x,r):
    results = [x]
    for _ in range(it):
        x = f(x, r)
        results.append(x)
    return results

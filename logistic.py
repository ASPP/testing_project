# Your code goes here
def f(x,r):
    return r * x * (1-x) 

def iterate_f(x,r,it):
    result = []
    for _ in range(it):
        result.append(f(result[-1],r))
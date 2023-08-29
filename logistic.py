def f(x, r):
    return r * x * (1-x)

def iterate_f(it, x, r):
    l = [x]
    for i in range(it):
        x = f(x, r)
        l.append(x)
    return l 

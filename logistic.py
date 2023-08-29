# Your code goes here
def f(x,r):
    return r*x*(1-x)

def iterate_f(x0,r,it):
    iteration_timecourse = []

    iteration_timecourse.append(x0)

    x = f(x0,r)
    iteration_timecourse.append(x)

    for i in range(it-1):
        x = f(x,r)
        iteration_timecourse.append(x)

    return iteration_timecourse 
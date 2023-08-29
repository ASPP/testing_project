
# Your code goes here

def f(x,r):
    return r*x*(1-x)


#print(f(0, 1.1))

def iterate_f(it,x,r):
    time_series=[]
    time_series.append(x)
    for i in range(it):
        x=r*x*(1-x)
        time_series.append(x)
    return time_series

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plot_logistic import plot_trajectory
    plot_trajectory(100,4,0)
    plt.show()

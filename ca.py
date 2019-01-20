import numpy as np
from numpy import array


minim = -5
maxim = 5
search_space = [[-5, 5]] * 3
print("search_space: ", search_space)
pop_size = 10
dimension = 2


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum


def rand_in_bounds(mini, maxi):
    y = mini + ((maxi-mini) * np.random.rand())
    return y


p = rand_in_bounds(minim, maxim)
print("p = ", p)


def random_vector(minmax):
    a = np.zeros([pop_size, dimension])
    print(a)
    print("print: ", np.size(minmax))
    for i in range(np.size(minmax)):
        a = rand_in_bounds(minmax[i][0], minmax[i][1])
    return a


j = random_vector(search_space)
print("j: ", j)

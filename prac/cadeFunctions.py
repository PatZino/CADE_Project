import numpy as np


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum

def function_two(x):
    result2 = 0
    for i in range(len(x)):
        result2 += (x[i] ** 2) - (10 * np.cos((2 * np.pi * x[i]) * np.pi/180)) + 10
    return result2


def function_three(x):
    sum = 0
    for i in range(len(x)):
        sum += i * (x[i]**4)
    return sum


alist = [2, 1, 4, 1, 3]

a = objective_function(alist)
print("output : ", a)

b =function_two(alist)
print("function two", b)

# print(np.sin(30 * (np.pi/180)))











import numpy as np
import random


# alist = [2, 1]
#len(x) = len(alist)


def function_one(x):
    result = 0
    for i in range(len(x)):
        result += x[i]**2
    return result


def function_two(x):
    result = 0
    for i in range(len(x)):
        result += (x[i] ** 2) - (10 * np.cos((2 * np.pi * x[i]) * np.pi/180)) + 10
    return result


def function_three(x):
    result = 0
    for i in range(len(x)):
        result += i * (x[i]**4)
    return result


def function_four(x):
    result = 0
    for i in range(len(x)):
        result += np.absolute(x[i] * np.sin(x[i] * np.pi/180) + (0.1 * x[i]))
    return result


def function_five(x):
    a = 0
    b = 0
    result = 0
    for i in range(len(x)):
        a += (x[i]) ** 2
        b += 0.5 * i * x[i]
        c = b ** 2
        d = b ** 4
        result = a + c + d
    return result


def function_six(x):
    result = 0
    for i in range(len(x)):
        result += (np.floor(x[i] + 0.5)) ** 2
    return result


def function_seven(x):
    result = 0
    for i in range(len(x)):
        result += (np.absolute(x[i])) ** (i + 1)
    return result


def function_eight(x):
    a = 0
    b = 1
    result = 0
    for i in range(len(x)):
        a += np.absolute(x[i])
        b *= np.absolute(x[i])
        result = a + b
    return result


def function_nine(x):
    a = 0
    result = 0
    for i in range(len(x)):
        a += i * (x[i]**4)
        b = random.random()
        result = a + b
    return result


def function_ten(x):
    result = 0
    for i in range(0, len(x) - 1):
        result += (100 * (x[i + 1] - (x[i] ** 2)) ** 2 + (x[i] - 1) ** 2)
    return result


def function_eleven(x):
    a = 0
    b = 0
    result = 0
    for i in range(len(x)):
        a += np.square(x[i])
        b *= (np.cos(2 * np.pi * x[i]) * np.pi/180)
        result += -20 * np.exp(-0.2 * np.sqrt((1/len(x)) * a)) - np.exp((1/len(x)) * b) + 20 + np.e
    return result


def function_twelve(x):
    lists = list()
    for i in range(len(x)):
        lists.append(np.absolute(x[i]))
    result = np.max(lists)
    return result


def function_thirteen(x):
    a = 0
    b = 1
    for i in range(len(x)):
       a += np.square(x[i])
       b *= (np.cos(x[i]/np.sqrt(i)) * np.pi/180)
    result = (1/400) * a - b + 1
    return result


def function_fourteen(x):
    a = 0
    result = 0
    for i in range(len(x)):
        a += x[i] * (np.sin(np.absolute(x[i])) * np.pi/180)
        result = -a + len(x) * 418.9828872743369
    return result


def function_fifteen(x):
    a = 0
    result = 0
    for i in range(1, len(x) + 1):
        for j in range(i):
            a += x[i-1]
        result += np.square(a)
    return result


# b =function_fifteen(alist)
# print("function fifteen: ", b)
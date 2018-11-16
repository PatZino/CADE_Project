import numpy as np


N = 6

A = np.zeros((N-2,N-2))
B = np.zeros((N-2,N-2))
b = np.zeros((N-2))

print("A: \n", A, "\n")
print("B: \n", B, "\n")
print("b: \n", b, "\n")
"""
A = np.array([[5, 3], [1, 2]])
B = np.array([40, 18])
#  use numpy's linear algebra solve function to solve the system
C = np.linalg.solve(A, B)
print("X and Y: ", C)

print("\nMethod 2:\n")
#  find the inverse of A
D = np.linalg.inv(A)
#  find the dot product of the inverse of the coefficient matrix and the results matrix
E = np.dot(D, B)
print("X and Y: ", E)

print("Gaussian Ellimination: \n")


def gaussian(m):
    for col in range(len(m[0])):
        for row in range(col+1, len(m)):
            r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]
    # now backsolve by substitution
    ans =[]
    m.reverse()  # makes it easier to backsolve
    for sol in range(len(m)):
        if sol == 0:
            ans.append(m[sol][-1] / m[sol][-2])
        else:
            inner = 0
            #  substitute in all known coefficients
            for x in range(sol):
                inner += (ans[x]*m[sol][-2-x])
                #  the equation is now reduced to ax + b = c form
                #  solve with (c - b) / a
                ans.append((m[sol][-1]-inner) / m[sol][-sol-2])
    ans.reverse()
    return ans


print(gaussian([[2.0, 4.0, 6.0, 8.0, 10.0, 0.0], [1.0, 3.0, 5.0, 8.0, 3.0, -1.0], [3.0, 8.0, 9.0, 20.0, 3.0, 5.0],
                [4.0, 8.0, 9.0, -2.0, 3.0, 5.0], [5.0, -3.0, 3.0, -2.0, 1.0, 0]]))

"""



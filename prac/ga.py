import random
from math import exp
import numpy
import matplotlib.pyplot as plt

out = []
fit = []
fit_prob = []
newGen = []
currentGene = []
probb_sum = 0

""" Define the count of the population required!!! """
pop_count = 100
val_count = 3  # number of variables in o/p equation


# Begin Genetic Algorithm Function

def gen_alg(inputt):
    gain1 = []
    del gain1[:]  # Clear all the ellements of the gain1 array
    gain1_check = 0

    gain2 = []
    del gain2[:]  # Clear all the ellements of the gain2 array

    gain2_check = 0

    # Generate random numbers between 0 & 1
    a = random.uniform(0.0, 0.9)
    print("a: ", a)

    # Roulette wheel Algorithm
    probb_sum = 0
    for i in range(0, pop_count):
        probb_sum = probb_sum + fit_prob[i]

        if a < probb_sum:
            # comparing random number 'a' with the 'prob_sum' upto the i'th element of the fit_probb array.
            gain1 = inputt[i]
            gain1_check = 1
            break

        if i == (pop_count - 1) and gain1_check == 0:
            gain1 = inputt[i]
            break

    b = random.uniform(0.0, 0.9)
    probb_sum = 0
    for j in range(0, pop_count):
        probb_sum = probb_sum + fit_prob[j]

        if b < probb_sum:
            # comparing random number 'a' with the 'prob_sum' upto the i'th element of the fit_probb array.
            gain2 = inputt[j]
            gain2_check = 1
            break

        if j == (pop_count - 1) and gain2_check == 0:
            gain2 = inputt[j]
            break

    # ~~~ Generating index for crossover variables

    temp1 = int(round(random.uniform(0, 20) / 10))
    temp2 = int(round(random.uniform(0, 20) / 10))

    cross = []  # ~~~ Create crossover array
    del cross[:]  # ~~~ empty the crossover array

    cross = gain1

    cross[temp1] = gain2[temp1]
    cross[temp2] = gain2[temp2]

    # ~~~ possibility of mutated variable generation
    chance = random.uniform(0.00, 0.09)
    if chance < 0.02:
        increment = random.uniform(-0.09, 0.09)
        index = int(round(random.uniform(0, 20) / 10))
        cross[index] = cross[index] + increment

    return cross


# ~~~ End Genetic Algorithm Function


'''
The input 'inputt' we are considering are in the 2D array format, like shown below
inputtArray = {
		[x1-1,x1-2,x1-3.....x1-10]
		[x2-1,x2-2,x2-3.....x2-10]
		[x3-1,x3-2,x3-3.....x3-10]
		.
		.
		.
		[x10-1,x10-2,x10-3.....x10-10]
	     }
where,	input-1 values are ---> (x1-1,x1-2,x1-3.....x1-10) 
	input-2 values are ---> (x2-1,x2-2,x2-3.....x2-10) 
for understanding purposes the elements of the array are viewed as Matrix elements
		x1-1	x1-2	x1-3 ... x1-10
		x2-1	x2-2	x2-3 ... x2-10
		x3-1	x3-2	x3-3 ... x3-10
		.	
		.
		.
		x10-1	x10-2	x10-3 ... x10-10
'''


def main():
    old_gen_count = 0
    global inputt
    inputt = []
    global out
    out = []

    for i in range(0, pop_count):
        inputt.append([])  # creates a new row in a matrix
        for j in range(0, val_count):  # val_count is the number of variables in the o/p equation
            x = random.uniform(0.00, 3.00)
            inputt[i].append(x)  # add j'th element in the row-i

    repeat = 1
    egg = 10
    count = 0
    while (count < 20):

        fit_sum = 0
        i = 0

        # ~~~ Output(y) and Fitness(f) calculation
        out = []
        del out[:]

        fit = []
        del fit[:]

        for i in range(0, pop_count):
            # ~~~ In the below line of code, by mentioning gen[i][0], inputt[i][1] .... and so on I am accessing all
            #  10 elements (in this case) of the i-th row of the matrix. Because elements of
            # the rows are our input values

            a = ((inputt[i][0] - 1) ** 2) + ((inputt[i][1] - 2) ** 2) + ((inputt[i][2] - 3) ** 2) + 1

            out.append(a)
            fit.append(exp(-out[i]))  # Describe fitness function

        fit_sum = sum(fit)  # Claculate fitness sum

        # ~~~ Calculating the Fitness Probability Array
        for p in range(0, pop_count):
            if fit_sum == 0:
                fit_sum = 1
                fit_prob.append(fit[p] / fit_sum)
            else:
                fit_prob.append(fit[p] / fit_sum)

        i = 0
        j = 0

        call = []
        del call[:]

        newGen = []
        del newGen[:]

        # ~~~ Begin: Creating a new generation of inputs
        for i in range(0, pop_count):
            call = gen_alg(inputt)  # ~~~ Calling the genetic algorithm function
            newGen.append(call)
        # ~~~ End: Created new generation of inputs

        inputt = newGen

        count += 1

    j = 0 
    i = 0 

    a = []
    b = []
    c = []

    A = []

    for i in range(0, pop_count):
        a.append(inputt[i][0])
        b.append(inputt[i][1])
        c.append(inputt[i][2])

    A.append(numpy.mean(a))
    A.append(numpy.mean(b))
    A.append(numpy.mean(c))

    print("done")
    print(A)
    print("\n")


main()
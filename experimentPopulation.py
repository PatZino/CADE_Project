import numpy as np


dimension = 3
bounds = [(-100, 100)] * dimension
max_gen = 200
population_size = 40
acceptedNumber = round(population_size * 0.20)
elites = 1
mutation_factor = 0.5
crossover_probability = 0.5


def expPop():
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(lower_bound - upper_bound)
    experimentpop = lower_bound + population * difference
    return experimentpop


generalPopulation = list()
for i in range(population_size):
    generalPopulation = expPop()

print("generalPopulation\n", generalPopulation)

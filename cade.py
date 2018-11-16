import numpy as np


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i] ** 2
    return sum / len(x)


# generate initial parameters of each techniques
bounds = [(-5, 5)] * 3
population_size = 10
max_gen = 2
dimension = 3

# Generate initial overall shared population
population = np.random.rand(population_size, dimension)
lower_bound, upper_bound = np.asarray(bounds).T
difference = np.fabs(lower_bound - upper_bound)
initial_population = lower_bound + population * difference
print("initial pop: \n", initial_population)

# evaluate the initial population
fitness = np.asarray([objective_function(ind) for ind in initial_population])
print("\nfitness : ", fitness)
best_index = np.argmin(fitness)
best = initial_population[best_index]
print("\nbest : ", best)

# compute initial participation ratio
initial_participation_ratio = population_size / 2
print("\ninitial_participation_ratio: ", initial_participation_ratio)
CA_population = np.zeros(int(initial_participation_ratio))
print("CA_population : \n", CA_population)
DE_population = np.zeros(int(initial_participation_ratio))
print("DE_population : \n", DE_population)

# produce a subset of new individuals by each technique T according to the participation ratio
Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                           population_size, replace=False), :]
print("Auxiliary_population : \n", Auxiliary_population)
print("\nCA_population : ")
for p in range(int(initial_participation_ratio)):
    CA_population = Auxiliary_population[p]
    print(CA_population)
print("\nDE_population : ")
for p in range(int(initial_participation_ratio), population_size):
    DE_population = Auxiliary_population[p]
    print(DE_population)












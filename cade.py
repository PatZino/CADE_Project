import numpy as np


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum / len(x)


# generate initial parameters of each techniques
bounds = [(-5, 5)] * 3
print("bounds: ", bounds)


def cade(objective_function, bounds, dimension=3, max_gen=2, population_size=12):
    # Generate initial overall shared population
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference
    print("initial pop: \n", initial_population)

    # compute initial participation ratio
    initial_participation_ratio = int(population_size / 2)
    print("\ninitial_participation_ratio: ", initial_participation_ratio, "\n")

    # evaluate the initial population
    fitness = np.asarray([objective_function(ind) for ind in initial_population])
    print("\nfitness : ", fitness)
    var1 = fitness[:initial_participation_ratio]
    print("var1", var1)
    best_index = np.argmin(fitness)
    best = initial_population[best_index]
    print("\nbest : ", best)

    CA_population = np.zeros(int(initial_participation_ratio))
    # print("CA_population : \n", CA_population)

    DE_population = np.zeros(int(initial_participation_ratio))
    # print("DE_population : \n", DE_population)

    # produce a subset of new individuals by each technique T according to the participation ratio
    Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                               population_size, replace=False), :]
    # print("Auxiliary_population : \n", Auxiliary_population)

    CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    # print(CA_population)
    s = []
    fitness_CA = np.asarray([objective_function(ind2) for ind2 in CA_population])
    print("\nCA population fitness : ", fitness_CA, "\n")

    DE_population = np.asarray(
        [Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    # print(DE_population)

    fitness_DE = np.asarray([objective_function(ind2) for ind2 in DE_population])
    print("\nDE population fitness : ", fitness_DE, "\n")

    for i in range(max_gen):

        for j in range(len(CA_population)):

            CA_fitness = np.asarray([objective_function(ind2) for ind2 in CA_population])
            # print("\nCA population fitness : ", CA_fitness, "\n")

            DE_fitness = np.asarray([objective_function(ind2) for ind2 in DE_population])

            # computing number of trial vector for CA technique

            if CA_fitness[j] < DE_fitness[j]:
                t = 1
                s.append(t)
            else:
                t = 0
                s.append(t)
        print("s: ", s, "\n")

    yield CA_population, DE_population


for j, k in cade(objective_function, bounds):
    print( "\nCA_population\n", j, "\nDE_population\n", k)

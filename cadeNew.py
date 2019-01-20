import numpy as np


# generate initial parameters of each techniques
bounds = [(-100, 100)] * 5


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum


def de(DE_population, bounds, dimension=5, max_gen=5, cr=0.7, mf=0.8):
    # print("DE_population \n", DE_population)
    population = np.random.rand(len(DE_population), dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    new_population = []
    fitness = np.asarray([objective_function(ind) for ind in DE_population])
    for i in range(max_gen):
        print("\nnew population:")
        for j in range(len(DE_population)):
            indices = [index for index in range(len(DE_population)) if index != j]

            x0, x1, x2 = DE_population[np.random.choice(indices, 3, replace=False)]

            mutant_vector = np.clip(x0 + mf * (x1 - x2), 0, 1)

            crossover = np.random.rand(dimension) < cr

            if not np.any(crossover):
                crossover[np.random.randint(0, dimension)] = True

            trial_vector = np.where(crossover, mutant_vector, population[j])

            new_population = lower_bound + trial_vector * difference
            print(new_population)

            new_fitness = objective_function(new_population)

            if new_fitness < fitness[j]:
                fitness[j] = new_fitness
                population[j] = trial_vector
        yield population


def cade(objective_function, bounds, dimension=5, max_gen=5, population_size=10, cr=0.7, mf=0.8):
    # Generate initial overall shared population
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference
    print("initial pop: \n", initial_population)

    # evaluate the initial population
    fitness = np.asarray([objective_function(ind) for ind in initial_population])
    print("\nfitness : ", fitness)
    var1 = fitness[:5]
    print(var1)
    best_index = np.argmin(fitness)
    best = initial_population[best_index]
    print("\nbest : ", best)

    # compute initial participation ratio
    initial_participation_ratio = population_size / 2
    print("\ninitial_participation_ratio: ", initial_participation_ratio, "\n")

    # CA_population = np.zeros(int(initial_participation_ratio))
    # print("CA_population : \n", CA_population)

    # DE_population = np.zeros(int(initial_participation_ratio))
    # print("DE_population : \n", DE_population)

    # produce a subset of new individuals by each technique T according to the participation ratio
    Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                               population_size, replace=False), :]
    # print("Auxiliary_population : \n", Auxiliary_population)

    CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    # print(CA_population)
    s = []
    fitness_CA = np.asarray([objective_function(ind2) for ind2 in CA_population])
    # print("\nCA population fitness : ", fitness_CA, "\n")

    DE_population = np.asarray(
        [Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    print("DE_population \n", DE_population)

    fitness_DE = np.asarray([objective_function(ind2) for ind2 in DE_population])
    # print("\nDE population fitness : ", fitness_DE, "\n")

    for pop in de(DE_population, bounds):
        print("pop: ", pop, "\n")

    for i in range(max_gen):

        for j in range(len(CA_population)):

            CA_fitness = np.asarray([objective_function(ind2) for ind2 in CA_population])
            # print("\nCA population fitness : ", CA_fitness, "\n")

            DE_fitness = np.asarray([objective_function(ind2) for ind2 in de(DE_population, bounds)])

            # computing number of trial vector for CA technique

            if CA_fitness[j] < DE_fitness[j]:
                t = 1
                s.append(t)
            else:
                t = 0
                s.append(t)
        print("s: ", s, "\n")

        yield CA_population, DE_population


for c, k in cade(objective_function, bounds):
    print("\nCA_population\n", c, "\nDE_population\n", k)



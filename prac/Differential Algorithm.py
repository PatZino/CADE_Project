import numpy as np
import functionSelection
# import functionDisplay


func = functionSelection.funct

max_gen = 200
mutation_factor = 0.5
crossover_probability = 0.5
dimension = 3
bounds = [(-10, 10)] * dimension
population_size = 40
# ---------------------------------DE---------------------------------------------------------


def differential_evolution(func, mutation_factor, crossover_probability):
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(lower_bound - upper_bound)
    differential_evolution_pop = lower_bound + population * difference
    # print("differential_evolution_pop", differential_evolution_pop, "\n")
    fitness = np.asarray([func(ind) for ind in differential_evolution_pop])
    # print("fitness : ", fitness)
    best_index = np.argmin(fitness)
    best = differential_evolution_pop[best_index]

    for i in range(max_gen):
        for j in range(population_size):
            indices = [index for index in range(population_size) if index != j]

            x0, x1, x2 = population[np.random.choice(indices, 3, replace=False)]

            mutant_vector = np.clip(x0 + mutation_factor * (x1 - x2), 0, 1)

            crossover = np.random.rand(dimension) < crossover_probability

            if not np.any(crossover):
                crossover[np.random.randint(0, dimension)] = True

            trial_vector = np.where(crossover, mutant_vector, population[j])
            # print("trial vector:\n", trial_vector)

            new_population = lower_bound + trial_vector * difference
            # print("new_population:\n", new_population)

            new_fitness = func(new_population)
            # print("new_fitness:\n", new_fitness)

            if new_fitness < fitness[j]:
                fitness[j] = new_fitness
                population[j] = trial_vector
                if new_fitness < fitness[best_index]:
                    best_index = j
                    best = new_population

        yield best, fitness[best_index]


# ----------------------------------End DE----------------------------------------------------

# Evolved DE Population using DE technique
for best, fitness in differential_evolution(func, mutation_factor,
                                                 crossover_probability):
    print("best: ", best, "fitness: ", fitness)

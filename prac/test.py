import numpy as np
import random


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum


# generate initial parameters of each techniques
bounds = [(-5, 5)] * 3
# print("bounds: ", bounds)
max_gen = 2
population_size = 12
dimension = 3
acceptedNumber = round(population_size * 0.20)
elites = 1
mutation_factor = 0.8
crossover_probability = 0.7


def cade(objective_function, max_gen, bounds, population_size, dimension):
    # Generate initial overall shared population
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference
    # print("initial pop: \n", initial_population)

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

    # produce a subset of new individuals by each technique T according to the participation ratio
    Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                               population_size, replace=False), :]
    # print("Auxiliary_population : \n", Auxiliary_population)

    CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    # print(CA_population)

# -------------------------CA-----------------------------------------------------------

    def rand_in_bounds(min, max):
        return min + ((max - min) * random.random())

    def random_vector(bounds):
        individuals = list()
        for i in range(len(bounds)):
            rand = rand_in_bounds(bounds[i][0], bounds[i][1])
            individuals.append(rand)
        return individuals

    def mutate(candidate, beliefs, bounds):
        v = list()
        for i in range(len(candidate["individuals"])):
            x = rand_in_bounds(beliefs["normative"][i][0], beliefs["normative"][i][1])
            if x < bounds[i][0]: x = bounds[i][0]
            if x > bounds[i][1]: x = bounds[i][1]
            v.append(x)
        return {"individuals": v}

    def binary_tournament(population_size, candidates, elites):
        new_population = list()
        candidates.sort(key=lambda c: c["fitness"])
        for i in range(elites):
            survivor = candidates.pop(0)
            new_population.append(survivor)
        for i in range(population_size):
            c1 = random.randint(0, len(candidates) - 1)
            c2 = random.randint(0, len(candidates) - 1)
            while c1 == c2:
                c2 = random.randint(0, len(candidates) - 1)
            if candidates[c1]["fitness"] < candidates[c2]["fitness"]:
                survivor = candidates.pop(c1)
            else:
                survivor = candidates.pop(c2)
            new_population.append(survivor)
        return new_population

    def initialize_population(population_size, bounds):
        population = list()
        for i in range(population_size):
            d = {"individuals": random_vector(bounds)}
            # d = {"individuals": initial_population}
            population.append(d)
        return population

    def beliefspaceInitialization(bounds):
        beliefspace = {}
        beliefspace["situational"] = None
        beliefspace["normative"] = list()
        for i in range(len(bounds)):
            beliefspace["normative"].append(list(bounds[i]))
        return beliefspace

    def situationalBeliefspace(beliefspace, best):
        currentBest = beliefspace["situational"]
        # print("current best situational = ", currentBest)
        if currentBest is None or best["fitness"] < currentBest["fitness"]:
            beliefspace["situational"] = best

    def normativeBeliefspace(beliefspace, acc):
        for i in range(len(beliefspace["normative"])):
            acc_min = min(acc, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][0] = acc_min["individuals"][i]
            acc_max = max(acc, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][1] = acc_max["individuals"][i]

    def culturalSearch(max_gens, bounds, population_size, acceptedNumber, elites):
        # initialize
        population = initialize_population(population_size, bounds)
        print(population)
        exit(1);
        beliefspace = beliefspaceInitialization(bounds)
        fitness_data = list()

        # evaluate
        for i in population:
            # i["fitness"] = objective_function(i["individuals"])
            i["fitness"] = objective_function(i["individuals"])


        # get current best
        best = min(population, key=lambda i: i["fitness"])

        # update situational knowledge
        situationalBeliefspace(beliefspace, best)

        # add situational_belief to fitness_data
        fitness_data.append(best["fitness"])

        # evolution:
        for gen in range(max_gens):
            # create new generation
            children = list()
            for i in range(population_size):
                new_child = mutate(population[i], beliefspace, bounds)
                children.append(new_child)
            # evaluate new generation
            for i in children:
                i["fitness"] = objective_function(i["individuals"])

            # survivor selection
            population = binary_tournament(population_size, children + population, elites)
            # print("new population = ", population)
            # get new current best
            best = min(population, key=lambda i: i["fitness"])

            # update situational knowledge
            situationalBeliefspace(beliefspace, best)

            # add situational_belief to fitness_data
            fitness_data.append(best["fitness"])

            # update normative knowledge
            population.sort(key=lambda i: i["fitness"])
            acccepted = population[:acceptedNumber]
            normativeBeliefspace(beliefspace, acccepted)

        return beliefspace["situational"]
# -------------------------------------End CA------------------------------------------------

    # Evolved CA Population using CA technique
    for i in range(population_size):
        Evolved_CA = culturalSearch(max_gen, bounds, population_size, acceptedNumber, elites)
        print("Evolved CA", Evolved_CA)

# ---------------------------------DE---------------------------------------------------------
    def differential_evolution(objective_function, bounds, max_gen, mutation_factor,
                               crossover_probability, population_size):
        population = np.random.rand(population_size, dimension)
        lower_bound, upper_bound = np.asarray(bounds).T
        difference = np.fabs(lower_bound - upper_bound)
        initial_population = lower_bound + population * difference
        # print("initial pop: ", initial_population)
        fitness = np.asarray([objective_function(ind) for ind in initial_population])
        # print("fitness : ", fitness)
        best_index = np.argmin(fitness)
        best = initial_population[best_index]
        # print("best : ", best)
        for i in range(max_gen):
            for j in range(population_size):
                indices = [index for index in range(population_size) if index != j]

                x0, x1, x2 = population[np.random.choice(indices, 3, replace=False)]

                mutant_vector = np.clip(x0 + mutation_factor * (x1 - x2), 0, 1)

                crossover = np.random.rand(dimension) < crossover_probability

                if not np.any(crossover):
                    crossover[np.random.randint(0, dimension)] = True

                trial_vector = np.where(crossover, mutant_vector, population[j])

                new_population = lower_bound + trial_vector * difference
                # print("new population: ", new_population)

                new_fitness = objective_function(new_population)

                if new_fitness < fitness[j]:
                    fitness[j] = new_fitness
                    population[j] = trial_vector
                    if new_fitness < fitness[best_index]:
                        best_index = j
                        best = new_population
            yield best, fitness[best_index]

# ----------------------------------End DE----------------------------------------------------

    # Evolved DE Population using DE technique
    for best, fitness in differential_evolution(objective_function, bounds, max_gen, mutation_factor,
                                                crossover_probability, population_size):
        print("best = ", best, "fitness = ", fitness)

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
        # print("s: ", s, "\n")

    yield CA_population, DE_population


for j, k in cade(objective_function, max_gen, bounds, population_size, dimension):
    print( "\nCA_population\n", j, "\nDE_population\n", k)

import numpy as np
import random
import functionSelection


# ------------------------ Test Functions --------------------------------------
func = functionSelection.funct
# -----------------------End Test Functions ------------------------------------

# ---------------generate initial parameters of each techniques ----------------
dimension = 3
bounds = [(-5, 5)] * dimension
max_gen = 60
population_size = 50
acceptedNumber = round(population_size * 0.20)
elites = 1
mutation_factor = 0.8
crossover_probability = 0.7
# ------------------ End initial parameters -------------------------------------


# ------------------------ CADE ------------------------------------------------


def cade(func, max_gen, bounds, population_size, dimension):
    # Generate initial overall shared population
    delta = 0
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference
    print("initial pop: \n", initial_population)

    # compute initial participation ratio
    initial_participation_ratio = int(population_size / 2)

    # evaluate the initial population
    fitness = np.asarray([func(ind) for ind in initial_population])
    print("\nfitness : \n", fitness, "\n\n")

    # var1 = fitness[:initial_participation_ratio]
    # print("var1", var1)
    # best_index = np.argmin(fitness)
    # best = initial_population[best_index]
    # print("\nbest : ", best)

    # produce a subset of new individuals by each technique T according to the participation ratio
    Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                               population_size, replace=False), :]
    # print("Auxiliary_population : \n", Auxiliary_population)

    CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    # print(CA_population)
    CAfitness = np.asarray([func(ind) for ind in CA_population])
    print("\ninitial_CAfitness : \n", CAfitness, "\n\n")

    DE_population = np.asarray(
        [Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    DEfitness = np.asarray([func(ind) for ind in DE_population])
    print("\ninitial_DEfitness : \n", DEfitness, "\n\n")

# -------------------------CA-----------------------------------------------------------

    def rand_in_bounds(min, max):
        return min + ((max - min) * random.random())

    def mutate(candidate, beliefs, bounds):
        v = list()
        for i in range(len(candidate["individuals"])):
            x = rand_in_bounds(beliefs["normative"][i][0], beliefs["normative"][i][1])
            if x < bounds[i][0]: x = bounds[i][0]
            if x > bounds[i][1]: x = bounds[i][1]
            v.append(x)
        return {"individuals": v}

    def selection(population_size, candidates, elites):
        new_population = list()
        candidates.sort(key=lambda b: b["fitness"])
        for i in range(elites):
            withstand = candidates.pop(0)
            new_population.append(withstand)
        for i in range(population_size):
            canda = random.randint(0, len(candidates) - 1)
            candb = random.randint(0, len(candidates) - 1)
            while canda == candb:
                candb = random.randint(0, len(candidates) - 1)
            if candidates[canda]["fitness"] < candidates[candb]["fitness"]:
                withstand = candidates.pop(canda)
            else:
                withstand = candidates.pop(candb)
            new_population.append(withstand)
        return new_population

    def format_population(initial_population):
        formatted_population  = list()
        for i in initial_population:
            d = {"individuals": i}
            formatted_population.append(d)
        return formatted_population

    def beliefspaceInitialization(bounds):
        beliefspace = {}
        beliefspace["normative"] = list()
        beliefspace["situational"] = None
        for i in range(len(bounds)):
            beliefspace["normative"].append(list(bounds[i]))
        return beliefspace

    def situationalBeliefspace(beliefspace, best):
        currentBest = beliefspace["situational"]
        # print("current best situational = ", currentBest)
        if currentBest is None or best["fitness"] < currentBest["fitness"]:
            beliefspace["situational"] = best

    def normativeBeliefspace(beliefspace, accepted):
        for i in range(len(beliefspace["normative"])):
            acceptedMin = min(accepted, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][0] = acceptedMin["individuals"][i]
            acceptedMax = max(accepted, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][1] = acceptedMax["individuals"][i]

    def culturalAlgorithm(initial_population, bounds,  acceptedNumber, elites):
        # initial population
        population = format_population(initial_population)
        population_size = len(population)
        beliefspace = beliefspaceInitialization(bounds)
        fitnessData = list()
        gens = max_gen
        ser = list()
        cer = list()

        # evaluate the population
        for i in population:
            i["fitness"] = func(i["individuals"])

        best = min(population, key=lambda i: i["fitness"])
        situationalBeliefspace(beliefspace, best)
        fitnessData.append(best["fitness"])

        for k in range(gens):
            individualsPop = list()
            for i in range(population_size):
                newIndividualsPop = mutate(population[i], beliefspace, bounds)
                individualsPop.append(newIndividualsPop)
            for i in individualsPop:
                i["fitness"] = func(i["individuals"])
            population = selection(population_size, individualsPop + population, elites)
            # current best
            best = min(population, key=lambda i: i["fitness"])

            # situational knowledge update
            situationalBeliefspace(beliefspace, best)
            fitnessData.append(best["fitness"])

            population.sort(key=lambda i: i["fitness"])
            acccepted = population[:acceptedNumber]
            # print("\n\naccepted: ", acccepted, "\n\n")
            normativeBeliefspace(beliefspace, acccepted)

            if k == (gens - 2):
                ser = beliefspace["situational"]["individuals"]

            if k == (gens - 1):
                cer = beliefspace["situational"]["individuals"]

        return ser, cer, beliefspace["situational"]["individuals"], best["fitness"]
# -------------------------------------End CA------------------------------------------------

    # Evolved CA Population using CA technique
    CA = list()
    CA_pop = list()
    ser = list()
    cer = list()
    for i in range(len(CA_population)):
        initial_ca_population = CA_population.tolist()
        ser1, cer1, Evolved_CA, Evolved_CA_Fitness = culturalAlgorithm(initial_ca_population, bounds, acceptedNumber, elites)
        CA.append(Evolved_CA_Fitness)
        CA_pop.append(Evolved_CA)
        ser.append(ser1)
        cer.append(cer1)
    print("Evolved CA Fitness\n", CA)
    print("\nEVOLVED CA POPULATION \n", CA_pop)
    print("\nCA Second to the last generation \n", ser)
    print("\nCA last generation \n", cer)
    print("\n\n")

# ---------------------------------DE---------------------------------------------------------
    def differential_evolution(func, initial_de_population, mutation_factor,
                               crossover_probability):

        population_size = len(initial_de_population)
        max_gen = population_size
        fitness = np.asarray([func(ind) for ind in initial_de_population])
        # print("fitness : ", fitness)
        best_index = np.argmin(fitness)
        best = initial_de_population[best_index]
        best2 = initial_de_population[best_index]
        daho = list()
        baho = list()

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

                new_fitness = func(new_population)

                if new_fitness < fitness[j]:
                    fitness[j] = new_fitness
                    population[j] = trial_vector
                    if new_fitness < fitness[best_index]:
                        best_index = j
                        best = new_population

                if new_fitness < fitness[j] and i == (max_gen - 2):
                    fitness[j] = new_fitness
                    population[j] = trial_vector
                    if new_fitness < fitness[best_index]:
                        best_index = j
                        best2 = new_population

            yield daho, best2, best, fitness[best_index]

# ----------------------------------End DE----------------------------------------------------

    # Evolved DE Population using DE technique
    DE = list()
    DE_pop = list()
    dah = list()
    bah = list()
    for daho, best2, best, fitness in differential_evolution(func, DE_population, mutation_factor,
                                                crossover_probability):
        DE_pop.append(best)
        DE.append(fitness)
        # print("dah: \n", daho)
        # print("bah: \n", baho)
        # dah.append(daho)
        bah.append(best2)
    print("Evolved DE Fitness: \n", DE, "\n")
    print("Evolved DE Pop: \n", DE_pop, "\n")
    # print("dah: \n", dah)
    print("bah: \n", bah)
    print("\n\n")

# ---------------------------------------------------------------------------------------------
    new_cade_pop = np.concatenate((CA_pop, DE_pop), axis=0) # new cade population

# ----------------------- loop in the cade algorithm -------------------------------------------
    for m in range(max_gen):
        fitness_ser = np.asarray([func(ind) for ind in ser])
        fitness_cer = np.asarray([func(ind) for ind in cer])
        fitness_del = np.asarray([func(ind) for ind in DE_pop])
        fitness_del2 = np.asarray([func(ind) for ind in bah])
        s = list()
        d = list()
        numTrialVectorCA = 0
        numTrialVectorDE = 0

        # Number of trial vector for CA algorithm
        for j in range(len(ser)):
            if fitness_cer[j] < fitness_ser[j]:
                t = 1
                s.append(t)
                numTrialVectorCA += t
            else:
                t = 0
                s.append(t)

        print("\ns: ", s, "\n")
        print("number of successful Trial Vector for CA = ", numTrialVectorCA)

        # Number of trial vector for DE algorithm
        for j in range(len(DE_population)):
            if fitness_del[j] < fitness_del2[j]:
                t = 1
                d.append(t)
                numTrialVectorDE += t
            else:
                t = 0
                d.append(t)

        print("\nd: ", d, "\n")
        print("number of successful Trial Vector for DE = ", numTrialVectorDE)

        if numTrialVectorCA > numTrialVectorDE:
            quality_func = (numTrialVectorCA - numTrialVectorDE) / numTrialVectorCA
        else:
            quality_func = (numTrialVectorDE - numTrialVectorCA) / numTrialVectorDE

        print("quality function: ", quality_func)

        participation_ratio = quality_func * population_size
        print("participation ratio: ", participation_ratio)

        cade_pop = np.concatenate((CA_pop, DE_pop), axis=0)

    yield new_cade_pop


for k in cade(func, max_gen, bounds, population_size, dimension):
    print("\n\nCADE population\n", k)

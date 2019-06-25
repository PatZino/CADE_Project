import numpy as np
import random
import functionSelection
from scipy.stats import  cauchy
from matplotlib import pyplot as plt

# ------------------------ Test Functions --------------------------------------
func = functionSelection.funct

xin = [2, 3, 5]
TestFunction = func(xin)
print("TestFunction: ", TestFunction)
# -----------------------End Test Functions ------------------------------------

# ---------------generate initial parameters of each techniques ----------------
dimension = 3
bounds = [(-5, 5)] * dimension
max_gen = 200
population_size = 40
acceptedNumber = round(population_size * 0.20)
elites = 1
mutation_factor = np.random.standard_cauchy(1)
crossover_probability = np.random.normal(0.5, 0.1, 1)
# ------------------ End initial parameters -------------------------------------


# ------------------------ CADE ------------------------------------------------


def cade(func, max_gen, bounds, population_size, dimension):
    # Generate initial overall shared population
    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference
    print("initial pop: \n", initial_population)
    quality_func = 0

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
    # print("\ninitial_CAfitness : \n", CAfitness, "\n\n")

    DE_population = np.asarray(
        [Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    print("Initial DE_population \n", DE_population)
    DEfitness = np.asarray([func(ind) for ind in DE_population])
    print("\ninitial_DEfitness : \n", DEfitness, "\n\n")

# -------------------------CA-----------------------------------------------------------

    def rand_in_bounds(min, max):
        return min + ((max - min) * random.random())

    def mutate(candidate, beliefs, bounds):
        v = list()
        for i in range(len(candidate["individuals"])):
            x = rand_in_bounds(beliefs["normative"][i][0], beliefs["normative"][i][1])
            if x < bounds[i][0]:
                x = bounds[i][0]
            if x > bounds[i][1]:
                x = bounds[i][1]
            v.append(x)
        return {"individuals": v}

    def mutateTopo():
        y = np.random.normal(0, 1, dimension)
        return y

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
        beliefspace["topographical"] = list()
        for i in range(len(bounds)):
            beliefspace["normative"].append(list(bounds[i]))
        return beliefspace

    def situationalBeliefspace(beliefspace, best):
        currentBest = beliefspace["situational"]
        # print("current best situational = ", currentBest)

        if currentBest is None or best["fitness"] < currentBest["fitness"]:
            beliefspace["situational"] = best

    def topographicalBeliefspace(best, best2):
        for i in range(dimension):
            if best["fitness"] < best2["fitness"]:
                y = mutateTopo()
            else:
                y = np.random.normal(0, 1, dimension)
                # print("y: ", y)
        # print("length: ", len(beliefspace["topographical"]))

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

        # evaluate the population using the obj()
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
            # print("new population: \n", population)

            # current best
            best2 = min(population, key=lambda i: i["fitness"])
            # print(k, "best CA : ", best2)

            # update situational knowledge source
            situationalBeliefspace(beliefspace, best2)
            fitnessData.append(best["fitness"])

            # update topographic knowledge source
            topographicalBeliefspace(best, best2)

            # update Normative knowledge source
            population.sort(key=lambda i: i["fitness"])
            acccepted = population[:acceptedNumber]
            # print("\n\naccepted: ", acccepted, "\n\n")
            normativeBeliefspace(beliefspace, acccepted)

            if k == (gens - 2):
                ser = beliefspace["situational"]["individuals"]

            if k == (gens - 1):
                cer = beliefspace["situational"]["individuals"]

        return ser, cer, beliefspace["situational"]["individuals"], best2["fitness"]
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
    # print("Evolved CA Fitness\n", CA)
    # print("\nEVOLVED CA POPULATION \n", CA_pop)
    # print("\nCA Second to the last generation \n", ser)
    # print("\nCA last generation \n", cer)
    # print("\n\n")

# ---------------------------------DE---------------------------------------------------------
    def differential_evolution(func, initial_de_population, mutation_factor,
                               crossover_probability):

        population_size = len(initial_de_population)
        max_gen = population_size
        # print("initial_de_population", initial_de_population, "\n")
        fitness = np.asarray([func(ind) for ind in initial_de_population])
        # print("fitness : ", fitness)
        best_index = np.argmin(fitness)
        best = initial_de_population[best_index]
        best2 = initial_de_population[best_index]
        daho = list()
        baho = list()
        # print("crossover probability: ", crossover_probability)
        # print("mutation factor: ", mutation_factor)

        for i in range(max_gen):
            for j in range(population_size):
                indices = [index for index in range(population_size) if index != j]
                print("indices: ", indices)

                x0, x1, x2 = initial_de_population[np.random.choice(indices, 3, replace=False)]
                print("x0, x1, x2: ", x0, x1, x2)

                mutant_vector = np.clip(x0 + mutation_factor * ((best - x0) + (x1 - x2)), 0, 1)  # DE/Current-to-best/2
                print("mutant vector:\n", mutant_vector)

                crossover = np.random.rand(dimension) < crossover_probability

                if not np.any(crossover):
                    crossover[np.random.randint(0, dimension)] = True

                trial_vector = np.where(crossover, mutant_vector, initial_de_population[j])
                print("trial vector:\n", trial_vector)

                new_population = lower_bound + trial_vector * difference
                print("new_population:\n", new_population)

                new_fitness = func(new_population)
                print("new_fitness:\n", new_fitness)

                if new_fitness < fitness[j]:
                    fitness[j] = new_fitness
                    initial_de_population[j] = trial_vector
                    if new_fitness < fitness[best_index]:
                        best_index = j
                        best = new_population


                if new_fitness < fitness[j] and i == (max_gen - 2):
                    fitness[j] = new_fitness
                    initial_de_population[j] = trial_vector
                    if new_fitness < fitness[best_index]:
                        best_index = j
                        best2 = new_population
                print(i, ":population \n", initial_de_population)
            mutation_factor = np.random.standard_cauchy(1)
            # print("mutation factor: ", mutation_factor)
            crossover_probability = np.random.normal(0.5, 0.1, 1)

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
    # print("Evolved DE Fitness: \n", DE, "\n")
    # print("Evolved DE Pop: \n", DE_pop, "\n")
    # print("dah: \n", dah)
    # print("bah: \n", bah)
    # print("\n\n")

# ---------------------------------------------------------------------------------------------

    updated_cade_pop = np.concatenate((DE_pop, CA_pop), axis=0)
    fitness_updated_cade = np.asarray([func(ind) for ind in updated_cade_pop])

    number_of_runs = 0
    diffVetorCA = 0
    diffVetorDE = 0
# ----------------------- loop in the cade algorithm -------------------------------------------

    while number_of_runs < 50:
        fitness_ser = np.asarray([func(ind) for ind in ser])
        fitness_cer = np.asarray([func(ind) for ind in cer])
        fitness_del = np.asarray([func(ind) for ind in DE_pop])
        print("fitness_del:", fitness_del)
        fitness_del2 = np.asarray([func(ind) for ind in bah])
        print("fitness_del2:", fitness_del2)
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

        # print("\ns: ", s, "\n")
        # print("number of successful Trial Vector for CA = ", numTrialVectorCA)

        # Number of trial vector for DE algorithm
        for j in range(len(DE_pop)):
            if fitness_del[j] < fitness_del2[j]:
                t = 1
                d.append(t)
                numTrialVectorDE += t
            else:
                t = 0
                d.append(t)

        # print("\nd: ", d, "\n")
        # print("number of successful Trial Vector for DE = ", numTrialVectorDE)

        for j in range(numTrialVectorCA):
            diffVetorCA += (fitness_ser[j]-fitness_cer[j])
        # print("diffVetorCA: ", diffVetorCA)

        for j in range(numTrialVectorCA):
            diffVetorDE += (fitness_del[j]-fitness_del2[j])
        # print("diffVetorDE: ", diffVetorDE)

        # determining the best technique and calcolating quality function

        if numTrialVectorCA == 0 and numTrialVectorDE == 0:
            break
        elif numTrialVectorCA > numTrialVectorDE:
            quality_func = (diffVetorCA - diffVetorDE) / diffVetorCA
        else:
            quality_func = (diffVetorDE - diffVetorCA) / diffVetorDE

        # print("quality function: ", quality_func)
        if quality_func == 1:
            break

        participation_ratio = np.int(quality_func * population_size)
        print("participation ratio: ", participation_ratio)


        updated_cade_pop = np.concatenate((DE_pop, CA_pop), axis=0)  # new cade population
        fitness_updated_cade = np.asarray([func(ind) for ind in updated_cade_pop])  # updated fitness value
        updated_Auxiliary_pop = updated_cade_pop[np.random.choice(updated_cade_pop.shape[0],
                                                                  population_size, replace=False), :]

        if numTrialVectorCA > numTrialVectorDE:
            updated_DE_pop = np.asarray(
                [updated_Auxiliary_pop[p] for p in range(int(participation_ratio), population_size)])
            updated_CA_pop = np.asarray([updated_Auxiliary_pop[p] for p in range(int(participation_ratio))])
        else:
            updated_DE_pop = np.asarray([updated_Auxiliary_pop[p] for p in range(int(participation_ratio))])
            updated_CA_pop = np.asarray(
                [updated_Auxiliary_pop[p] for p in range(int(participation_ratio), population_size)])

        # Evolved CA Population using CA technique
        CA = list()
        CA_pop = list()
        ser = list()
        cer = list()
        for i in range(len(updated_CA_pop)):
            updated_ca_population = updated_CA_pop.tolist()
            ser1, cer1, Evolved_CA, Evolved_CA_Fitness = culturalAlgorithm(updated_ca_population, bounds,
                                                                           acceptedNumber, elites)
            CA.append(Evolved_CA_Fitness)
            CA_pop.append(Evolved_CA)
            ser.append(ser1)
            cer.append(cer1)
        # print("Evolved CA Fitness\n", CA)
        # print("\nEVOLVED CA POPULATION \n", CA_pop)
        # print("\nCA Second to the last generation \n", ser)
        # print("\nCA last generation \n", cer)
        # print("\n\n")

        # Evolved DE Population using DE technique
        DE = list()
        DE_pop = list()
        dah = list()
        bah = list()
        for daho, best2, best, fitness in differential_evolution(func, updated_DE_pop, mutation_factor,
                                                                 crossover_probability):
            DE_pop.append(best)
            DE.append(fitness)
            # print("dah: \n", daho)
            # print("bah: \n", baho)
            # dah.append(daho)
            bah.append(best2)
        # print("Evolved DE Fitness: \n", DE, "\n")
        # print("Evolved DE Pop: \n", DE_pop, "\n")
        # ---print("dah: \n", dah)
        # print("bah: \n", bah)
        # print("\n\n")

        # cade_pop = np.concatenate((CA_pop, DE_pop), axis=0)

    yield updated_cade_pop, fitness_updated_cade


for p, q in cade(func, max_gen, bounds, population_size, dimension):
    # print("\n\nCADE population ", p, "\n\nfitness = ", q)
    print("\n\nCADE population ", p, "\n\nfitness = ", q)

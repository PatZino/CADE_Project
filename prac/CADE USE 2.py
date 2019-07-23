import numpy as np
import random
import functionSelection


# ------------------------ Test Functions --------------------------------------
func = functionSelection.funct

# xin = [2, 3, 5]
# TestFunction = func(xin)
# print("TestFunction: ", TestFunction)

# -----------------------End Test Functions ------------------------------------

# ---------------generate initial parameters of each techniques ----------------
dimension = 3
bounds = [(-10, 10)] * dimension
max_gen = 200
population_size = 80
acceptedNumber = round(population_size * 0.20)
els = 1
mutation_factor = 0.5
crossover_probability = 0.5
# ------------------ End initial parameters -------------------------------------


# ------------------------ CADE ------------------------------------------------

def cade(func, max_gen, bounds, population_size, dimension):
    # Generate initial overall shared population

    population = np.random.rand(population_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    initial_population = lower_bound + population * difference

    # print("initial pop: \n", initial_population)
    quality_func = 0

    # compute initial participation ratio
    initial_participation_ratio = int(population_size / 2)

    # evaluate the initial population
    fitness = np.asarray([func(ind) for ind in initial_population])
    # print("\nfitness : \n", fitness, "\n\n")

    FEs_best_index = np.argmin(fitness)
    FEs_best = initial_population[FEs_best_index]
    FEs_fitness = fitness[FEs_best_index]
    print("FEs_best: ", FEs_best)
    print("FEs_fitness: ", FEs_fitness)

    # produce a subset of new individuals by each technique T according to the participation ratio
    Auxiliary_population = initial_population[np.random.choice(initial_population.shape[0],
                                                               population_size, replace=False), :]
    # print("Auxiliary_population : \n", Auxiliary_population)

    # CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    CA_population = np.asarray([Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    # print(CA_population)
    CAfitness = np.asarray([func(ind) for ind in CA_population])
    # print("\ninitial_CAfitness : \n", CAfitness, "\n\n")

    # DE_population = np.asarray(
        # [Auxiliary_population[p] for p in range(int(initial_participation_ratio), population_size)])
    DE_population = np.asarray(
        [Auxiliary_population[p] for p in range(int(initial_participation_ratio))])
    # print("Initial DE_population \n", DE_population)
    DEfitness = np.asarray([func(ind) for ind in DE_population])
    # print("\ninitial_DEfitness : \n", DEfitness, "\n\n")

    # ---------------------------------DE---------------------------------------------------------
    def differential_evolution(func, initial_de_population, mutation_factor,
                               crossover_probability):

        population_size = len(initial_de_population)
        fitness = np.asarray([func(ind) for ind in initial_de_population])
        # print("fitness : ", fitness)
        best_index = np.argmin(fitness)
        best = initial_de_population[best_index]
        bah = ()

        for i in range(max_gen):
            for j in range(population_size):
                indices = [index for index in range(population_size) if index != j]
                # print("indices: ", indices)

                x0, x1, x2 = population[np.random.choice(indices, 3, replace=False)]
                # print("x0, x1, x2: ", x0, x1, x2)

                #  DE/Current-to-best/2
                # mutant_vector = np.clip(x0 + mutation_factor * ((best - x0) + (x1 - x2)), 0, 1)

                mutant_vector = np.clip(x0 + mutation_factor * (x1 - x2), 0, 1)
                # print("mutant vector:\n", mutant_vector)

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

                if [max_gen - 2]:
                    bah = best
                    mutation_factor = np.random.standard_cauchy(1)
                    # print("mutation factor: ", mutation_factor)
                    crossover_probability = np.random.normal(0.5, 0.1, 1)

            yield bah, best, fitness[best_index]

    # ----------------------------------End DE----------------------------------------------------

    # Evolved DE Population using DE technique
    DE = list()
    DE_pop = list()
    bah1 = list()
    for bah, best, fitness in differential_evolution(func, DE_population, mutation_factor,
                                                     crossover_probability):
        DE_pop.append(best)
        DE.append(fitness)
        bah1.append(bah)
        # print("best: ", best, "fitness: ", fitness)

    # print("DE: \n", DE)

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

    def mutateTopo(c):
        y = np.random.normal(0, 1, c)
        return y

    def selection(population_size, candidates, els):
        new_population = list()
        candidates.sort(key=lambda b: b["fitness"])
        for i in range(els):
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

    def topographicalBeliefspace(gen, best, best2, dim):
        if i != gen:
            if best2["fitness"] < best["fitness"]:
                y = mutateTopo(dim)
            else:
                y = mutateTopo(dim)
        else:
            if best2["fitness"] < best["fitness"]:
                y = mutateTopo(dim)
            else:
                y = mutateTopo(dim)
                # print("y: ", y)
        return y
        # print("length: ", len(beliefspace["topographical"]))

    def normativeBeliefspace(beliefspace, accepted):
        for i in range(len(beliefspace["normative"])):
            acceptedMin = min(accepted, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][0] = acceptedMin["individuals"][i]
            acceptedMax = max(accepted, key=lambda v: v["individuals"][i])
            beliefspace["normative"][i][1] = acceptedMax["individuals"][i]

    def culturalAlgorithm(initial_population, bounds,  acceptedNumber, els):
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
            population = selection(population_size, individualsPop + population, els)
            # print("new population: \n", population)

            # current best
            best2 = min(population, key=lambda i: i["fitness"])
            # print(k, "best CA : ", best2)

            # update situational knowledge source
            situationalBeliefspace(beliefspace, best2)
            fitnessData.append(best2["fitness"])

            # update topographic knowledge source
            topographicalBeliefspace(gens, best, best2, dimension)

            # update Normative knowledge source
            population.sort(key=lambda i: i["fitness"])
            acccepted = population[:acceptedNumber]
            # print("\n\naccepted: ", acccepted, "\n\n")
            normativeBeliefspace(beliefspace, acccepted)

            if k == (gens - 2):
                ser = beliefspace["situational"]["individuals"]

            if k == (gens - 1):
                cer = beliefspace["situational"]["individuals"]

        return ser, cer, beliefspace["situational"]["individuals"], beliefspace["situational"]["fitness"]
# -------------------------------------End CA------------------------------------------------

    # Evolved CA Population using CA technique
    CA = list()
    CA_pop = list()
    ser = list()
    cer = list()
    # CA_fit_list = list()
    for i in range(max_gen):
        initial_ca_population = CA_population.tolist()
        ser1, cer1, Evolved_CA, Evolved_CA_Fitness = culturalAlgorithm(initial_ca_population, bounds, acceptedNumber, els)
        CA.append(Evolved_CA_Fitness)
        # CA_fit_list.append(CA)
        CA_pop.append(Evolved_CA)
        ser.append(ser1)
        cer.append(cer1)
# ---------------------------------------------------------------------------------------------

    # fit = [0] * max_gen
    # fit = [0] * max_gen
    # fitness_updated_cade = fit
    fitness_updated_cade = [0] * max_gen
    updated_cade_popt = list()
    updated_cade_poop = list()
    updated_cade = list()
    DE_list = list(DE)
    print("DE fitness list\n", DE_list)
    CA_fit_list = CA
    print("CA_fit_list\n", CA_fit_list)
    DE_popt = DE_pop
    # print("DE list: \n", DE_list)
    CA_popt = CA_pop
    # print("CA_fit_list:\n", CA_fit_list)

    for p in range(max_gen):
        if DE_list[p] < CA_fit_list[p]:
            fitness_updated_cade[p] = DE_list[p]
            # idx = fitness_updated_cade.index(DE_list[p])
            idx = DE_list.index(DE_list[p])
            # DE_popt[p] = DE_pop[idx]
            updated_cade = updated_cade_popt.append(list(DE_pop)[idx])
        else:
            fitness_updated_cade[p] = CA_fit_list[p]
            # idx = fitness_updated_cade.index(CA_fit_list[p])
            idx = CA_fit_list.index(CA_fit_list[p])
            # CA_popt[p] = CA_pop[idx]
            updated_cade = updated_cade_popt.append(list(CA_pop)[idx])

    # updated_cade_pop = np.array(updated_cade_popt)
    updated_cade = np.array(updated_cade)
    fitness_updated_cade = np.array(fitness_updated_cade)
    updated_cade_poop = np.concatenate((DE_pop, CA_pop), axis=0)
    updated_cade_pop = updated_cade
    # fitness_updated_cade = np.asarray([func(ind) for ind in updated_cade_pop])
    # fitness_updated_cade = np.array(fitness_updated_cades)
    # fitness_updated_cades = [func(ind) for ind in updated_cade_pop]
    # fitness_updated_cade = list(fitness_updated_cades)
    best_index = np.argmin(fitness_updated_cade)
    print("best index: ", best_index)
    # cade_best = updated_cade_pop[best_index]
    # print("cade_best: ", cade_best)
    cade_best_fitness = fitness_updated_cade[best_index]
    # cade_best_fitness = min(fitness_updated_cade)
    print("cade_best_fitness: ", cade_best_fitness, "\n\n")

    number_of_runs = 0
    diffVetorCA = 0
    diffVetorDE = 0
# ----------------------- loop in the cade algorithm -------------------------------------------

    while number_of_runs < 50:
        fitness_ser = np.asarray([func(ind) for ind in ser])
        fitness_cer = np.asarray([func(ind) for ind in cer])
        fitness_del = np.asarray([func(ind) for ind in DE_pop])
        fitness_del2 = np.asarray([func(ind) for ind in bah1])

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

        # determining the best technique and calcolating quality function

        if numTrialVectorCA > numTrialVectorDE:
            quality_func = (numTrialVectorCA - numTrialVectorDE) / numTrialVectorCA
        elif numTrialVectorCA == 0 and numTrialVectorDE == 0:
            break
        else:
            quality_func = (numTrialVectorDE - numTrialVectorCA) / numTrialVectorDE
        print("quality function: ", quality_func)
        if quality_func == 1:
            break

        participation_ratio = np.int(quality_func * population_size)
        print("participation ratio: ", participation_ratio)


        updated_cade_pop = updated_cade
        fitness_updated_cade = np.array(fitness_updated_cade)
        # fitness_updated_cades = np.asarray([func(ind) for ind in updated_cade_pop])
        # fitness_updated_cades = [func(ind) for ind in updated_cade_pop]
        # fitness_updated_cade = list(fitness_updated_cades)
        best_index = np.argmin(fitness_updated_cade)
        # cade_best = updated_cade_pop[best_index]
        # print("cade_best: ", cade_best)
        cade_best_fitness = fitness_updated_cade[best_index]
        # cade_best_fitness = min(fitness_updated_cade)
        # print("cade_best_fitness: ", cade_best_fitness, "\n\n")
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
        updated_cade_poop = list()
        # CA_fit_list = list()
        for i in range(len(updated_CA_pop)):
            updated_ca_population = updated_CA_pop.tolist()
            ser1, cer1, Evolved_CA, Evolved_CA_Fitness = culturalAlgorithm(updated_ca_population, bounds,
                                                                           acceptedNumber, els)
            CA.append(Evolved_CA_Fitness)
            # CA_fit_list.append(CA)
            CA_pop.append(Evolved_CA)
            ser.append(ser1)
            cer.append(cer1)

        # Evolved DE Population using DE technique
        DE = list()
        DE_pop = list()
        bah1 = list()
        # updated_DE_pop = list()
        for bah, best, fitness in differential_evolution(func, updated_DE_pop, mutation_factor,
                                                         crossover_probability):
            DE_pop.append(best)
            DE.append(fitness)
            bah1.append(bah)
            # DE_pop.append(updated_DE_pop)

        updated_cade_poop = np.concatenate((CA_pop, DE_pop), axis=0)

    yield updated_cade_pop, fitness_updated_cade


for p, q in cade(func, max_gen, bounds, population_size, dimension):
    print("\n\nfitness = ", q)


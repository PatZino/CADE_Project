import numpy as np
import random
import functionSelection
# import functionDisplay

# ------------------------ Test Functions --------------------------------------
func = functionSelection.funct

# -----------------------End Test Functions ------------------------------------

# ---------------generate initial parameters of each techniques ----------------
dimension = 3
bounds = [(-10, 10)] * dimension
max_gen = 200
population_size = 40
acceptedNumber = round(population_size * 0.20)
elites = 1
# ------------------ End initial parameters -------------------------------------

# -------------------------CA-----------------------------------------------------------


population = np.random.rand(population_size, dimension)
lower_bound, upper_bound = np.asarray(bounds).T
difference = np.fabs(lower_bound - upper_bound)
ca_population= lower_bound + population * difference


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


def format_population(cultural_algorithm_pop):
    formatted_population  = list()
    for i in cultural_algorithm_pop:
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


def normativeBeliefspace(beliefspace, accepted):
    for i in range(len(beliefspace["normative"])):
        acceptedMin = min(accepted, key=lambda v: v["individuals"][i])
        beliefspace["normative"][i][0] = acceptedMin["individuals"][i]
        acceptedMax = max(accepted, key=lambda v: v["individuals"][i])
        beliefspace["normative"][i][1] = acceptedMax["individuals"][i]


def culturalAlgorithm(cultural_algorithm_pop, bounds,  acceptedNumber, elites):
    population = format_population(cultural_algorithm_pop)
    population_size = len(population)
    beliefspace = beliefspaceInitialization(bounds)
    fitnessData = list()

    # evaluate the population using the obj()
    for i in population:
        i["fitness"] = func(i["individuals"])

    best = min(population, key=lambda i: i["fitness"])
    situationalBeliefspace(beliefspace, best)
    fitnessData.append(best["fitness"])

    for k in range(max_gen):
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

        # update situational knowledge source
        situationalBeliefspace(beliefspace, best2)
        fitnessData.append(best2["fitness"])

        # update topographic knowledge source
        topographicalBeliefspace(max_gen, best, best2, dimension)

        # update Normative knowledge source
        population.sort(key=lambda i: i["fitness"])
        acccepted = population[:acceptedNumber]
        # print("\n\naccepted: ", acccepted, "\n\n")
        normativeBeliefspace(beliefspace, acccepted)

    return beliefspace["situational"]["individuals"], beliefspace["situational"]["fitness"]
# -------------------------------------End CA------------------------------------------------

# Evolved CA Population using CA technique
CA_fit = list()
CA_pop = list()
CA_fit_list = list()
CA_pop_list = list()
for i in range(max_gen):
    cultural_algorithm_pop = ca_population.tolist()
    Evolved_CA, Evolved_CA_fitness = culturalAlgorithm(ca_population, bounds, acceptedNumber, elites)
    CA_pop.append(Evolved_CA)
    CA_pop_list = list(CA_pop)
    CA_fit.append(Evolved_CA_fitness)
    CA_fit_list = list(CA_fit)

# print("CA Population: ", "\n", CA_pop)
print("CA_pop_list\n", CA_pop_list)
print("\n\n")
# print("CA Fitness: ", "\n", CA_fit)
print("\n\n")
print("CA_fit_list", "\n", CA_fit_list)


ca_best_fitness = min(CA_fit_list)
print("ca_best_fitness: ", ca_best_fitness)

minposition = CA_fit_list.index(min(CA_fit_list))
print("minimum position: ", minposition)

best_individual = CA_pop_list[minposition]
print("best individual: ", best_individual)




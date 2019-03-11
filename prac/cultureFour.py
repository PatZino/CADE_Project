import test
import random


def objective_function(individuals):
    v = 0.0
    for i in individuals:
        v +=i ** 2.0
    return v


def rand_in_bounds(min, max):
    return min + ((max-min) * random.random())


bounds = [(-5, 5)] * 3


def random_vector(minmax):
    individuals = list()
    for i in range(len(minmax)):
        rand = rand_in_bounds(minmax[i][0], minmax[i][1])
        individuals.append(rand)
    return individuals


def mutate_with_inf(candidate, beliefs, minmax):
    v = list()
    for i in range(len(candidate["individuals"])):
        x = rand_in_bounds(beliefs["normative"][i][0], beliefs["normative"][i][1])
        if x < minmax[i][0]: x = minmax[i][0]
        if x > minmax[i][1]: x = minmax[i][1]
        v.append(x)
    return {"individuals": v}


def binary_tournament(pop_size, contestants, elites):
    new_population = list()
    # keep top contestants
    contestants.sort(key = lambda c: c["fitness"])
    for i in range(elites):
        survivor = contestants.pop(0)
        new_population.append(survivor)
    # select two different contestants
    for i in range(pop_size):
        c1 = random.randint(0, len(contestants)-1)
        c2 = random.randint(0, len(contestants)-1)
        while c1 == c2:
            c2 = random.randint(0, len(contestants)-1)
        # take the better of the two
        if contestants[c1]["fitness"] < contestants[c2]["fitness"]:
            survivor = contestants.pop(c1)
        else:
            survivor = contestants.pop(c2)
        new_population.append(survivor)
    return new_population


def initialize_population(pop_size, search_space):
    population = list()
    for i in range(pop_size):
        d = {"individuals": random_vector(search_space)}
        population.append(d)
    return population


def initialize_beliefspace(bounds):
    belief_space = {}
    belief_space["situational"] = None
    belief_space["normative"] = list()
    for i in range(len(bounds)):
        belief_space["normative"].append(list(bounds[i]))
    return belief_space


def update_beliefspace_situational(belief_space, best):
    curr_best = belief_space["situational"]
    # print("current best situational = ", curr_best)
    if curr_best is None or best["fitness"] < curr_best["fitness"]:
        belief_space["situational"] = best


def update_beliefspace_normative(belief_space, acc):
    for i in range(len(belief_space["normative"])):
        acc_min = min(acc, key = lambda v: v["individuals"][i])
        belief_space["normative"][i][0] = acc_min["individuals"][i]
        acc_max = max(acc, key = lambda v: v["individuals"][i])
        belief_space["normative"][i][1] = acc_max["individuals"][i]


def cultural_search(max_gens, bounds, pop_size, num_accepted, elites):
    # initialize
    population = initialize_population(pop_size, bounds)
    belief_space = initialize_beliefspace(bounds)
    fitness_data = list()

    # evaluate
    for c in population:
        c["fitness"] = objective_function(c["individuals"])

    # get current best
    best = min(population, key=lambda c: c["fitness"])

    # update situational knowledge
    update_beliefspace_situational(belief_space, best)

    # add situational_belief to fitness_data
    fitness_data.append(best["fitness"])

    # evolution:
    for gen in range(max_gens):
        # create new generation
        children = list()
        for c in range(pop_size):
            new_child = mutate_with_inf(population[c], belief_space, bounds)
            children.append(new_child)
        # evaluate new generation
        for c in children:
            c["fitness"] = objective_function(c["individuals"])

        # survivor selection
        population = binary_tournament(pop_size, children + population, elites)
        print("new population = ", population)
        # get new current best
        best = min(population, key=lambda c: c["fitness"])

        # update situational knowledge
        update_beliefspace_situational(belief_space, best)

        # add situational_belief to fitness_data
        fitness_data.append(best["fitness"])

        # update normative knowledge
        population.sort(key=lambda c: c["fitness"])
        acccepted = population[:num_accepted]
        update_beliefspace_normative(belief_space, acccepted)

    return belief_space["situational"], fitness_data


if __name__ == "__main__":
    # problem configuration
    problem_size = 3
    # bounds = create_search_space(problem_size)

    # algorithm configuration
    trial_runs = 5
    max_gens = 20
    pop_size = 5
    num_accepted = round(pop_size * 0.20)
    elites = 1

    # execute the cultural algorithm
    cultural_runs = list()
    for i in range(trial_runs):
        best, fitness_data = cultural_search(max_gens, bounds, pop_size, num_accepted, elites)
        print("best= ", best, "\n", "fitness_data",  fitness_data, "\n\n")
        cultural_runs.append({"best": best, "fitness_data": fitness_data})

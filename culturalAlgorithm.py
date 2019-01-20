import numpy as np


def objective_function(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]**2
    return sum


"""
def accept(fitness, pop):
    for i in range(1, len(pop)):
        # print("len(pop): ", len(pop))
        # ("i: ", i)
        # print("fitness[i-1]: ", fitness[i-1], "\t fitness[i]: ", fitness[i])
        if fitness[i-1] < fitness[i]:
            # pprint("fitness[i-1] < fitness[i]: ", fitness[i-1] < fitness[i])
            belief_space[i-1] = pop[i-1]
        else:
            belief_space[i-1] = pop[i]
    return belief_space
"""

g = 20
pop_size = 10
bounds = [(-100, 100)] * 3
# belief_space = [0]*pop_size


# ---------------------------Accept()---------------------------------------
# using roulette wheel to define the accept function
def get_probability_list(pop_fitness):
    fitness = pop_fitness
    total_fit = float(sum(fitness))
    # print("total fitness: ", total_fit, "\n")
    relative_fitness = [f/total_fit for f in fitness]
    # print("relative_fitness: ", relative_fitness, "\n")
    probabilities = [sum(relative_fitness[:i+1])
                     for i in range(len(relative_fitness))]
    return probabilities


def roulette_wheel_pop(population, probabilities, number):
    chosen = []
    for n in range(number):
        r = np.random.random()
        # print("r: ", r)
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    return chosen
# -------------------------------Accept() End---------------------------------------------


# -------------------------------Update()-------------------------------------------------
# update belief space situational
def situational_knowledge(belief_space):
    y = []

    fitness = np.asarray([objective_function(ind2) for ind2 in belief_space])
    best_index = np.argmin(fitness)
    gbest = belief_space[best_index]
    print("\nbest : \n", gbest, "\n")

    for j in range(len(belief_space)):
        if fitness[j] < fitness[best_index]:
            y = np.random.rand(j, best_index)
            print("y1: ", y)
        elif fitness[j] > fitness[best_index]:
            y = np.random.rand(best_index, j)
            print("y2: ", y)
        else:
            y = np.random.rand(j)
            print("y3: ", y)
    return y
# --------------------------------Update() End--------------------------------------------


# -------------------------------Main()---------------------------------------------------
def ca(objective_function, bounds, dimension=3):
    # Generate initial overall shared population
    population = np.random.rand(pop_size, dimension)
    lower_bound, upper_bound = np.asarray(bounds).T
    difference = np.fabs(upper_bound - lower_bound)
    population_space = lower_bound + population * difference
    # print("initial pop: \n", population_space)

    fitness_CA = np.asarray([objective_function(ind2) for ind2 in population_space])
    # print("\nCA population fitness : ", fitness_CA, "\n")

    belief_space = roulette_wheel_pop(population_space, get_probability_list(fitness_CA), pop_size)
    # print("belief space: \n", belief_space, "\n")

    for i in range(g):
        for j in range(pop_size):
            d = j
    return belief_space


print("believe space: \n", ca(objective_function, bounds))


t = situational_knowledge(ca(objective_function, bounds))
print("t: ", t)

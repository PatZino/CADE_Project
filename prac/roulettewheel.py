import numpy as np


population = [[0,0,1,0,0,0,1],[0,1,0,0,0,0,0]]
population_fitness_dictionary = {"0,0,1,0,0,0,1": 1.245, "0,1,0,0,0,0,0":1.658}
number = 8


def get_probability_list():
    fitness = population_fitness_dictionary.values()
    total_fit = float(sum(fitness))
    relative_fitness = [f/total_fit for f in fitness]
    probabilities = [sum(relative_fitness[:i+1])
                     for i in range(len(relative_fitness))]
    return probabilities


def roulette_wheel_pop(population, probabilities, number):
    chosen = []
    for n in range(number):
        #print("population: ", population)
        r = np.random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    return chosen


result = roulette_wheel_pop(population, get_probability_list(), number)
print("result: ", result)

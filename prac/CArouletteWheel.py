import numpy as np


population = [[46.94361761, 17.48479729, 47.49432797], [-9.98047683, -12.78904719, -52.69542499],
               [-70.92093287,  61.52653135, -14.3317462], [34.90565828, 0.68760669, -81.4917581],
               [14.85534724,  -6.67935006,  78.43931518], [-81.38012176,  54.29722495,  -4.45279312],
               [72.93282153, -45.51604331, 37.10654901], [-8.12693965, -88.37470751,  41.75972508],
               [58.10432125, -49.89561983,  1.62970611], [-50.44582831, -62.64793805, 26.6667423]]

print("population: \n", population, "\n")

number = len(population)
print("population size: ", number, "\n")
population_fitness_dictionary = {}

CA_population_fitness = [12733.31728144, 17583.33993048, 13114.69935403, 13955.55854452,
                         11461.78502385, 11004.52959447, 15218.07413066,  6393.92153009,
                         15821.92456317,  9208.44005827]
print("population fitness: \n", CA_population_fitness, "\n")


def get_probability_list():
    fitness = CA_population_fitness
    total_fit = float(sum(fitness))
    print("total fitness: ", total_fit, "\n")
    relative_fitness = [f/total_fit for f in fitness]
    print("relative_fitness: ", relative_fitness, "\n")
    probabilities = [sum(relative_fitness[:i+1])
                     for i in range(len(relative_fitness))]
    return probabilities


print("probabilities: \n", get_probability_list(), "\n")


def roulette_wheel_pop(population, probabilities, number):
    chosen = []
    for n in range(number):
        r = np.random.random()
        print("r: ", r)
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    return chosen


chosen_pop = roulette_wheel_pop(population, get_probability_list(), number)
print("chosen population: ", chosen_pop, "\n")

sade = 5**2
print("sade: ", sade)
print(round(45.8))
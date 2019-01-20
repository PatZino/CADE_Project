import numpy as np


population = [[46.94361761, 17.48479729, 47.49432797], [-9.98047683, -12.78904719, -52.69542499],
               [-70.92093287,  61.52653135, -14.3317462], [34.90565828, 0.68760669, -81.4917581],
               [14.85534724,  -6.67935006,  78.43931518], [-81.38012176,  54.29722495,  -4.45279312],
               [72.93282153, -45.51604331, 37.10654901], [-8.12693965, -88.37470751,  41.75972508],
               [58.10432125, -49.89561983,  1.62970611], [-50.44582831, -62.64793805, 26.6667423]]

"""
population_fitness_dictionary = {"46.94361761, 17.48479729, 47.49432797": 4765.13256047,
                                 "-9.98047683, -12.78904719, -52.69542499": 3039.977461,
                                 "-70.92093287,  61.52653135, -14.3317462": 9020.69172888,
                                 "34.90565828, 0.68760669, -81.4917581": 7859.78442017,
                                 "14.85534724,  -6.67935006,  78.43931518": 6418.02122456,
                                 "-81.38012176,  54.29722495,  -4.45279312": 9590.74022133,
                                 "72.93282153, -45.51604331, 37.10654901": 8767.80263491,
                                 "-8.12693965, -88.37470751,  41.75972508": 9620.01071416,
                                 "58.10432125, -49.89561983,  1.62970611": 5868.34096844,
                                 "-50.44582831, -62.64793805, 26.6667423": 7180.66088097}
"""

number = len(population)
print("number: ", number)
population_fitness_dictionary = {}

CA_population_fitness = [12733.31728144, 17583.33993048, 13114.69935403, 13955.55854452,
                         11461.78502385, 11004.52959447, 15218.07413066,  6393.92153009,
                         15821.92456317,  9208.44005827]

for i in range(number):
    y = {'population[i]': CA_population_fitness[i]}
    population_fitness_dictionary.update(y)
print("population_fitness_dictionary: ", population_fitness_dictionary)


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

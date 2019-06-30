def parameters():
    dimension = 3
    bounds = [(-100, 100)] * dimension
    max_gen = 200
    population_size = 40
    acceptedNumber = round(population_size * 0.20)
    elites = 1
    mutation_factor = 0.5
    crossover_probability = 0.5
    return dimension, bounds, max_gen, population_size,acceptedNumber, elites, mutation_factor, crossover_probability

def dim():
    dimension = 3
    return dimension


def bound():
    bounds = [(-100, 100)] * dim.dimension
    return bounds
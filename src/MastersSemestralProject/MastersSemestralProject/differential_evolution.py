import numpy as np
import random
import collections

def de(fitness_func, input_population, iterations, type = 'min'):
    F = 0.5 # [0,2]
    CR = 0.9 # [0,1]
    N = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    population = list(map(lambda params: (params, fitness_func(params)), input_population))

    def append_min(population, new_params, old_params):
        population.append(new_params if new_params[1] < params[1] else params)
    def append_max(population, new_params, old_params):
        population.append(new_params if new_params[1] > params[1] else params)
    append_better = append_min if type == 'min' else append_max

    def get_best_min(population):
        return min(population, key=lambda x: x[1])[0]
    def get_best_max(population):
        return max(population, key=lambda x: x[1])[0]
    get_best = get_best_min if type == 'min' else get_best_max

    def get_three_random_agents(i):
        result = set()
        while len(result) < 3:
            j = np.random.randint(N)
            if i != j:
                result.add(j)
        return result

    # DE/rand/1/exp
    def compute_param(x, j):
        r = np.random.uniform()
        return a[j] + F * (b[j] - c[j]) if r < CR or j == R else x

    for i in range(iterations):
        print("iteration=", i)
        new_population = []
        for i, params in enumerate(population):
            indexes = get_three_random_agents(i)
            a,b,c = map(lambda i: population[i][0], indexes)
            R = random.randrange(0, D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params[0])], dtype)
            new_params = (new_params, fitness_func(new_params))

            append_better(new_population, new_params, params)

        population = new_population
        
    return get_best(population)
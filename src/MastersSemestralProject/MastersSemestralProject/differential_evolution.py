import numpy as np
import random
import collections

class DifferentialEvolution(object):
    def __init__(self, fitness_func):
        self.F = 0.5 # [0,2]
        self.CR = 0.9 # [0,1]
        self.fitness = fitness_func

    def run(self, input_population, iterations):
        N = len(input_population)
        D = len(input_population[0])
        dtype = input_population[0].dtype

        population = list(map(lambda params: (params, self.fitness(params)), input_population))

        def get_three_random_agents(i):
            result = set()
            while len(result) < 3:
                j = np.random.randint(N)
                if i != j:
                    result.add(j)
            return result

        def compute_param(x, j):
            r = np.random.uniform()
            return a[j] + self.F * (b[j] - c[j]) if r < self.CR or j == R else x

        for i in range(iterations):
            print("iteration=", i)
            new_population = []
            for i, params in enumerate(population):
                indexes = get_three_random_agents(i)
                a,b,c = map(lambda i: population[i][0], indexes)
                R = random.randrange(0, D)

                new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params[0])], dtype)
                new_params = (new_params, self.fitness(new_params))

                #new_population.append(new_params if new_params[1] < params[1] else params)
                new_population.append(new_params if new_params[1] > params[1] else params)

            population = new_population
        
        #return min(population, key=lambda x: x[1])[0]
        return max(population, key=lambda x: x[1])[0]
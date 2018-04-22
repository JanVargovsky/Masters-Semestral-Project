import numpy as np
import random
import collections

class DifferentialEvolution(object):
    def __init__(self, fitness_func):
        self.F = 0.5 # [0,2]
        self.CR = 0.9 # [0,1]
        self.fitness = fitness_func

    def run(self, input_population, iterations):
        population = [list(p) for p in input_population]
        N = len(population)
        D = len(population[0])
        dtype = input_population[0].dtype

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

        for _ in range(iterations):
            new_population = []
            for i, params in enumerate(population):
                indexes = get_three_random_agents(i)
                a,b,c = map(lambda i: population[i], indexes)
                R = random.randrange(0, D)

                new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params)], dtype)
                new_population.append(new_params if self.fitness(new_params) < self.fitness(params) else params)

            population = new_population
        
        return min(population, key=self.fitness)
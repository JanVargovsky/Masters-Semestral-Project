import numpy as np
import random

class DifferentialEvolution(object):
    def __init__(self, fitness_func):
        self.F = 0.5 # [0,2]
        self.CR = 0.9 # [0,1]
        self.fitness = fitness_func

    def run(self, population, iterations):
        N = len(population)
        D = len(population[0])

        for _ in range(iterations):
            new_population = []
            for i, params in enumerate(population):
                #indexes = random.sample(filter(lambda x: x != i, range(0, N)), 3)
                indexes = random.sample(range(0, N), 3)
                a,b,c = map(lambda i: population[i], indexes)
                R = random.randrange(0, D)

                def compute_param(x, j):
                    r = np.random.uniform()
                    return a[j] + self.F * (b[j] - c[j]) if r < self.CR or j == R else x

                new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params)], params.dtype)
                new_population.append(new_params if self.fitness(new_params) < self.fitness(params) else params)

            population = new_population
        
        return min(population, key=self.fitness)
import unittest
import numpy as np
from DifferentialEvolution import DifferentialEvolution

class DifferentialEvolutionTest(unittest.TestCase):
    def test_de(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        de = DifferentialEvolution(fitness_func)
        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de.run(population, 100)
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

if __name__ == '__main__':
    unittest.main()

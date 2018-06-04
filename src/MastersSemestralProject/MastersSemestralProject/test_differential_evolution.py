import unittest
import numpy as np
from differential_evolution import de, de_jade_with_archive, de_shade

class DifferentialEvolutionTest(unittest.TestCase):
    def test_de_find_min(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de(fitness_func, population, 100, 'min')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

    def test_de_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de(fitness_func, population, 100, 'max')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

    def test_de_jade_with_archive_find_min(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de_jade_with_archive(fitness_func, population, 100, 'min')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

    def test_de_jade_with_archive_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de_jade_with_archive(fitness_func, population, 100, 'max')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

    def test_de_shade_find_min(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de_shade(fitness_func, population, 100, 'min')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

    def test_de_shade_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        actual_params = de_shade(fitness_func, population, 100, 'max')
        actual_fitness = fitness_func(actual_params)

        self.assertAlmostEqual(expected_fitness, actual_fitness, 8)
        [self.assertAlmostEqual(exp, act, 4) for exp, act in zip(expected_params, actual_params)]

if __name__ == '__main__':
    unittest.main()

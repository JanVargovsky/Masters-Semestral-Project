import unittest
import numpy as np
from math import isclose
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

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de(fitness_func, population, end_condition, 'min')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

    def test_de_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de(fitness_func, population, end_condition, 'max')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

    def test_de_jade_with_archive_find_min(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de_jade_with_archive(fitness_func, population, end_condition, 'min')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

    def test_de_jade_with_archive_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de_jade_with_archive(fitness_func, population, end_condition, 'max')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

    def test_de_shade_find_min(self):
        def fitness_func(params):
            return sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de_shade(fitness_func, population, end_condition, 'min')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

    def test_de_shade_find_max(self):
        def fitness_func(params):
            return -sum(map(lambda x: x ** 2, params))

        NP = 50
        D = 5
        population = np.random.sample((NP, D)) * 2 - 1

        expected_params = np.zeros(D)
        expected_fitness = 0.0

        end_condition = lambda fitness, g: isclose(fitness, 0, abs_tol=1e-10) or g >= 150
        result = de_shade(fitness_func, population, end_condition, 'max')

        np.testing.assert_almost_equal(result.fitness, expected_fitness, 10)
        np.testing.assert_array_almost_equal(result.params, expected_params, 5)
        self.assertLessEqual(result.generations, 150)

if __name__ == '__main__':
    unittest.main()

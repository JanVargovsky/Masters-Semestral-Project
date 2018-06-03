import numpy as np
import random
import collections
from scipy.stats import cauchy

def _append_min(population, new_params, old_params):
    is_better = new_params[1] < old_params[1]
    population.append(new_params if is_better else old_params)
    return is_better
def _append_max(population, new_params, old_params):
    is_better = new_params[1] > old_params[1]
    population.append(new_params if is_better else old_params)
    return is_better
def _append_better_func(type):
    return _append_min if type == 'min' else _append_max

def _get_best_min(population):
    return min(population, key=lambda x: x[1])
def _get_best_max(population):
    return max(population, key=lambda x: x[1])
def _get_best_func(type):
    return _get_best_min if type == 'min' else _get_best_max

# TODO: make all stragegies as a separate functions that may be reusable

def de(fitness_func, input_population, iterations, type):
    F = 0.5 # [0,2]
    CR = 0.9 # [0,1]
    N = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    population = list(map(lambda params: (params, fitness_func(params)), input_population))
    append_better = _append_better_func(type)
    get_best = _get_best_func(type)

    def get_three_random_agents(i):
        result = set()
        while len(result) < 3:
            j = np.random.randint(N)
            if i != j:
                result.add(j)
        return result
    """
    Computes param based on the 'rand/1' strategy
    """
    def compute_param(x, j):
        r = np.random.uniform()
        return a[j] + F * (b[j] - c[j]) if r < CR or j == R else x

    for g in range(iterations):
        print("iteration=", g)
        new_population = []
        for i, params in enumerate(population):
            indexes = get_three_random_agents(i)
            a,b,c = map(lambda i: population[i][0], indexes)
            R = random.randrange(0, D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params[0])], dtype)
            new_params = (new_params, fitness_func(new_params))

            append_better(new_population, new_params, params)

        population = new_population
        
    return get_best(population)[0]

def de_jade_with_archive(fitness_func, input_population, iterations, type):
    NP = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    C = 0.8 # [0,1]
    u_CR = 0.5
    u_F = 0.5
    A = list()

    population = list(map(lambda params: (params, fitness_func(params)), input_population))
    append_better = _append_better_func(type)
    get_best = _get_best_func(type)

    def lehmer_mean(x, p=2):
        return sum(map(lambda x: x ** p, x)) / sum(map(lambda x: x ** (p - 1), x))
        #return sum(x ** p) / sum(x ** (p - 1)) in case of np.array
    def get_random_best(population):
        best_fitness = get_best(population)[1]
        # get all 100 percentile from population
        all_bests = [x for x in population if x[1] == best_fitness]
        index = random.randrange(0, len(all_bests))
        return all_bests[index][0]
    def get_r1(P, i):
        while(True):
            index = np.random.randint(NP)
            r1 = P[index][0]
            if not np.array_equal(r1, i):
                break
        return r1
    def get_r2(P, A, i, r1):
        while(True):
            index = np.random.randint(NP + len(A))
            r2 = (P[index] if index < NP else A[index - NP])[0]
            if not np.array_equal(r2, i) and not np.array_equal(r2, r1):
                break
        return r2
    """
    Computes param based on the 'current-to-best/1' strategy
    """
    def compute_param(x, j):
        r = np.random.random_sample()
        return x + F * (best[j] - x) + F * (r1[j] - r2[j]) \
            if j == j_rand or r < CR else \
            x
    def get_CR():
        CR = np.random.normal(u_CR, 0.1)
        CR = max(0, CR)
        CR = min(1, CR)
        return CR
    def get_F():
        while True:
            F = cauchy.rvs(u_F, 0.1)
            if F > 0:
                F = min(1, F)
                break
        return F

    for g in range(iterations):
        print("iteration=", g)
        new_population = []
        s_CR = list()
        s_F = list()

        best = get_random_best(population)
        for params in population:
            CR = get_CR()
            F = get_F()

            #best = get_random_best(population)
            r1 = get_r1(population, params[0])
            r2 = get_r2(population, A, params[0], r1)
            j_rand = np.random.randint(D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(params[0])], dtype)
            new_params = (new_params, fitness_func(new_params))

            if append_better(new_population, new_params, params):
                A.append(new_params)
                s_CR.append(CR)
                s_F.append(F)

        population = new_population
        while len(A) > NP:
            del A[np.random.randint(0, len(A))]
        if s_CR and s_F:
            u_CR = (1 - C) * u_CR + C * np.mean(s_CR)
            u_F = (1 - C) * u_F + C * lehmer_mean(s_F)

    best = get_best(population)
    return best[0]
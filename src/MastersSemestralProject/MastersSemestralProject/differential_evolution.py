import numpy as np
from collections import namedtuple
from scipy.stats import cauchy

PopulationMember = namedtuple('PopulationMember', ['params', 'fitness'])
Result = namedtuple('Result', ['params', 'fitness', 'generations', 'history'])

def _append_min(population, new, old):
    is_better = new.fitness < old.fitness
    population.append(new if is_better else old)
    return is_better
def _append_max(population, new, old):
    is_better = new.fitness > old.fitness
    population.append(new if is_better else old)
    return is_better
def _append_better_func(type):
    return _append_min if type == 'min' else _append_max

def _get_best_min(population):
    return min(population, key=lambda x: x.fitness)
def _get_best_max(population):
    return max(population, key=lambda x: x.fitness)
def _get_best_func(type):
    return _get_best_min if type == 'min' else _get_best_max

def _is_better_min(new, old):
    return new.fitness < old.fitness
def _is_better_max(new, old):
    return new.fitness > old.fitness
def _get_is_better(type):
    return _is_better_min if type == 'min' else _is_better_max

def _is_better_or_eq_min(new, old):
    return new.fitness <= old.fitness
def _is_better_or_eq_max(new, old):
    return new.fitness >= old.fitness
def _get_is_better_or_eq(type):
    return _is_better_or_eq_min if type == 'min' else _is_better_or_eq_max

def _filter_pbest_min(population, p):
    fitnesses = list(map(lambda x: x.fitness, population))
    percentile = np.percentile(fitnesses, p)
    result = [x for x in population if x.fitness <= percentile]
    return result
def _filter_pbest_max(population, p):
    fitnesses = list(map(lambda x: x.fitness, population))
    percentile = np.percentile(fitnesses, 100 - p)
    result = [x for x in population if x.fitness >= percentile]
    return result
def _get_filter_pbest(type):
    return _filter_pbest_min if type == 'min' else _filter_pbest_max


# TODO: make all stragegies as a separate functions that may be reusable

def de(fitness_func, input_population, end_condition, type):
    F = 0.5 # [0,2]
    CR = 0.9 # [0,1]
    N = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    population = list(map(lambda params: PopulationMember(params, fitness_func(params)), input_population))
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

    g = 0
    history = []
    while True:
        print("Generation=", g)
        new_population = []
        for i, member in enumerate(population):
            indexes = get_three_random_agents(i)
            a,b,c = map(lambda i: population[i].params, indexes)
            R = np.random.randint(0, D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(member.params)], dtype)
            new_member = PopulationMember(new_params, fitness_func(new_params))

            append_better(new_population, new_member, member)

        population = new_population

        best = get_best(population)
        history.append(best.fitness)
        g = g + 1
        if(end_condition(best.fitness)):
            break
        
    return Result(best.params, best.fitness, g, history)

def de_jade_with_archive(fitness_func, input_population, end_condition, type):
    NP = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    C = 0.8 # [0,1]
    u_CR = 0.5
    u_F = 0.5
    A = list()

    population = list(map(lambda params: PopulationMember(params, fitness_func(params)), input_population))
    append_better = _append_better_func(type)
    get_best = _get_best_func(type)

    def lehmer_mean(x, p=2):
        return sum(map(lambda x: x ** p, x)) / sum(map(lambda x: x ** (p - 1), x))
        #return sum(x ** p) / sum(x ** (p - 1)) in case of np.array
    def get_random_best(population):
        best_fitness = get_best(population).fitness
        # get all 100 percentile from population
        all_bests = [x for x in population if x.fitness == best_fitness]
        index = np.random.randint(len(all_bests))
        return all_bests[index].params
    def get_r1(P, i):
        while(True):
            index = np.random.randint(NP)
            r1 = P[index].params
            if not np.array_equal(r1, i):
                break
        return r1
    def get_r2(P, A, i, r1):
        while(True):
            index = np.random.randint(NP + len(A))
            r2 = (P[index] if index < NP else A[index - NP]).params
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

    g = 0
    history = []
    while True:
        print("Generation=", g)
        new_population = []
        s_CR = list()
        s_F = list()

        best = get_random_best(population)
        for member in population:
            CR = get_CR()
            F = get_F()

            r1 = get_r1(population, member.params)
            r2 = get_r2(population, A, member.params, r1)
            j_rand = np.random.randint(D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(member.params)], dtype)
            new_member = PopulationMember(new_params, fitness_func(new_params))

            if append_better(new_population, new_member, member):
                A.append(new_member)
                s_CR.append(CR)
                s_F.append(F)

        population = new_population
        while len(A) > NP:
            del A[np.random.randint(0, len(A))]
        if s_CR and s_F:
            u_CR = (1 - C) * u_CR + C * np.mean(s_CR)
            u_F = (1 - C) * u_F + C * lehmer_mean(s_F)

        best = get_best(population)
        history.append(best.fitness)
        g = g + 1
        if(end_condition(best.fitness)):
            break

    return Result(best.params, best.fitness, g, history)

def de_shade(fitness_func, input_population, end_condition, type):
    NP = len(input_population)
    D = len(input_population[0])
    dtype = input_population[0].dtype

    p_min = 1 / NP # modifiable
    H = 10 # modifiable
    M_CR = [0.5] * H
    M_F = [0.5] * H
    A = list()
    k = 0

    population = list(map(lambda params: PopulationMember(params, fitness_func(params)), input_population))
    is_better = _get_is_better(type)
    is_better_or_eq = _get_is_better_or_eq(type)
    get_best = _get_best_func(type)
    filter_pbest = _get_filter_pbest(type)

    def weighted_arithmetic_mean(x, w):
        sum_w = sum(w)
        return sum([(w[k] / sum_w) * x[k] for k in range(len(x))])
    def weighted_lehmer_mean(x, w, p=2):
        sum_w = sum(w)
        a = sum([(w[k] / sum_w) * x[k] ** p for k in range(len(x))])
        b = sum([(w[k] / sum_w) * x[k] ** (p - 1) for k in range(len(x))])
        return a / b
    def get_pbest(population):
        p = np.random.uniform(p_min, 0.2) * 100
        all_bests = filter_pbest(population, p)
        index = np.random.randint(len(all_bests))
        return all_bests[index].params
    def get_r1(P, i):
        while(True):
            index = np.random.randint(NP)
            r1 = P[index].params
            if not np.array_equal(r1, i):
                break
        return r1
    def get_r2(P, A, i, r1):
        while(True):
            index = np.random.randint(NP + len(A))
            r2 = (P[index] if index < NP else A[index - NP]).params
            if not np.array_equal(r2, i) and not np.array_equal(r2, r1):
                break
        return r2
    """
    Computes param based on the 'current-to-pbest/1' strategy
    """
    def compute_param(x, j):
        r = np.random.random_sample()
        return x + F * (pbest[j] - x) + F * (r1[j] - r2[j]) \
            if j == j_rand or r < CR else \
            x
    def get_CR(r):
        CR = np.random.normal(M_CR[r], 0.1)
        CR = max(0, CR)
        CR = min(1, CR)
        return CR
    def get_F(r):
        while True:
            F = cauchy.rvs(M_F[r], 0.1)
            if F > 0:
                F = min(1, F)
                break
        return F

    g = 0
    history = []
    while True:
        print("iteration=", g)
        new_population = []
        s_CR = list()
        s_F = list()
        s_W = list() # just memory for the delta f_k
        for member in population:
            r = np.random.randint(H)
            CR = get_CR(r)
            F = get_F(r)

            pbest = get_pbest(population)
            r1 = get_r1(population, member.params)
            r2 = get_r2(population, A, member.params, r1)
            j_rand = np.random.randint(D)

            new_params = np.fromiter([compute_param(x,j) for j,x in enumerate(member.params)], dtype)
            new_member = PopulationMember(new_params, fitness_func(new_params))

            new_population.append(new_member if is_better_or_eq(new_member, member) else member)
            if is_better(new_member, member):
                A.append(new_member)
                s_CR.append(CR)
                s_F.append(F)
                s_W.append(new_member.fitness - member.fitness)

        population = new_population
        while len(A) > NP:
            del A[np.random.randint(0, len(A))]
        if s_CR and s_F:
            M_CR[k] = weighted_arithmetic_mean(s_CR, s_W)
            M_F[k] = weighted_lehmer_mean(s_F, s_W)
            k = k + 1
            if k >= H:
                k = 0

        best = get_best(population)
        history.append(best.fitness)
        g = g + 1
        if(end_condition(best.fitness)):
            break

    return Result(best.params, best.fitness, g, history)
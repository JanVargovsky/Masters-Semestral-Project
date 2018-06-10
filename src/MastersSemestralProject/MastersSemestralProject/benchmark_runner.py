from collections import namedtuple
from timeit import default_timer as timer
from numpy import min, max, median, mean, std

Stats = namedtuple('Stats', ['min', 'max', 'median', 'mean', 'sd'])
BenchmarkResult = namedtuple('BenchmarkResult', ['Result', 'Time', 'Results', 'Times'])

def _create_stats(a):
    return Stats(min(a), max(a), median(a), mean(a), std(a))

def run(n, action, result_to_value):
    results = []
    times = []

    for i in range(n):
        print("Benchmark runner, iteration = {}/{}".format(i + 1, n))
        start = timer()
        result = action()
        end = timer()

        results.append(result)
        times.append(end - start)
         
    resultStats = _create_stats([result_to_value(x) for x in results])
    timesStats = _create_stats(times)
    return BenchmarkResult(resultStats, timesStats, results, times)
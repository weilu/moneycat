import unittest
import random
from app import reservior_sampling
from statistics import mean
import pandas as pd

MAX_INT = 1000
SAMPLE_SIZE = 10
population = list(range(1, MAX_INT+1))
# random.shuffle(population)
POP = pd.DataFrame(data={'x': population})
REPEAT_TIMES = 50
THRESHOLD = 0.03

class TestApp(unittest.TestCase):

    def test_reservior_sampling(self):
        means = []
        for _ in range(REPEAT_TIMES):
            samples = reservior_sampling(SAMPLE_SIZE, POP)[0]
            means.append(mean(samples['x']))
        # according to central limit theorem
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))

    def test_reservior_sampling_remaining(self):
        samples, remaining = reservior_sampling(SAMPLE_SIZE, POP)
        self.assertEqual(len(samples), SAMPLE_SIZE)
        self.assertEqual(len(remaining), MAX_INT - SAMPLE_SIZE)

    def test_reservior_sampling_with_existing_samples(self):
        means = []
        for _ in range(REPEAT_TIMES):
            k = 5
            samples = reservior_sampling(SAMPLE_SIZE, POP.loc[k:], k, POP.loc[0:k-1])[0]
            means.append(mean(samples['x']))
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))

    def test_reservior_sampling_with_full_reservior(self):
        means = []
        for _ in range(REPEAT_TIMES):
            k = 10
            samples = reservior_sampling(SAMPLE_SIZE, POP.loc[k:], k, POP.loc[0:k-1])[0]
            means.append(mean(samples['x']))
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))


if __name__ == '__main__':
    unittest.main()


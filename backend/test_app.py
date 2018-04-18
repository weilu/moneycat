import unittest
import random
from app import reservior_sampling
from statistics import mean

MAX_INT = 1000
POP = random.sample(range(1, MAX_INT+1), MAX_INT)
REPEAT_TIMES = 200
THRESHOLD = 0.03

class TestApp(unittest.TestCase):

    def test_reservior_sampling(self):
        means = []
        for _ in range(REPEAT_TIMES):
            means.append(mean(reservior_sampling(10, POP)))
        # according to central limit theorem
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))

    def test_reservior_sampling_with_existing_samples(self):
        means = []
        for _ in range(REPEAT_TIMES):
            k = 5
            means.append(mean(reservior_sampling(10, POP[k:], k, POP[0:k])))
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))

    def test_reservior_sampling_with_full_reservior(self):
        means = []
        for _ in range(REPEAT_TIMES):
            k = 10
            means.append(mean(reservior_sampling(10, POP[k:], k, POP[0:k])))
        self.assertTrue(abs(mean(means) - MAX_INT/2) < THRESHOLD * (MAX_INT/2))


if __name__ == '__main__':
    unittest.main()


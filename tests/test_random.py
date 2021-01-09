import pytest
from clfel.util.random import Random
from random import randint
import numpy as np

MAX_SEED = 10000
TESTS = 100
SIZE = 1000
TAKE = 100
SHIFT = 100

def test_creation():
    r = Random(randint(0, MAX_SEED))

def test_shift_idx():
    for i in range(TESTS):
        prime_idx = randint(0, MAX_SEED)
        random = Random(prime_idx)
        r1 = random.get_array(SIZE)
        shift = randint(0, SHIFT)
        random = Random(prime_idx)
        random.set_idx(shift)
        r2 = random.get_array(SIZE)
        assert np.array_equal(r1[shift:(shift+TAKE)], r2[:TAKE])

def test_random_access():
        prime_idx = randint(0, MAX_SEED)
        random = Random(prime_idx)
        r1 = random.get_array(SIZE)
        for i in range(TESTS):
            idx = randint(0, SIZE)
            random.set_idx(idx)
            assert random.get_value() == r1[idx]


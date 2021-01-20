import pytest
from pyfel.util.hammersley import Hammersley, PRIME_VECTOR
from random import randint
import numpy as np
from scipy.special import erfcinv
from skopt.sampler import Hammersly as SkoptHammersley
from skopt.space import Space

MAX_BASE = 100
TESTS = 100
SIZE = 1000
TAKE = 100
SHIFT = 100

def test_creation():
    h = Hammersley(randint(0, MAX_BASE))

def test_shift_idx():
    for i in range(TESTS):
        prime_idx = randint(0, MAX_BASE)
        hammer = Hammersley(prime_idx)
        h1 = hammer.get_array(SIZE)
        shift = randint(0, SHIFT)
        hammer = Hammersley(prime_idx)
        hammer.set_idx(shift)
        h2 = hammer.get_array(SIZE)
        assert np.array_equal(h1[shift:(shift+TAKE)], h2[:TAKE])

def test_normal():
    for i in range(TESTS):
        prime_idx = randint(0, MAX_BASE)
        hammer = Hammersley(prime_idx)
        h1 = hammer.get_array(SIZE)
        hammer.set_idx(0)
        h2 = hammer.get_normal_array(SIZE)
        assert np.allclose(erfcinv(h1*2.0), h2)

def test_random_access():
        prime_idx = randint(0, MAX_BASE)
        hammer = Hammersley(prime_idx)
        h1 = hammer.get_array(SIZE)
        for i in range(TESTS):
            idx = randint(0, SIZE-1)
            hammer.set_idx(idx)
            assert hammer.get_value() == h1[idx]

def test_skopt():
    space = Space([(0.0, 1.0)])
    for i in range(TESTS):
        prime_idx = randint(0, MAX_BASE)
        hammer = Hammersley(prime_idx)
        skhammer = SkoptHammersley(0, 0, [PRIME_VECTOR[prime_idx]])
        h1 = hammer.get_array(SIZE)
        h2 = np.squeeze(skhammer.generate(space.dimensions, SIZE))
        assert np.allclose(h1, h2)



import pytest
from clfel.base.beam import Beam
from clfel.util.random import Random
import numpy as np
from numpy.random import rand

SEED = 12345
TEST_SIZE = 1000000
TEST_SIZE2 = 454545
TEST_SIZE3 = 666666

test_array = np.array(rand(6, TEST_SIZE), order="C")
test_array2 = np.array(rand(6, TEST_SIZE2), order="C")
test_array3 = np.array(rand(6, TEST_SIZE3), order="C")

wrong_array1 = np.array(rand(5, TEST_SIZE), order="C")
wrong_array2 = np.array(rand(7, TEST_SIZE), order="C")

def get_random_beam(size, long_distrib=(0.0, 1.0), seed=None):
    if not seed:
        seed=SEED
    random = Random(seed)
    theta = (random.get_array(size).reshape((1, size))*(long_distrib[1] - long_distrib[0]) +
             long_distrib[1])
    beam = random.get_normal_array(5*size).reshape((5, size))
    beam = np.vstack((beam[:4, :], theta, beam[4, :]))
    return Beam(beam), beam

def test_creation():
    beam = Beam(test_array)
    assert len(beam) == TEST_SIZE
    beam = Beam(np.array(test_array, order="C"))
    assert len(beam) == TEST_SIZE
    beam = Beam(np.array(test_array, order="C").astype(np.float32))
    assert len(beam) == TEST_SIZE

    with pytest.raises(AssertionError):
        Beam(wrong_array1)

    with pytest.raises(AssertionError):
        Beam(wrong_array2)

def test_roundtrip():
    beam = Beam(test_array)
    assert np.array_equal(beam.as_numpy(), test_array)
    beam = Beam(np.array(test_array, order="C"))
    assert np.array_equal(beam.as_numpy(), test_array)

def test_roundtrip_fraction():
    beam = Beam(test_array)
    assert np.array_equal(beam.as_numpy((0, TEST_SIZE2)), test_array[:, :TEST_SIZE2])
    assert np.array_equal(beam.as_numpy((TEST_SIZE2, 0)), test_array[:, TEST_SIZE2:])
    assert np.array_equal(beam.as_numpy((TEST_SIZE2, TEST_SIZE3)), test_array[:, TEST_SIZE2:TEST_SIZE3]) 

def test_merge_particles():
    beam = Beam(test_array)
    beam.merge_particles([test_array2, test_array3])
    concatenated = np.hstack((test_array, test_array2, test_array3))
    assert np.array_equal(beam.as_numpy(), concatenated)

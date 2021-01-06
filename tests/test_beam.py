import pytest
from clfel.base.beam import Beam
import numpy as np
from numpy.random import rand

TEST_SIZE = 1000000
TEST_SIZE2 = 454545
TEST_SIZE3 = 666666

test_array = np.array(rand(TEST_SIZE, 6), order="F")
test_array2 = np.array(rand(TEST_SIZE2, 6), order="F")
test_array3 = np.array(rand(TEST_SIZE3, 6), order="F")

wrong_array1 = np.array(rand(TEST_SIZE, 5), order="F")
wrong_array2 = np.array(rand(TEST_SIZE, 7), order="F")

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
    assert np.array_equal(beam.as_numpy((0, TEST_SIZE2)), test_array[:TEST_SIZE2,:])
    assert np.array_equal(beam.as_numpy((TEST_SIZE2, 0)), test_array[TEST_SIZE2:,:])
    assert np.array_equal(beam.as_numpy((TEST_SIZE2, TEST_SIZE3)), test_array[TEST_SIZE2:TEST_SIZE3,:]) 

def test_merge_particles():
    beam = Beam(test_array)
    beam.merge_particles([test_array2, test_array3])
    concatenated = np.concatenate((test_array, test_array2, test_array3))
    assert np.array_equal(beam.as_numpy(), concatenated)

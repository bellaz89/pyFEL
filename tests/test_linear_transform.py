import numpy as np
from pyfel.base.clctx import cl_ftype
from pyfel.base.linear_transform import LinearTransform
from pyfel.util.random import Random
from .test_beam import get_random_beam
from .tolerance import cl_tol

SEED = 54321
BEAM_SIZE = 1000 

def test_dense():
    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    random_matrix = random.get_array(36).astype(cl_ftype).reshape((6,6))
    random_vector = random.get_array(6).astype(cl_ftype)
    lin_transform = LinearTransform(random_matrix, random_vector, 'dense')
    assert lin_transform.matrix_format == 'dense'
    assert np.array_equal(lin_transform.dense_matrix[:6, :6], random_matrix)
    assert np.array_equal(lin_transform.dense_matrix[:6, 6], random_vector)
    lin_transform.transform(beam_dev)
    beam_host = random_matrix.dot(beam_host) + random_vector[:, np.newaxis]
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    random_matrix = random.get_array(36).reshape((6,6))
    lin_transform = LinearTransform(random_matrix, None, 'dense')
    assert lin_transform.matrix_format == 'dense'
    lin_transform.transform(beam_dev)
    lin_transform.transform(beam_dev)
    lin_transform.transform(beam_dev)
    beam_host = random_matrix.dot(beam_host)
    beam_host = random_matrix.dot(beam_host)
    beam_host = random_matrix.dot(beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

def test_sparse():
    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    random_matrix = random.get_array(36).reshape((6,6))
    random_vector = random.get_array(6)
    lin_transform = LinearTransform(random_matrix, random_vector, 'sparse')
    assert lin_transform.matrix_format == 'sparse'
    lin_transform.transform(beam_dev)
    beam_host = random_matrix.dot(beam_host) + random_vector[:, np.newaxis]
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    zero_matrix = np.zeros((6,6)) 
    random_vector = random.get_array(6)
    lin_transform = LinearTransform(zero_matrix, random_vector, 'sparse')
    assert lin_transform.matrix_format == 'sparse'
    lin_transform.transform(beam_dev)
    beam_host = zero_matrix.dot(beam_host) + random_vector[:, np.newaxis]
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    eye_matrix = np.eye(6) 
    lin_transform = LinearTransform(eye_matrix, None, 'sparse')
    assert lin_transform.sparse_flags == 0
    assert lin_transform.matrix_format == 'sparse'
    lin_transform.transform(beam_dev)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    matrix = np.eye(6) 
    matrix[2,3] = 2.0
    matrix[5,0] = 1.0

    lin_transform = LinearTransform(matrix, None, 'sparse')

    assert lin_transform.SPARSE_SAVE_X     & lin_transform.sparse_flags
    assert lin_transform.SPARSE_LOAD_X     & lin_transform.sparse_flags
    assert lin_transform.SPARSE_SAVE_PY    & lin_transform.sparse_flags
    assert lin_transform.SPARSE_LOAD_PY    & lin_transform.sparse_flags
    assert lin_transform.SPARSE_LOAD_THETA & lin_transform.sparse_flags
    assert lin_transform.SPARSE_LOAD_Y     & lin_transform.sparse_flags

    assert (lin_transform.SPARSE_SAVE_X |   
            lin_transform.SPARSE_LOAD_X |  
            lin_transform.SPARSE_SAVE_PY |  
            lin_transform.SPARSE_LOAD_PY | 
            lin_transform.SPARSE_LOAD_THETA |
            lin_transform.SPARSE_LOAD_Y) == lin_transform.sparse_flags

    assert lin_transform.matrix_format == 'sparse'
    lin_transform.transform(beam_dev)
    beam_host = matrix.dot(beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)


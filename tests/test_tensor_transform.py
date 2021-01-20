import numpy as np
from clfel.base.clctx import cl_ftype
from clfel.base.tensor_transform import TensorTransform
from clfel.util.random import Random
from .test_beam import get_random_beam
from .tolerance import cl_tol

SEED = 54321
BEAM_SIZE = 1000 

def tensor_mul(np_tensor, np_beam):
    '''
        Multiplies a numpy tensor with a numpy beam. For debugging purpourses
    '''
    
    np_transf_beam = np.zeros_like(np_beam, dtype=cl_ftype)
    
    for index in np.ndindex(np_tensor.shape):
        line = np.ones(np_beam.shape[1], dtype=cl_ftype)*np_tensor[index]
        for i in index[1:]:
            line *= np_beam[i, :]

        np_transf_beam[index[0], :] += line

    return np_transf_beam

def tensors_mul(np_tensors, np_beam):
    '''
        Multiplies a list of numpy tensors with a numpy beam. For debugging purpourses
    '''
    np_transf_beam = np.zeros_like(np_beam)
    for np_tensor in np_tensors:
        np_transf_beam += tensor_mul(np_tensor, np_beam)
    
    return np_transf_beam

def test_tensor():
    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    tensors = []
    
    tensors.append(random.get_array(6*6*6*6).astype(cl_ftype).reshape((6,6,6,6)))
    tensors.append(random.get_array(6*6*6).astype(cl_ftype).reshape((6,6,6)))
    tensors.append(random.get_array(6*6).astype(cl_ftype).reshape((6,6)))
    tensors.append(random.get_array(6).astype(cl_ftype))
    
    tens_transform = TensorTransform(tensors)
    assert tens_transform.flags == 0x00000FFF
    tens_transform.transform(beam_dev)
    beam_host = tensors_mul(tensors, beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    tensors = [np.zeros(tuple([6]*5))]
    
    tensors[0][0,0,0,0,5] = 1.0
    tensors[0][2,1,1,2,2] = 2.0
    tensors[0][5,3,4,2,1] = 3.0
    tensors[0][1,2,2,1,2] = 4.0
    tensors[0][1,2,2,2,1] = 5.0
    tensors[0][3,0,3,3,0] = 6.0
    tensors[0][4,0,4,4,0] = 7.0
    
    tens_transform = TensorTransform(tensors)
    assert tens_transform.flags == 0x0000FFF
    tens_transform.transform(beam_dev)
    beam_host = tensors_mul(tensors, beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    tensors = [np.zeros(tuple([6]*5))]
    
    tensors[0][0,0,0,0,5] = 1.0
    tensors[0][2,1,1,2,2] = 2.0
    
    tens_transform = TensorTransform(tensors)
    tens_transform.transform(beam_dev)
    beam_host = tensors_mul(tensors, beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    tensors = [np.zeros(6)]
    
    tensors[0][0] = 7.0
    tensors[0][1] = 6.0
    tensors[0][2] = 5.0
    tensors[0][3] = 4.0
    tensors[0][4] = 3.0
    tensors[0][5] = 1.0
    
    tens_transform = TensorTransform(tensors)
    tens_transform.transform(beam_dev)
    beam_host = tensors_mul(tensors, beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)

    beam_dev, beam_host = get_random_beam(BEAM_SIZE)
    random = Random(SEED)
    tensors = [np.eye(6)]
    
    tens_transform = TensorTransform(tensors)
    
    assert tens_transform.flags == 0

    tens_transform.transform(beam_dev)
    beam_host = tensors_mul(tensors, beam_host)
    assert np.allclose(beam_dev[:], beam_host, **cl_tol)


'''
    Perform a nonlinear tensor transform
'''

import numpy as np
import pyopencl as cl
from .beam_transform import BeamTransform
from .beam import Beam
from .clctx import cl_queue, cl_ctx, cl_ftype, F 

from pyfel.util.init_class import init_class

@init_class
class TensorTransform(BeamTransform):
    '''
        Class representing a linear 6D transform of the beam
    '''

    SPARSE_SAVE_THETA = 0x00000001
    SPARSE_SAVE_GAMMA = 0x00000002
    SPARSE_SAVE_X     = 0x00000004
    SPARSE_SAVE_Y     = 0x00000008
    SPARSE_SAVE_PX    = 0x00000010
    SPARSE_SAVE_PY    = 0x00000020
    SPARSE_LOAD_THETA = 0x00000040
    SPARSE_LOAD_GAMMA = 0x00000080
    SPARSE_LOAD_X     = 0x00000100
    SPARSE_LOAD_Y     = 0x00000200
    SPARSE_LOAD_PX    = 0x00000400
    SPARSE_LOAD_PY    = 0x00000800

    KERNEL = '''
                #define SPARSE_SAVE_THETA 0x00000001
                #define SPARSE_SAVE_GAMMA 0x00000002
                #define SPARSE_SAVE_X     0x00000004
                #define SPARSE_SAVE_Y     0x00000008
                #define SPARSE_SAVE_PX    0x00000010
                #define SPARSE_SAVE_PY    0x00000020
                #define SPARSE_LOAD_THETA 0x00000040
                #define SPARSE_LOAD_GAMMA 0x00000080
                #define SPARSE_LOAD_X     0x00000100
                #define SPARSE_LOAD_Y     0x00000200
                #define SPARSE_LOAD_PX    0x00000400
                #define SPARSE_LOAD_PY    0x00000800

                // Sparse tensor transformation of the beam.
                __kernel void tensor_optic_sparse(__global FLOAT_TYPE* x,
                                                  __global FLOAT_TYPE* px,
                                                  __global FLOAT_TYPE* y,
                                                  __global FLOAT_TYPE* py,
                                                  __global FLOAT_TYPE* theta,
                                                  __global FLOAT_TYPE* gamma,
                                                  __constant FLOAT_TYPE* transfer_matrix,
                                                  __constant uchar* indices,
                                                  const uint transfer_matrix_size,
                                                  const uint flags){

                    const ulong gid = get_global_id(0);
                    FLOAT_TYPE state[6], new_state[6];

                    if(flags & SPARSE_SAVE_X)     state[0] = x[gid];
                    if(flags & SPARSE_SAVE_PX)    state[1] = px[gid];
                    if(flags & SPARSE_SAVE_Y)     state[2] = y[gid];
                    if(flags & SPARSE_SAVE_PY)    state[3] = py[gid];
                    if(flags & SPARSE_SAVE_THETA) state[4] = theta[gid];
                    if(flags & SPARSE_SAVE_GAMMA) state[5] = gamma[gid];

                    for(uint i = 0; i < 6; i++) {
                        new_state[i] = 0.0;
                    }

                    for(uint i = 0; i < transfer_matrix_size; i++){

                        FLOAT_TYPE term = transfer_matrix[i]; 
                            
                        for(uint j = 0; j < 6; j++){
                            term *= pown(state[j], indices[i*7 + j + 1]);
                        }

                        new_state[indices[i*7]] += term; 
                    }

                    if(flags & SPARSE_SAVE_X)     x[gid]     = new_state[0];
                    if(flags & SPARSE_SAVE_PX)    px[gid]    = new_state[1];
                    if(flags & SPARSE_SAVE_Y)     y[gid]     = new_state[2];
                    if(flags & SPARSE_SAVE_PY)    py[gid]    = new_state[3];
                    if(flags & SPARSE_SAVE_THETA) theta[gid] = new_state[4];
                    if(flags & SPARSE_SAVE_GAMMA) gamma[gid] = new_state[5];
                }
    '''
 

    def __init__(self, tensors):
        '''
            Initialize the object with a list of 6D tensors of arbitrary order. 
            Note that all dimensions should be "6"
        '''
        for tensor in tensors:
            for dims in tensor.shape:
                assert dims == 6, "All tensor dimensions should be 6" 
   
        mf = cl.mem_flags
        self.values, self.indices, self.flags, self.elements = self.analyze_tensors(tensors)

        if self.flags != 0:
            self.device_values = cl.Buffer(cl_ctx, 
                                           mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                           hostbuf=self.values)

            self.device_indices = cl.Buffer(cl_ctx, 
                                            mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                            hostbuf=self.indices)


    @staticmethod
    def analyze_tensors(tensors):
        '''
            Decompose a dense matrix in a sparse format
        '''
        values = []
        indices = []
        flags = 0x0
        elements = 0
        tensor_dict = dict()

        for tensor in tensors:
            for index in np.ndindex(tensor.shape):
                if tensor[index] != 0:
                    dest = index[0]
                    idx = [0]*6
                    for i in range(6):
                        idx[i] = index[1:].count(i)

                    key = (dest, tuple(idx))

                    if key in tensor_dict:
                        tensor_dict[key] += tensor[index]
                    else:
                        tensor_dict[key] = tensor[index]
                        elements += 1

            
        for i in range(6):
            line_elements = list(filter(lambda x: x[0] == i, tensor_dict))
            idx = [0]*6
            idx[i] = 1
            idx = tuple(idx)

            if (len(line_elements) == 1 and 
                line_elements[0] == (i, idx) and
                tensor_dict[line_elements[0]] == 1.0):

                tensor_dict.pop(line_elements[0])
            else:
                flags = flags | (0x1 << i)

        for i in range(6):
            line_elements = list(filter(lambda x: x[1][i] != 0, tensor_dict))

            if len(line_elements) > 0:
                flags = flags | (0x1 << (i+6))

        for k,v in tensor_dict.items():
            values.append(v)
            indices.append(k[0])
            indices.extend(k[1])

        values = np.array(values, dtype=cl_ftype)
        indices = np.array(indices, dtype=np.uint8)

        return values, indices, flags, elements

    def transform(self, beam):
        '''
            Apply the transform to the beam
        '''
        
        if self.flags != 0:
            event = self.program.tensor_optic_sparse(cl_queue, (len(beam),),
                                             None,
                                             beam.x.data, beam.px.data,
                                             beam.y.data, beam.py.data,
                                             beam.theta.data, beam.gamma.data,
                                             self.device_values,
                                             self.device_indices,
                                             np.uint32(self.elements),
                                             np.uint32(self.flags))
            beam.events.append(event)

    @classmethod
    def initialize(cls):
        '''
            Compile kernels
        '''
        cls.program = cl.Program(cl_ctx, F(cls.KERNEL)).build()


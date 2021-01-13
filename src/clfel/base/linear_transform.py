'''
    Perform a linear transform of the beam
'''

import numpy as np
import pyopencl as cl
from .beam_transform import BeamTransform
from .beam import Beam
from .clctx import cl_queue, cl_ctx 

from clfel.util.init_class import init_class

@init_class
class LinearTransform(BeamTransform):
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
                // Dense linear transformation of the beam. The result is new_state = M*state
                __kernel void linear_optic_dense(__global double* x,
                                                 __global double* px,
                                                 __global double* y,
                                                 __global double* py,
                                                 __global double* theta,
                                                 __global double* gamma,
                                                 __constant double* transfer_matrix) {
                    
                    const ulong gid = get_global_id(0);
                    double state[6];
                    double new_state[6];

                    state[0] = x[gid];
                    state[1] = px[gid];
                    state[2] = y[gid];
                    state[3] = py[gid];
                    state[4] = theta[gid];
                    state[5] = gamma[gid];


                    for(int i = 0; i < 6; i++) { 
                        new_state[i] = transfer_matrix[i + 36];
                    }

                    for(int j = 0; j < 6; j++) {
                        for(int i = 0; i < 6; i++) { 
                            new_state[i] += transfer_matrix[i + j*6] * state[j];
                        }
                    }

                    x[gid]     = new_state[0];
                    px[gid]    = new_state[1];
                    y[gid]     = new_state[2];
                    py[gid]    = new_state[3];
                    theta[gid] = new_state[4];
                    gamma[gid] = new_state[5];
                }


                #define SPARSE_SAVE_X       0x00000001
                #define SPARSE_SAVE_PX      0x00000002
                #define SPARSE_SAVE_Y       0x00000004
                #define SPARSE_SAVE_PY      0x00000008
                #define SPARSE_SAVE_THETA   0x00000010
                #define SPARSE_SAVE_GAMMA   0x00000020
                #define SPARSE_LOAD_X       0x00000040
                #define SPARSE_LOAD_PX      0x00000080
                #define SPARSE_LOAD_Y       0x00000100
                #define SPARSE_LOAD_PY      0x00000200
                #define SPARSE_LOAD_THETA   0x00000400
                #define SPARSE_LOAD_GAMMA   0x00000800
                
                // Sparse linear transformation of the beam.
                __kernel void linear_optic_sparse(__global double* x,
                                                  __global double* px,
                                                  __global double* y,
                                                  __global double* py,
                                                  __global double* theta,
                                                  __global double* gamma,
                                                  __constant double* transfer_matrix,
                                                  __constant uchar2* indices,
                                                  const uint transfer_matrix_size,
                                                  const uint flags){

                    const ulong gid = get_global_id(0);
                    double state[6], new_state[6];

                    if(flags & SPARSE_LOAD_X)     state[0] = x[gid];
                    if(flags & SPARSE_LOAD_PX)    state[1] = px[gid];
                    if(flags & SPARSE_LOAD_Y)     state[2] = y[gid];
                    if(flags & SPARSE_LOAD_PY)    state[3] = py[gid];
                    if(flags & SPARSE_LOAD_THETA) state[4] = theta[gid];
                    if(flags & SPARSE_LOAD_GAMMA) state[5] = gamma[gid];

                    for(uint i = 0; i < 6; i++) {
                        new_state[i] = transfer_matrix[i];
                    }

                    for(uint i = 0; i < transfer_matrix_size; i++){
                        uchar2 pos = indices[i]; 
                        new_state[pos.x] += transfer_matrix[i+6] * state[pos.y];
                    }

                    if(flags & SPARSE_SAVE_X)     x[gid]     = new_state[0];
                    if(flags & SPARSE_SAVE_PX)    px[gid]    = new_state[1];
                    if(flags & SPARSE_SAVE_Y)     y[gid]     = new_state[2];
                    if(flags & SPARSE_SAVE_PY)    py[gid]    = new_state[3];
                    if(flags & SPARSE_SAVE_THETA) theta[gid] = new_state[4];
                    if(flags & SPARSE_SAVE_GAMMA) gamma[gid] = new_state[5];
                }
    '''
 

    def __init__(self, transfer_matrix, shift_vector=None, matrix_format=None, format_threshold=8):
        '''
            Initialize the object with a 6D transfer matrix. matrix_format can be
            either 'dense' or 'sparse'. If None, the matrix format is chosen 
            whether the required load/stores are higher than format_threshold.
        '''

        self.dense_matrix = np.array(transfer_matrix, dtype=np.float64, order='F')

        assert self.dense_matrix.shape == (6,6), "Error, the transfer matrix in" \
                                                 "linear transform must be 6x6" 

        if not isinstance(shift_vector, np.ndarray):
            shift_vector = np.zeros(6, dtype=np.float64)
     
        assert shift_vector.shape == (6,), "Error, the shift vector length" \
                                           "in linear transform must be 6"

        expanded_matrix = np.empty((6,7), dtype=np.float64, order='F')
        expanded_matrix[:,:6] = self.dense_matrix
        expanded_matrix[:,6] = shift_vector 
        self.dense_matrix = expanded_matrix

        (self.sparse_values, 
         self.sparse_indices, 
         self.sparse_flags,
         self.sparse_mem_op) = self.analyze_matrix(self.dense_matrix)

        if (matrix_format == 'dense') or (matrix_format == 'sparse'):
            self.set_matrix_format(matrix_format)
        elif matrix_format == None:
            if self.sparse_mem_op > format_threshold:
                self.set_matrix_format('dense')
            else:
                self.set_matrix_format('sparse')
        else:
            raise RuntimeError("Unrecognized matrix format \'" + 
                               matrix_format + 
                               "\' in creating a linear transform")

    @staticmethod
    def analyze_matrix(dense_matrix):
        '''
            Decompose a dense matrix in a sparse format
        '''
        values = []
        indices = []
        flags = 0x0
        mem_op = 0
        identity = np.eye(6, 7, dtype=np.float64)
        
        for i in range(6):
            if not np.array_equal(dense_matrix[i,:], identity[i,:]):
                flags = flags | (0x1 << i)
                mem_op += 1

            if ((not np.array_equal(dense_matrix[:,i], identity[:,i])) or 
                ((flags & (0x1 << i)) and not np.array_equal(dense_matrix[i,:6], identity[i,:6]))):
                flags = flags | (0x1 << (i+6))
                mem_op += 1

        for i in range(6):
            for j in range(6):
                if (dense_matrix[i,j] != 0.0) and (flags & (0x1 << i)):
                    values.append(dense_matrix[i,j])
                    indices.extend([i, j])

        values = np.array(values, dtype=np.float64)
        indices = np.array(indices, dtype=np.uint8)

        return values, indices, flags, mem_op

    def transform(self, beam):
        '''
            Apply the transform to the beam
        '''
        beam.wait()
        if self.matrix_format == 'dense':
            event = self.program.linear_optic_dense(cl_queue, (len(beam),),
                                                    None,
                                                    beam.x, beam.px,
                                                    beam.y, beam.py,
                                                    beam.theta, beam.gamma,
                                                    self.device_dense_matrix)
            beam.events.append(event)

        if self.matrix_format == 'sparse':

            values_n = self.sparse_values.shape[0]
            
            if self.sparse_flags != 0:
                event = self.program.linear_optic_sparse(cl_queue, (len(beam),),
                                                 None,
                                                 beam.x, beam.px,
                                                 beam.y, beam.py,
                                                 beam.theta, beam.gamma,
                                                 self.device_sparse_values,
                                                 self.device_sparse_indices,
                                                 np.int32(values_n),
                                                 np.int32(self.sparse_flags))
                beam.events.append(event)


    def set_matrix_format(self, matrix_format):
        '''
            Set the matrix format and allocates the necessary arrays
        '''
        assert matrix_format == 'dense' or matrix_format == 'sparse', "Matrix format not recognized" 
        self.matrix_format = matrix_format
        self.device_dense_matrix = None
        self.device_sparse_values = None
        self.device_sparse_indices = None

        mf = cl.mem_flags
        if matrix_format == 'dense':
            self.device_dense_matrix = cl.Buffer(cl_ctx, 
                                                 mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                                 hostbuf=self.dense_matrix)
        
        if matrix_format == 'sparse':
            extended_sparse_values = np.concatenate((self.dense_matrix[:,6], 
                                                     self.sparse_values)) 

            self.device_sparse_values = cl.Buffer(cl_ctx, 
                                                  mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                                  hostbuf=extended_sparse_values)

            sparse_indices = self.sparse_indices if self.sparse_indices.shape[0]else np.array([0])
            self.device_sparse_indices = cl.Buffer(cl_ctx, 
                                                   mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                                   hostbuf=sparse_indices)
            


    @classmethod
    def initialize(cls):
        '''
            Compile kernels
        '''
        cls.program = cl.Program(cl_ctx, cls.KERNEL).build()


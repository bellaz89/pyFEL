'''
    Beam module.
'''
from .clctx import cl_ctx, cl_queue
import pyopencl as cl
import numpy as np

from clfel.util.init_class import init_class

@init_class
class Beam(object):
    '''
        Beam class, a collection of slices.
    '''
    PREAMBLE = '''
                // atomic addition for double \n
                double __attribute__((overloadable)) atomic_add(__global double *ptr, 
                                                                double delta) {
                    typedef union { 
                        double d;
                        ulong  ul;
                    } d_conversion;

                    d_conversion u1, u2;
                    
                    do {
                        u1.d = *ptr;
                        u2.d = u1.d + delta;
                    } while (atom_cmpxchg((volatile __global ulong*)ptr, u1.ul, u2.ul) != u1.ul);
                    return u1.d;
                }

                // atomic min for double \n
                double __attribute__((overloadable)) atomic_min(__global double *ptr, double value) {
                    typedef union {
                        double d;
                        ulong  ul;
                    } d_conversion;

                    d_conversion u1, u2;
                    
                    do {
                        u1.d = *ptr;
                        u2.d = min(u1.d, value);
                    } while (atom_cmpxchg((volatile __global ulong*)ptr, u1.ul, u2.ul) != u1.ul);
                    return u1.d;
                }

                // atomic max for double \n
                double __attribute__((overloadable)) atomic_max(__global double *ptr, double value) {
                    typedef union {
                        double d;
                        ulong  ul;
                    } d_conversion;

                    d_conversion u1, u2;
                    
                    do {
                        u1.d = *ptr;
                        u2.d = max(u1.d, value);
                    } while (atom_cmpxchg((volatile __global ulong*)ptr, u1.ul, u2.ul) != u1.ul);
                    return u1.d;
                }
                
                //  discretize value by the discretization len 'disc' \n
                uint discretize(double value, double disc){
                    return (uint) floor(value/disc); 
                }

                //  discretize value by the invers discretization len 'invdisc' \n
                uint discretize_inv(double value, double invdisc){
                    return (uint) floor(value*invdisc); 
                }
                '''

    KERNEL = PREAMBLE +'''
                // This kernel shifts the memory on the left by 'size'.\n
                __kernel void shift(__global double* arr, 
                                    __local double* tmp_arr,
                                    const ulong size, 
                                    const ulong shift){
                 
                    const ulong lid = get_local_id(0);
                    const ulong lsize = get_local_size(0);
                 
                    for(ulong idx = lid; idx<size; idx += lsize){
                        tmp_arr[lid] = arr[idx+shift];
                        barrier(CLK_LOCAL_MEM_FENCE);
                        arr[idx] = tmp_arr[lid];
                    }
                }
                
                // This kernel performs diagnostics on the beam (version1). \n
                __kernel void diagnostic1(__global double* theta,
                                          __global double* gamma,
                                          __global double* x,
                                          __global double* y,
                                          __global double* px,
                                          __global double* py,
                                          __global double* diagnostic_arr,
                                          __local double* tmp,
                                          const ulong size,
                                          const uint harmonics){

                    const ulong gid = get_global_id(0);
                    const ulong lid = get_local_id(0);
                    const ulong lsize = get_local_size(0);

                    if(gid < size){
                        tmp[lid+lsize*0] = theta[gid];
                        tmp[lid+lsize*1] = sin(tmp[lid+lsize*0]);

                        for(uint i = 0; i < harmonics*2; i+=2) {
                            double theta_harm = tmp[lid+lsize*0]*((double)(i+2));
                            tmp[lid+lsize*(10+i)] = cos(theta_harm);  
                            tmp[lid+lsize*(11+i)] = sin(theta_harm);
                        }
                        
                        tmp[lid+lsize*0] = cos(tmp[lid+lsize*0]);
                        tmp[lid+lsize*2] = x[gid];
                        tmp[lid+lsize*3] = y[gid];
                        tmp[lid+lsize*4] = px[gid];
                        tmp[lid+lsize*5] = py[gid];
                        tmp[lid+lsize*6] = tmp[lid+lsize*2] * tmp[lid+lsize*2];
                        tmp[lid+lsize*7] = tmp[lid+lsize*3] * tmp[lid+lsize*3];
                        tmp[lid+lsize*8] = gamma[gid];
                        tmp[lid+lsize*9] = tmp[lid+lsize*8] * tmp[lid+lsize*8];

                    } else {
                        for(uint i = 0; i < 10+harmonics*2; i++) {
                            tmp[lid+lsize*i] = 0.0; 
                        }
                    }
                    
                    barrier(CLK_LOCAL_MEM_FENCE);

                    for(uint i = gid/2; i>0; i >>= 1) {
                        if(lid < i) {
                            for(uint j = 0; j < 10+harmonics*2; j++) {
                                    tmp[lid+lsize*j] += tmp[lid+lsize*j+1];
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                        
                    for(uint i = lid; i < 10+harmonics*2; i+=lsize) {
                        atomic_add(diagnostic_arr+i, tmp[lsize*i]);
                    }
                }

                // This kernel performs diagnostics on the beam (version2). \n
                __kernel void diagnostic2(__global double* gamma,
                                          __global double* x,
                                          __global double* y,
                                          __global double* px,
                                          __global double* py,
                                          __global double* diagnostic_arr,
                                          __local double* tmp,
                                          const ulong size){

                    const ulong gid = get_global_id(0);
                    const ulong lid = get_local_id(0);
                    const ulong lsize = get_local_size(0);

                    if(gid < size){
                        tmp[lid+lsize*0] = gamma[gid];
                        tmp[lid+lsize*1] = x[gid];
                        tmp[lid+lsize*2] = y[gid];
                        tmp[lid+lsize*3] = px[gid];
                        tmp[lid+lsize*4] = py[gid];
                        tmp[lid+lsize*5] = tmp[lid+lsize*1]*tmp[lid+lsize*1];
                        tmp[lid+lsize*6] = tmp[lid+lsize*2]*tmp[lid+lsize*2];
                        tmp[lid+lsize*7] = tmp[lid+lsize*3]*tmp[lid+lsize*3];
                        tmp[lid+lsize*8] = tmp[lid+lsize*4]*tmp[lid+lsize*4];
                        tmp[lid+lsize*9]  = tmp[lid+lsize*1]*tmp[lid+lsize*3];
                        tmp[lid+lsize*10] = tmp[lid+lsize*2]*tmp[lid+lsize*4];

                    } else {
                        for(uint i = 0; i < 11; i++) {
                            tmp[lid+lsize*i] = 0.0; 
                        }
                    }
                    
                    barrier(CLK_LOCAL_MEM_FENCE);

                    for(uint i = gid/2; i>0; i >>= 1) {
                        if(lid < i) {
                            for(uint j = 0; j < 11; j++) {
                                    tmp[lid+lsize*j] += tmp[lid+lsize*j+1];
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                        
                    for(uint i = lid; i < 11; i+=lsize) {
                        atomic_add(diagnostic_arr+i, tmp[lsize*i]);
                    }
                }

                // Dense linear transformation of the beam. The result is new_state = M*state
                __kernel void linear_optic_dense(__global double* theta,
                                                 __global double* gamma,
                                                 __global double* x,
                                                 __global double* y,
                                                 __global double* px,
                                                 __global double* py,
                                                 __constant double* transfer_matrix) {
                    
                    const ulong gid = get_global_id(0);
                    double state[6], new_state[6];
                    
                    state[0] = x[gid];
                    state[1] = px[gid];
                    state[2] = y[gid];
                    state[3] = py[gid];
                    state[4] = theta[gid];
                    state[5] = gamma[gid];

                    for(uint i = 0; i < 6; i++) {
                        new_state[i] = 0.0;
                        
                        for(uint j = 0; j < 6; i++) {
                            new_state[i] += transfer_matrix[i + j*6] * state[j];
                        }
                    }

                    x[gid]     = state[0];
                    px[gid]    = state[1];
                    y[gid]     = state[2];
                    py[gid]    = state[3];
                    theta[gid] = state[4];
                    gamma[gid] = state[5];
                }

                #define SPARSE_LOAD_X       0x00000001
                #define SPARSE_LOAD_PX      0x00000002
                #define SPARSE_LOAD_Y       0x00000004
                #define SPARSE_LOAD_PY      0x00000008
                #define SPARSE_LOAD_THETA   0x00000010
                #define SPARSE_LOAD_GAMMA   0x00000020
                #define SPARSE_SAVE_X       0x00000040
                #define SPARSE_SAVE_PX      0x00000080
                #define SPARSE_SAVE_Y       0x00000100
                #define SPARSE_SAVE_PY      0x00000200
                #define SPARSE_SAVE_THETA   0x00000400
                #define SPARSE_SAVE_GAMMA   0x00000800


                // Sparse linear transformation of the beam.
                __kernel void linear_optic_(__global double* theta,
                                            __global double* gamma,
                                            __global double* x,
                                            __global double* y,
                                            __global double* px,
                                            __global double* py,
                                            __constant double* transfer_matrix,
                                            __constant uint2* indices,
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
                        new_state[i] = 0.0;
                    }

                    for(uint i = 0; i < transfer_matrix_size; i++){
                        uint2 index = indices[i]; 
                        new_state[index.y] += transfer_matrix[i] * state[index.x];
                    }

                    if(flags & SPARSE_SAVE_X)     x[gid]     = new_state[0];
                    if(flags & SPARSE_SAVE_PX)    px[gid]    = new_state[1];
                    if(flags & SPARSE_SAVE_Y)     y[gid]     = new_state[2];
                    if(flags & SPARSE_SAVE_PY)    py[gid]    = new_state[3];
                    if(flags & SPARSE_SAVE_THETA) theta[gid] = new_state[4];
                    if(flags & SPARSE_SAVE_GAMMA) gamma[gid] = new_state[5];
                }
                
                '''

    def __init__(self, beam_array):
        '''
            Creates a new beam from a an array of 6 columns, X rows
            representing the particles
            The columns of the array represents: theta, gamma, x, y, px, py
        '''
        assert beam_array.shape[1] == 6, "The columns must be 6"
        self.size = beam_array.shape[0]
        self.len = beam_array.shape[0]
        self.events = []

        if beam_array.dtype != np.float64:
            beam_array = beam_array.astype(np.float64)

        if not beam_array.flags.f_contiguous:
            beam_array = np.array(beam_array, order="F")

        mf = cl.mem_flags
        
        self.theta = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 0])

        self.gamma = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 1])

        self.x = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 2])

        self.y = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 3])

        self.px = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 4])

        self.py = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                                hostbuf=beam_array[:, 5])
       
    def wait(self):
        '''
            Waits for the events to be completed
        '''
        for event in self.events:
            event.wait()
        self.events = []

    def as_numpy(self, beam_range=None):
        '''
            Returns the beam as numpy array (see the constructor for the order)
        '''
        self.wait()
       
        if beam_range:
            beam_range = list(beam_range)
            if beam_range[1] <= 0:
                beam_range[1] += self.len

            arr_len = beam_range[1] - beam_range[0]
            beam_array = np.empty((arr_len, 6), dtype=np.float64, order="F")
            
            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 0], 
                                               self.theta,
                                               device_offset=beam_range[0]*8))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 1], 
                                               self.gamma,
                                               device_offset=beam_range[0]*8)) 

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 2], 
                                               self.x,
                                               device_offset=beam_range[0]*8))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 3], 
                                               self.y,
                                               device_offset=beam_range[0]*8))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 4], 
                                               self.px,
                                               device_offset=beam_range[0]*8))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[:, 5], 
                                               self.py,
                                               device_offset=beam_range[0]*8)) 

            return beam_array
        else:
            beam_array = np.empty((self.len, 6), dtype=np.float64, order="F")
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 0], self.theta))
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 1], self.gamma)) 
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 2], self.x))    
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 3], self.y))   
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 4], self.px))   
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[:, 5], self.py)) 
        
            return beam_array

    def __len__(self):
        '''
            Returns the number of particles
        '''
        return self.len

    def merge_particles(self, beam_arrays):
        '''
            Merge new particles in the beam.
        '''
        
        new_len = self.len
        for i in range(len(beam_arrays)):
            beam_array = beam_arrays[i]
            assert beam_array.shape[1] == 6, "The columns must be 6"
            new_len += beam_array.shape[0]

            if beam_array.dtype != np.float64:
                beam_array = beam_array.astype(np.float64)

            if not beam_array.flags.f_contiguous:
                beam_array = np.array(beam_array, order="F")

            beam_arrays[i] = beam_array

        self.resize(new_len)

        for beam_array in beam_arrays:
            offset = self.len*8
            self.len += beam_array.shape[0]
            self.events.append(cl.enqueue_copy(cl_queue, self.theta, 
                                               beam_array[:, 0], 
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.gamma, 
                                               beam_array[:, 1],
                                               device_offset=offset))
            
            self.events.append(cl.enqueue_copy(cl_queue, self.x, 
                                               beam_array[:, 2],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.y, 
                                               beam_array[:, 3],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.px, 
                                               beam_array[:, 4],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.py, 
                                               beam_array[:, 5],
                                               device_offset=offset))


    def resize(self, size):
        '''
            Change the array size
        '''
        if size < self.len:
            self.len = size
        elif size > self.size:
            mf = cl.mem_flags
            theta = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
            gamma = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
            x = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
            y = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
            px = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
            py = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*8)
       
            self.wait()
            self.events.append(cl.enqueue_copy(cl_queue, theta, self.theta))
            self.events.append(cl.enqueue_copy(cl_queue, gamma, self.gamma))
            self.events.append(cl.enqueue_copy(cl_queue, x, self.x))
            self.events.append(cl.enqueue_copy(cl_queue, y, self.y))
            self.events.append(cl.enqueue_copy(cl_queue, px, self.px))
            self.events.append(cl.enqueue_copy(cl_queue, py, self.py))

            self.theta = theta
            self.gamma = gamma
            self.x = x
            self.y = y
            self.px = px
            self.py = py

            self.size = size

    @classmethod
    def initialize(cls):
        '''
            Compile kernels
        '''
        cls.program = cl.Program(cl_ctx, cls.KERNEL).build()


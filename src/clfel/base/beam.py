'''
    Beam module.
'''
from .clctx import cl_ctx, cl_queue, cl_ftype, cl_ftype_nbytes, F
import pyopencl as cl
import numpy as np

from clfel.util.init_class import init_class

@init_class
class Beam(object):
    '''
        Beam class, a collection of slices.
    '''
    PREAMBLE = '''
                // atomic addition for FLOAT_TYPE \n
                FLOAT_TYPE __attribute__((overloadable)) atomic_add(__global FLOAT_TYPE *ptr, 
                                                                FLOAT_TYPE delta) {
                    typedef union { 
                        FLOAT_TYPE d;
                        ulong  ul;
                    } d_conversion;

                    d_conversion u1, u2;
                    
                    do {
                        u1.d = *ptr;
                        u2.d = u1.d + delta;
                    } while (atom_cmpxchg((volatile __global ulong*)ptr, u1.ul, u2.ul) != u1.ul);
                    return u1.d;
                }

                // atomic min for FLOAT_TYPE \n
                FLOAT_TYPE __attribute__((overloadable)) atomic_min(__global FLOAT_TYPE *ptr, FLOAT_TYPE value) {
                    typedef union {
                        FLOAT_TYPE d;
                        ulong  ul;
                    } d_conversion;

                    d_conversion u1, u2;
                    
                    do {
                        u1.d = *ptr;
                        u2.d = min(u1.d, value);
                    } while (atom_cmpxchg((volatile __global ulong*)ptr, u1.ul, u2.ul) != u1.ul);
                    return u1.d;
                }

                // atomic max for FLOAT_TYPE \n
                FLOAT_TYPE __attribute__((overloadable)) atomic_max(__global FLOAT_TYPE *ptr, FLOAT_TYPE value) {
                    typedef union {
                        FLOAT_TYPE d;
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
                uint discretize(FLOAT_TYPE value, FLOAT_TYPE disc){
                    return (uint) floor(value/disc); 
                }

                //  discretize value by the invers discretization len 'invdisc' \n
                uint discretize_inv(FLOAT_TYPE value, FLOAT_TYPE invdisc){
                    return (uint) floor(value*invdisc); 
                }
                '''

    KERNEL = PREAMBLE +'''
                // This kernel shifts the memory on the left by 'size'.\n
                __kernel void shift(__global FLOAT_TYPE* arr, 
                                    __local FLOAT_TYPE* tmp_arr,
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
                __kernel void diagnostic1(__global FLOAT_TYPE* x,
                                          __global FLOAT_TYPE* px,
                                          __global FLOAT_TYPE* y,
                                          __global FLOAT_TYPE* py,
                                          __global FLOAT_TYPE* theta,
                                          __global FLOAT_TYPE* gamma,
                                          __global FLOAT_TYPE* diagnostic_arr,
                                          __local FLOAT_TYPE* tmp,
                                          const ulong size,
                                          const uint harmonics){

                    const ulong gid = get_global_id(0);
                    const ulong lid = get_local_id(0);
                    const ulong lsize = get_local_size(0);

                    if(gid < size){
                        tmp[lid+lsize*0] = theta[gid];
                        tmp[lid+lsize*1] = sin(tmp[lid+lsize*0]);

                        for(uint i = 0; i < harmonics*2; i+=2) {
                            FLOAT_TYPE theta_harm = tmp[lid+lsize*0]*((FLOAT_TYPE)(i+2));
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
                __kernel void diagnostic2(__global FLOAT_TYPE* x,
                                          __global FLOAT_TYPE* px,
                                          __global FLOAT_TYPE* y,
                                          __global FLOAT_TYPE* py,
                                          __global FLOAT_TYPE* gamma,
                                          __global FLOAT_TYPE* diagnostic_arr,
                                          __local FLOAT_TYPE* tmp,
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

                '''

    def __init__(self, beam_array):
        '''
            Creates a new beam from a an array of 6 columns, X rows
            representing the particles
            The columns of the array represents: theta, gamma, x, y, px, py
        '''
        assert beam_array.shape[0] == 6, "The rows must be 6"
        self.size = beam_array.shape[1]
        self.len = beam_array.shape[1]
        self.events = []

        if beam_array.dtype != cl_ftype:
            beam_array = beam_array.astype(cl_ftype)

        if not beam_array.flags.f_contiguous:
            beam_array = np.array(beam_array, order="C")

        mf = cl.mem_flags
        
        self.theta = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)
        self.gamma = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)
        self.x = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)
        self.y = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)
        self.px = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)
        self.py = cl.Buffer(cl_ctx, mf.READ_WRITE, size=self.size*cl_ftype_nbytes)

        self.events.append(cl.enqueue_copy(cl_queue, self.x, beam_array[0, :]))
        self.events.append(cl.enqueue_copy(cl_queue, self.px, beam_array[1, :]))
        self.events.append(cl.enqueue_copy(cl_queue, self.y, beam_array[2, :]))
        self.events.append(cl.enqueue_copy(cl_queue, self.py, beam_array[3, :]))
        self.events.append(cl.enqueue_copy(cl_queue, self.theta, beam_array[4, :]))
        self.events.append(cl.enqueue_copy(cl_queue, self.gamma, beam_array[5, :]))

       
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
            beam_array = np.empty((6, arr_len), dtype=cl_ftype, order="C")
            
            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[0, :], 
                                               self.x,
                                               device_offset=beam_range[0]*cl_ftype_nbytes))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[1, :], 
                                               self.px,
                                               device_offset=beam_range[0]*cl_ftype_nbytes)) 

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[2, :], 
                                               self.y,
                                               device_offset=beam_range[0]*cl_ftype_nbytes))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[3, :], 
                                               self.py,
                                               device_offset=beam_range[0]*cl_ftype_nbytes))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[4, :], 
                                               self.theta,
                                               device_offset=beam_range[0]*cl_ftype_nbytes))

            self.events.append(cl.enqueue_copy(cl_queue, 
                                               beam_array[5, :], 
                                               self.gamma,
                                               device_offset=beam_range[0]*cl_ftype_nbytes)) 

            return beam_array
        else:
            beam_array = np.empty((6, self.len), dtype=cl_ftype, order="C")
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[0, :], self.x))
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[1, :], self.px)) 
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[2, :], self.y))    
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[3, :], self.py))   
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[4, :], self.theta))   
            self.events.append(cl.enqueue_copy(cl_queue, beam_array[5, :], self.gamma)) 
        
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
            assert beam_array.shape[0] == 6, "Rows must be 6"
            new_len += beam_array.shape[1]

            if beam_array.dtype != cl_ftype:
                beam_array = beam_array.astype(cl_ftype)

            if not beam_array.flags.f_contiguous:
                beam_array = np.array(beam_array, order="C")

            beam_arrays[i] = beam_array

        self.resize(new_len)

        for beam_array in beam_arrays:
            offset = self.len*cl_ftype_nbytes
            self.len += beam_array.shape[1]
            self.events.append(cl.enqueue_copy(cl_queue, self.x, 
                                               beam_array[0, :], 
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.px, 
                                               beam_array[1, :],
                                               device_offset=offset))
            
            self.events.append(cl.enqueue_copy(cl_queue, self.y, 
                                               beam_array[2, :],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.py, 
                                               beam_array[3, :],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.theta, 
                                               beam_array[4, :],
                                               device_offset=offset))

            self.events.append(cl.enqueue_copy(cl_queue, self.gamma, 
                                               beam_array[5, :],
                                               device_offset=offset))


    def resize(self, size):
        '''
            Change the array size
        '''
        if size < self.len:
            self.len = size
        elif size > self.size:
            mf = cl.mem_flags
            x = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
            px = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
            y = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
            py = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
            theta = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
            gamma = cl.Buffer(cl_ctx, mf.READ_WRITE, size=size*cl_ftype_nbytes)
       
            self.wait()
            self.events.append(cl.enqueue_copy(cl_queue, x, self.x))
            self.events.append(cl.enqueue_copy(cl_queue, px, self.px))
            self.events.append(cl.enqueue_copy(cl_queue, y, self.y))
            self.events.append(cl.enqueue_copy(cl_queue, py, self.py))
            self.events.append(cl.enqueue_copy(cl_queue, theta, self.theta))
            self.events.append(cl.enqueue_copy(cl_queue, gamma, self.gamma))

            self.px = px
            self.x = x
            self.py = py
            self.y = y
            self.theta = theta
            self.gamma = gamma

            self.size = size

    @classmethod
    def initialize(cls):
        '''
            Compile kernels
        '''
        cls.program = cl.Program(cl_ctx, F(cls.KERNEL)).build()


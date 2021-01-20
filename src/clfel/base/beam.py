'''
    Beam module.
'''
from .clctx import cl_ctx, cl_queue, cl_ftype, cl_ftype_nbytes, F
import pyopencl.array as cl_array
import pyopencl as cl
from pyopencl.algorithm import RadixSort
from pyopencl.tools import ImmediateAllocator, MemoryPool, VectorArg, ScalarArg
from pyopencl.scan import GenericScanKernel
from copy import copy
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

        self.allocator = MemoryPool(ImmediateAllocator(cl_queue)) 
        
        self.x     = cl_array.to_device(cl_queue, beam_array[0, :], allocator=self.allocator)
        self.px    = cl_array.to_device(cl_queue, beam_array[1, :], allocator=self.allocator)
        self.y     = cl_array.to_device(cl_queue, beam_array[2, :], allocator=self.allocator)
        self.py    = cl_array.to_device(cl_queue, beam_array[3, :], allocator=self.allocator)
        self.theta = cl_array.to_device(cl_queue, beam_array[4, :], allocator=self.allocator)
        self.gamma = cl_array.to_device(cl_queue, beam_array[5, :], allocator=self.allocator)
       
    def wait(self):
        '''
            Waits for the events to be completed
        '''
        for event in self.events:
            event.wait()
        self.events = []

    def __getitem__(self, beam_range):
        '''
            Returns the beam as numpy array (see the constructor for the order)
        '''
        
        start = beam_range.start if beam_range.start != None else 0
        stop = beam_range.stop  if beam_range.stop != None else self.len 
        step = beam_range.step if beam_range.step != None else 1

        if stop > self.len:
            raise IndexError("Beam index out of range")

        if start >= self.len:
            raise IndexError("Beam index out of range")

        if stop < 0:
            stop = self.len+1+stop
        
        if start < 0:
            start = self.len+1+beam_range.start

        self.wait()
        x = self.x[start:stop:step].get(cl_queue)
        px = self.px[start:stop:step].get(cl_queue)
        y = self.y[start:stop:step].get(cl_queue)
        py = self.py[start:stop:step].get(cl_queue)
        theta = self.theta[start:stop:step].get(cl_queue)
        gamma = self.gamma[start:stop:step].get(cl_queue)

        return np.vstack((x, px, y, py, theta, gamma))

    def __len__(self):
        '''
            Returns the number of particles
        '''
        return self.len

    def merge_particles(self, beam_arrays):
        '''
            Merge new particles in the beam.
        '''
        
        old_len = self.len
        assert beam_arrays[0].shape[0] == 6, "Rows must be 6"
        concatenated_array = np.hstack(beam_arrays)
        self.resize(old_len + concatenated_array.shape[1])
        self.len = old_len + concatenated_array.shape[1]

        if concatenated_array.shape[1]:
            self.x[old_len:self.len]     = concatenated_array[0, :].astype(cl_ftype)
            self.px[old_len:self.len]    = concatenated_array[1, :].astype(cl_ftype)
            self.y[old_len:self.len]     = concatenated_array[2, :].astype(cl_ftype)
            self.py[old_len:self.len]    = concatenated_array[3, :].astype(cl_ftype)
            self.theta[old_len:self.len] = concatenated_array[4, :].astype(cl_ftype)
            self.gamma[old_len:self.len] = concatenated_array[5, :].astype(cl_ftype)

    def resize(self, size):
        '''
            Change the array size
        '''
        if size < self.len:
            self.len = size
        elif size > self.size:
            x     = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator) 
            px    = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator) 
            y     = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator) 
            py    = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator)
            theta = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator)
            gamma = cl_array.empty(cl_queue, (size,), cl_ftype, allocator=self.allocator)
       
            self.wait()

            x[:self.len]     = self.x[:self.len]
            px[:self.len]    = self.px[:self.len]
            y[:self.len]     = self.y[:self.len]
            py[:self.len]    = self.py[:self.len]
            theta[:self.len] = self.theta[:self.len]
            gamma[:self.len] = self.gamma[:self.len]

            self.px = px
            self.x = x
            self.py = py
            self.y = y
            self.theta = theta
            self.gamma = gamma
            self.size = size

    def sort_longitudinal(self, 
                          slice_len, 
                          long_shift = 0.0, 
                          cutoff_bin = -(2**31)):
        '''
            Sort the beam longitudinally, using a discretization of 'slice_len'
        '''
        (sorted_arr, 
         event) = self.longitudinal_sort_kernel(self.x,
                                                self.px,
                                                self.y,
                                                self.py,
                                                self.theta,
                                                self.gamma,
                                                cl_ftype(1.0/slice_len),
                                                **{"allocator" : self.allocator})

        self.events.append(event)
       
        self.x = sorted_arr["x"]
        self.px = sorted_arr["px"]
        self.y = sorted_arr["y"]
        self.py = sorted_arr["py"]
        self.theta = sorted_arr["theta"]
        self.gamma = sorted_arr["gamma"]
        self.size = self.len

    def sort_longitudinal_traverse(self, 
                                  slice_len, 
                                  traverse_len, 
                                  traverse_bins, 
                                  long_shift = 0.0, 
                                  cutoff_bin = -(2**31)):
        '''
            Sort the beam. The longitudinal discretization is 'slice_len',
            the traversal discretization is 'traverse_len'. The number of traversal bins
            is 'traverse bins'. 
        '''
        (sorted_arr, 
         event) = self.longitudinal_traverse_sort_kernel(self.x,
                                                         self.px,
                                                         self.y,
                                                         self.py,
                                                         self.theta,
                                                         self.gamma,
                                                         cl_ftype(1.0/slice_len),
                                                         cl_ftype(1.0/traverse_len),
                                                         np.int32(bins),
                                                        **{"allocator" : self.allocator})

        self.events.append(event)

        self.x = sorted_arr["x"]
        self.px = sorted_arr["px"]
        self.y = sorted_arr["y"]
        self.py = sorted_arr["py"]
        self.theta = sorted_arr["theta"]
        self.gamma = sorted_arr["gamma"]
        self.size = self.len

    @classmethod
    def initialize(cls):
        '''
            Compile kernels
        '''
        cls.program = cl.Program(cl_ctx, F(cls.KERNEL)).build()
        cls.longitudinal_sort_kernel = RadixSort(cl_ctx,
                                                 [VectorArg(cl_ftype, "x"), 
                                                  VectorArg(cl_ftype, "px"),
                                                  VectorArg(cl_ftype, "y"),
                                                  VectorArg(cl_ftype, "py"),
                                                  VectorArg(cl_ftype, "theta"),
                                                  VectorArg(cl_ftype, "gamma"),
                                                  ScalarArg(cl_ftype, "inv_slice_len")],
                                                 key_expr="(int) floor(theta[i]*inv_slice_len)",
                                                 sort_arg_names=["x", "px", "y", "py", "theta", "gamma"],
                                                 key_dtype=np.int32)

        class LongitudinalTraverseScanKernel(GenericScanKernel):
            '''
                Adds a preamble method for the longitudinal traverse sort
            '''
            def __init__(self, *argl, **argd):
                '''
                    Patch argd['preamble']
                '''

                sort_fun = '''
                            int sort_fun(FLOAT_TYPE x, 
                                         FLOAT_TYPE y, 
                                         FLOAT_TYPE theta, 
                                         FLOAT_TYPE inv_slice_len, 
                                         FLOAT_TYPE inv_traverse_len,
                                         int bins) {
                                         
                                         FLOAT_TYPE xnorm = 0.5 + (inv_traverse_len*x);
                                         FLOAT_TYPE ynorm = 0.5 + (inv_traverse_len*y);
                                         int xbin = (int) floor(xnorm * inv_traverse_len);
                                         int ybin = (int) floor(ynorm * inv_traverse_len);
                                         int zbin = (int) floor(theta*inv_slice_len);

                                         if ((xbin < 0) || (xbin >= bins) || (ybin < 0) || (ybin >= bins)) {
                                            xbin = 0;
                                            ybin = 0;

                                         }

                                         return xbin+bins*(ybin+bins*zbin);
                            }
                           '''
                
                new_argd = dict(argd)
                new_argd['preamble'] = F(sort_fun + new_argd['preamble'])
                super().__init__(*argl, **new_argd)
        
        cls.longitudinal_traverse_sort_kernel = RadixSort(cl_ctx,
                                                          [VectorArg(cl_ftype, "x"), 
                                                           VectorArg(cl_ftype, "px"),
                                                           VectorArg(cl_ftype, "y"),
                                                           VectorArg(cl_ftype, "py"),
                                                           VectorArg(cl_ftype, "theta"),
                                                           VectorArg(cl_ftype, "gamma"),
                                                           ScalarArg(cl_ftype, "inv_slice_len"),
                                                           ScalarArg(cl_ftype, "inv_traverse_len"),
                                                           ScalarArg(np.int32, "bins")],
                                                           key_expr="sort_fun(x[i],y[i],theta[i], inv_slice_len, inv_traverse_len, bins)",
                                                           sort_arg_names=["x", "px", "y", "py", "theta", "gamma"],
                                                           scan_kernel = LongitudinalTraverseScanKernel,
                                                           key_dtype=np.int32)


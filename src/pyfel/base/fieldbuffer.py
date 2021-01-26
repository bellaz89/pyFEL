'''
    The field buffer structure
'''

import numpy as np
import pyopencl as cl
from .clctx import cl_queue, cl_ctx, cl_ctype

class FieldBuffer(object):
    '''
        Holds a 3 dimensional grid with the field.
        Can be implemented either with texture of
        global device memory
    ''' 
    def __init__(self, shape=None, data=None, fitype="global"):
        '''
            Initialize the buffer.
            At least shape or data must be defined.
            shape must be a three element tuple. If only shape is defined,
            A new zeroed buffer with shape dimensions is created
            data is a numpy array of complex values. If only data is passed, it must have
            a shape len equal to 3.
            If both shape and data are passed, data gets a new shape

            fitype is either "global" or "texture"
        '''
        
        host_data = None
        self.fitype = fitype
        self.order = "F" if fitype == "global" else "C"

        if not shape and not isinstance(data, np.ndarray):
            raise RuntimeError("At least shape or data must be defined")
        elif shape and not isinstance(data, np.ndarray):
            assert len(shape) == 3, "shape must be a three elements long tuple"
            self.shape = shape
            host_data = np.zeros(self.shape, dtype=cl_ctype, order=self.order)
        elif not shape and isinstance(data, np.ndarray):
            assert len(data.shape) == 3, "shape of data must be a three elements long tuple"
            self.shape = data.shape
            host_data = np.array(data, dtype=cl_ctype, order=self.order)
        else:
            assert len(shape) == 3, "shape must be a three elements long tuple"
            self.shape = shape
            host_data = np.array(data, dtype=cl_ctype, order=self.order).reshape(shape)

        self.data, self.evs = self.get_image_from_shape(host_data, fitype)      


    @staticmethod
    def get_image_from_shape(host_data, fitype):
        '''
            Upload data to the device
        '''
        data = None
        evs = []

        if fitype == "global":
            mf = cl.mem_flags
            data = cl.Buffer(cl_ctx, 
                            mf.READ_WRITE | mf.COPY_HOST_PTR, 
                            hostbuf=host_data)
        else:
            channel_type = (cl.channel_type.FLOAT 
                            if cl_ctype == np.complex128 
                            else cl.channel_ctype.UNSIGNED_INT16)

            image_format = cl.ImageFormat(cl.channel_order.RGBA, channel_type)
            mf = cl.mem_flags
            data = cl.Image(cl_ctx, mf.READ_WRITE, image_format, host_data.shape)
            evs.append(cl.enqueue_copy(cl_queue, 
                                       data, 
                                       host_data,
                                       origin=(0,0,0),
                                       region=host_data.shape))

        return data, evs

    def get(self):
        '''
            Get data from from the buffer
        '''
        self.wait()
        host_data = np.empty(self.shape, dtype=cl_ctype, order=self.order)

        if self.fitype == "global":
            self.evs.append(cl.enqueue_copy(cl_queue, 
                                            host_data, 
                                            self.data)) 
        else:
            self.evs.append(cl.enqueue_copy(cl_queue, 
                                            host_data, 
                                            self.data, 
                                            origin=(0,0,0),
                                            region=self.shape))

        return host_data

    def wait(self):
        '''
            Waits until all the events associated to the buffer are consumed 
        '''
        for ev in self.evs:
            ev.wait()

        self.evs = []


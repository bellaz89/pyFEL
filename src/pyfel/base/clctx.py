'''
    Provides the OpenCL context and queue
'''
import numpy as np
import pyopencl as cl
from config_path import ConfigPath
import toml

from pyfel.util.init_class import init_class

VENDOR = "pyfel"
NAME = "pyfel"
EXTENSION = ".toml"
DEFAULT_WAVEFRONT = 32
DEFAULT_FTYPE = np.float64

@init_class
class CLCtx:
    '''
        Class that keeps the OpenCL context and queue
    '''

    @classmethod
    def initialize(cls):
        conf_path = ConfigPath(VENDOR, NAME, EXTENSION)
        path = conf_path.readFilePath()
        if path:
            conf = toml.load(path)
            cl_platform = conf["cl_platform"]
            cl_device = conf["cl_device"]

            cls.ftype = np.float64 if conf["cl_ftype"] == 64 else np.float32 
            cls.wavefront = conf["cl_wavefront"]
            cls.device = cl.get_platforms()[cl_platform].get_devices()[cl_device]
            cls.ctx = cl.Context([cls.device])
        else:
            cls.wavefront = DEFAULT_WAVEFRONT
            cls.ctx = cl.create_some_context()
            cls.device = cls.ctx.devices[0]
            cls.ftype = DEFAULT_FTYPE 

            conf = dict()
            conf["cl_platform"] = cl.get_platforms().index(cls.device.platform)
            conf["cl_device"] = cls.device.platform.get_devices().index(cls.device)
            conf["cl_wavefront"] = cls.wavefront
            conf["cl_ftype"] = 64 if cls.ftype == np.float64 else 32
            toml.dump(conf, open(conf_path.saveFilePath(), "w"))

        cls.ftype_nbytes = cls.ftype(0).nbytes
        cls.queue = cl.CommandQueue(cls.ctx)

cl_device = CLCtx.device
cl_ctx = CLCtx.ctx
cl_queue = CLCtx.queue
cl_wavefront = CLCtx.wavefront
cl_ftype = CLCtx.ftype
cl_ftype_nbytes = CLCtx.ftype_nbytes

def F(source):
    '''
        substitute FLOAT_TYPE in the source with a concrete OpenCL floating point type.
    '''
    if cl_ftype == np.float64:
        return source.replace("FLOAT_TYPE", "double")
    else:
        return source.replace("FLOAT_TYPE", "float")


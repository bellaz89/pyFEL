'''
    Provides the OpenCL context and queue
'''
import pyopencl as cl
from config_path import ConfigPath
import toml

VENDOR = "clfel"
NAME = "clfel"
EXTENSION = ".toml"
DEFAULT_WAVEFRONT = 32

class CLCtx:
    '''
        Class that keeps the OpenCL context and queue
    '''

    initialized = False

    @classmethod
    def initialize(cls):
        if not cls.initialized:
            conf_path = ConfigPath(VENDOR, NAME, EXTENSION)
            path = conf_path.readFilePath()
            if path:
                conf = toml.load(path)
                cl_platform = conf["cl_platform"]
                cl_device = conf["cl_device"]

                cls.wavefront = conf["cl_wavefront"]
                cls.device = cl.get_platforms()[cl_platform].get_devices()[cl_device]
                cls.ctx = cl.Context([cls.device])
            else:
                cls.wavefront = DEFAULT_WAVEFRONT
                cls.ctx = cl.create_some_context()
                cls.device = cls.ctx.devices[0]

                conf = dict()
                conf["cl_platform"] = cl.get_platforms().index(cls.device.platform)
                conf["cl_device"] = cls.device.platform.get_devices().index(cls.device)
                conf["cl_wavefront"] = cls.wavefront
                toml.dump(conf, open(conf_path.saveFilePath(), "w"))

            cls.queue = cl.CommandQueue(cls.ctx)
            cls.initialized = True

CLCtx.initialize()

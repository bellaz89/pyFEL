import numpy as np
from pyfel.base.clctx import cl_ftype

FLOAT64_TOLERANCE = {"rtol" : 1e-05, "atol" : 1e-08}
FLOAT32_TOLERANCE = {"rtol" : 1e-03, "atol" :1e-05}

cl_tol =  FLOAT64_TOLERANCE if cl_ftype == np.float64 else FLOAT32_TOLERANCE

 

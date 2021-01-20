'''
    Skippable random generator
'''

import numpy as np
from numpy.random import Generator, PCG64

class Random(object):
    '''
        Random generator based on PCG64
    '''
    def __init__(self, seed):
        '''
            Init the sequence with a random seed
        '''
        self.seed = seed
        self.idx = 1
        self.set_idx(0)

    def set_idx(self, new_idx):
        '''
           Skip the sequence to new_idx 
        '''
        if self.idx > new_idx:
            self.pcg_instance = PCG64(self.seed)
            self.generator = Generator(self.pcg_instance)
            self.idx = 0

        self.pcg_instance.advance(new_idx - self.idx)
        self.idx = new_idx

    def get_value(self):
        '''
            Get an element from the sequence
        '''
        self.idx += 1
        return self.generator.random(dtype=np.float64)

    def get_array(self, n):
        '''
            Get a numpy 1D array with length 'n'
        '''
        self.idx += n
        return self.generator.random(size=n, dtype=np.float64)

    def get_normal_array(self, n):
        '''
            Get a normal distribuited 1D array with length 'n'
        '''
        self.idx += n
        return self.generator.normal(size=n) 


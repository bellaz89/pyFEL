'''
    Transform the beam
'''

import abc

class BeamTransform(metaclass=abc.ABCMeta):
    '''
        Class representing an abstract beam transformation
    '''
    @abc.abstractmethod
    def transform(self, beam):
        '''
            Apply a transform to the beam
        '''
        pass

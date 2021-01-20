'''
    Decorator for class initialization
'''

def init_class(cls):
    '''
        Call cls.initialize() and return the class
    '''
    cls.initialize()
    return cls

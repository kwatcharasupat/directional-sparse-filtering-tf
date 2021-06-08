import numpy as np

def nextpow2(x):
    p2 = np.ceil(np.log2(x))
    return int(np.power(2., p2))   
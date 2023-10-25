
import numpy as np

def coinsizes(lower, upper):
    """
    returns an array of numbers corresponding to approximately exponential 'coin-size' numbers:
    1,2,5,10,20,50,100,200,500,...
    lower, upper are inclusive.
    """
    lower_log = int(np.floor(np.log10(lower)))
    upper_log = int(np.floor(np.log10(upper)))
    result = []
    for i in range(lower_log, upper_log + 1):
        base = int(10 ** i)
        result += [factor * base for factor in [1,2,5] if factor * base >= lower and factor * base <= upper]
    return result
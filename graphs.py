import numpy as np
from numpy.random import rand


class ERGraph:
    def __init__(self, arms, r):
        self.arms = arms
        self.r = r

    def getObserved(self, pulled):
        observed = np.zeros(self.arms)
        observed[pulled] = 1
        return np.minimum(observed + (rand(self.arms)<self.r),1)
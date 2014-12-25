import numpy as np

class Losses:
    def __init__(self, arms):
        self.arms = arms
        self.minExpected = min(map(lambda x : x.mean, arms))

    def getLosses(self):
        losses = np.zeros(len(self.arms))
        for i, arm in enumerate(self.arms):
            losses[i] = arm.getLoss()
        return losses

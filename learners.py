import numpy as np
from numpy.random import multinomial

class BaseLearner(object):
    def __init__(self, arms):
        self.arms = arms

    def start(self):
        self.weights = np.ones(self.arms)
        self.probas = np.ones(self.arms)

    def getArm(self):
        raise(Exception('Not Implemented'))
    
    def observe(self, observed, losses):
        raise(Exception('Not Implemented'))

class PullOneArm(BaseLearner):
    def __init__(self, arm, **kwargs):
        super(PullOneArm, self).__init__(**kwargs)
        self.arm = arm

    def getArm(self):
        return self.arm

    def observe(self, observed, losses):
        return

class EXP3(BaseLearner):
    def __init__(self, gamma, eta, **kwargs):
        super(EXP3, self).__init__(**kwargs)
        self.gamma = gamma
        self.eta = eta

    def getArm(self):
        self.probas = (1-self.gamma)*self.weights / sum(self.weights)
        self.probas += np.ones(self.arms)*(self.gamma/self.arms)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        return self.chosen

    def observe(self, observed, losses):
        r = losses[self.chosen]
        self.weights[self.chosen] *= np.exp(self.eta*r/self.probas[self.chosen])
        return
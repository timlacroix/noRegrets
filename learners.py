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

    def getArm(self, t):
        self.probas = (1-self.gamma)*self.weights / sum(self.weights)
        self.probas += np.ones(self.arms)*(self.gamma/self.arms)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        return self.chosen

    def observe(self, observed, losses, t):
        r = losses[self.chosen]
        self.weights[self.chosen] *= np.exp(self.eta*r/self.probas[self.chosen])
        return


class DuplEXP3(BaseLearner):
    def start(self, **kwargs):
        super(DuplEXP3, self).start(**kwargs)
        self.L = [np.zeros(self.arms), np.zeros(self.arms)]
        self.previous_O = np.zeros(self.arms)
        self.previous_O[self.arms-1] = 1
        self.previous_proba = np.zeros((1,self.arms))
        self.previous_loss_estimates = np.zeros((1,self.arms))

    def getArm(self, t):
        # previous_proba has dimension t-1 * N
        previous_proba = self.previous_proba[1:, :]
        # previous_loss_estimates the same
        previous_loss_estimates = self.previous_loss_estimates[1:, :]
        tau = np.arange(1, t) % 2 == t % 2
        tmp = np.sum(np.sum(previous_proba * previous_loss_estimates**2, 1)[tau])
        eta = np.sqrt(np.log(self.arms) / (self.arms**2 + tmp))

        self.weights = np.exp(-eta * self.L[t % 2]) / self.arms
        self.probas = self.weights / sum(self.weights)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        self.previous_proba = np.vstack((self.previous_proba, self.probas))
        return self.chosen

    def observe(self, observed, losses, t):
        M = np.argmax(self.previous_O)
        K = np.random.geometric(self.probas, self.arms)
        G = np.minimum(K, M)
        lhat = losses * observed * G
        self.L[t % 2] += lhat
        self.previous_O = np.append(np.delete(observed, self.chosen), 1)
        self.previous_loss_estimates = np.vstack((self.previous_loss_estimates, lhat))
        return lhat





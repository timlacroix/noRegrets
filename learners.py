import numpy as np
from numpy.random import multinomial, randint


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
        self.weights[self.chosen] *= np.exp(-self.eta*r/self.probas[self.chosen])
        return


class DuplEXP3(BaseLearner):
    def start(self, **kwargs):
        super(DuplEXP3, self).start(**kwargs)
        self.L = [np.zeros(self.arms), np.zeros(self.arms)]
        self.previous_O = np.zeros(self.arms)
        self.previous_O[-1] = 1
        
        self.previous_proba = np.ones((2,self.arms))/self.arms
        self.previous_loss_estimates = np.zeros((2,self.arms))

    def getArm(self, t):
        tau = np.arange(0,t) % 2 == t % 2
        tmp = np.sum(self.previous_proba[tau] * self.previous_loss_estimates[tau]**2)
        eta = np.sqrt(np.log(self.arms) / (self.arms**2 + tmp))

        self.weights = np.exp(-eta * self.L[t % 2]) / self.arms
        self.probas = self.weights / sum(self.weights)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        self.previous_proba = np.vstack((self.previous_proba, self.probas))
        return self.chosen

    def observe(self, observed, losses, t):
        M = np.argmax(self.previous_O)
        K = np.random.geometric(self.probas)
        G = np.minimum(K, M)
        lhat = losses * observed * G
        self.L[t % 2] += lhat
        self.previous_O = np.append(np.delete(observed, self.chosen), 1)
        self.previous_loss_estimates = np.vstack((self.previous_loss_estimates, lhat))
        return lhat


class EstimatingR(BaseLearner):
    def __init__(self, k, C, graph, **kwargs):
        super(EstimatingR, self).__init__(**kwargs)
        self.k = k
        self.C = C
        self.graph = graph

    def getEstimate(self):
        j = 0
        c = 0
        M = np.zeros(self.k)
        for t in range(self.C):
            I = randint(self.arms)
            Obs = self.graph.getObserved(I)
            c += np.sum(Obs[list(range(self.arms).pop(I))])
        if c/(self.C*(self.arms - 1)) <= 3/(2*self.arms):
            return 0
        else:
            for t in range(self.C, T):
                I = randint(self.arms)
                Obs = self.graph.getObserved(I)
                for i in range(self.arms):
                    M[j] += int(i != I)
                    j += Obs(i) * int(i != I)
                    if j == k:
                        return (np.max(M) + 1) ** (-1)
                    else:
                        M[j] = 0

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
        self.weights *= np.exp(-self.eta*losses*observed/self.probas)
        return


class DuplEXP3(BaseLearner):
    def start(self, **kwargs):
        super(DuplEXP3, self).start(**kwargs)
        self.L = [np.zeros(self.arms), np.zeros(self.arms)]
        self.previous_O = np.zeros(self.arms)
        self.previous_O[-1] = 1
        self.previous_proba = np.ones((2, self.arms))/self.arms
        self.previous_loss_estimates = np.zeros((2, self.arms))
        self.Mocc = list()

    def getArm(self, t):
        tau = np.arange(0, t) % 2 == t % 2
        tmp = np.sum(self.previous_proba[tau] * self.previous_loss_estimates[tau]**2)
        eta = np.sqrt(np.log(self.arms) / (self.arms**2 + tmp))

        self.weights = np.exp(-eta * self.L[t % 2]) / self.arms
        self.probas = self.weights / sum(self.weights)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        self.previous_proba = np.vstack((self.previous_proba, self.probas))
        return self.chosen

    def observe(self, observed, losses, t):
        M = np.argmax(self.previous_O) + 1
        self.Mocc.append(M)
        K = np.random.geometric(self.probas)
        G = np.minimum(K, M)
        lhat = losses * observed * G
        self.L[t % 2] += lhat
        self.previous_O = np.append(np.delete(observed, self.chosen), 1)
        self.previous_loss_estimates = np.vstack((self.previous_loss_estimates, lhat))
        return lhat


class BAEXP3(BaseLearner):
    def __init__(self, gamma, eta, **kwargs):
        super(BAEXP3, self).__init__(**kwargs)
        self.gamma = gamma
        self.eta = eta
        self.setK(**kwargs)

    def setK(self, arms):
        n_pk = sum([x**(-3) for x in range(1, arms)])
        self.K = 2*sum([(x**(-2)/n_pk) for x in range(1, arms)])/(arms-1)

    def getArm(self, t):
        self.probas = (1-self.gamma)*self.weights / sum(self.weights)
        self.probas += np.ones(self.arms)*(self.gamma/self.arms)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        return self.chosen

    def observe(self, observed, losses, t):
        estimated_loss = observed*losses/(self.probas + self.K*(1-self.probas))
        self.weights *= np.exp(-self.eta*estimated_loss)
        return


class GeneralDuplExp3(BaseLearner):
    def __init__(self, A, **kwargs):
        super(GeneralDuplExp3, self).__init__(**kwargs)
        self.A = A

    def start(self, **kwargs):
        super(GeneralDuplExp3, self).start(**kwargs)
        self.weights /= self.arms
        self.probas = self.weights / sum(self.weights)
        self.L = [np.zeros(self.arms), np.zeros(self.arms)]
        self.current_O = np.zeros((self.A, self.arms))
        self.current_losses = np.zeros((self.A, self.arms))
        self.previous_O_seconde = np.zeros((self.A * (self.arms - 1) + 1))
        self.previous_O_seconde[-1] = 1
        self.Mj = np.argmax(self.previous_O_seconde) + 1
        self.previous_proba = np.ones((2, self.arms))/self.arms
        self.previous_loss_estimates = np.zeros((2, self.arms))
        self.K = np.random.geometric(self.probas)
        self.G = np.minimum(self.K, self.Mj)
        self.eta = np.sqrt(np.log(self.arms) / ((self.arms * self.A)**2))
        self.Mocc = list()

    def end_episode_updates(self, j):
        lhat = self.G * np.sum(self.current_losses * self.current_O, 0)
        # we have set the Oji to be Oti, but weird
        self.L[(j - 1) % 2] += lhat
        self.previous_loss_estimates = np.vstack((self.previous_loss_estimates, lhat))

        tau = np.arange(0, j) % 2 == j % 2
        tm = np.sum(self.previous_proba[tau] * self.previous_loss_estimates[tau] ** 2)
        self.eta = np.sqrt(np.log(self.arms) / ((self.arms * self.A)**2 + tm))
        self.weights = np.exp(-self.eta * self.L[j % 2]) / self.arms
        self.probas = self.weights / sum(self.weights)
        self.previous_proba = np.vstack((self.previous_proba, self.probas))
        self.Mj = np.argmax(self.previous_O_seconde) + 1
        self.Mocc.append(self.Mj)
        self.previous_O_seconde = np.zeros((self.A * (self.arms - 1) + 1))
        self.previous_O_seconde[-1] = 1
        self.K = np.random.geometric(self.probas)
        self.G = np.minimum(self.K, self.Mj)

    def getArm(self, t):
        j = int(t/self.A) + 1
        if t == (j-1) * self.A:
            self.end_episode_updates(j)
        self.chosen = np.where(multinomial(1, self.probas))[0][0]
        return self.chosen

    def observe(self, observed, losses, t):
        estimated_loss = observed*losses*self.G
        self.weights *= np.exp(-self.eta*estimated_loss) / self.arms
        subt = t % self.A
        self.current_losses[subt, :] = losses
        self.current_O[subt, :] = observed
        self.previous_O_seconde[subt*(self.arms - 1): (subt+1) * (self.arms - 1)] = \
            np.delete(observed, self.chosen)
        return

import numpy as np
import pylab

from losses import *
from arms import *
from learners import *
from graphs import *

def applyLearner(learner, graph, losses, horizon=100, repeat=1):
    regrets = np.zeros((repeat, horizon))
    for repetition in range(repeat):
        print(repetition)
        learner.start()
        for t in range(horizon):
            current_losses = losses.getLosses()
            arm = learner.getArm(t)
            observed = graph.getObserved(arm)

            learner.observe(observed, current_losses, t)
            regrets[repetition, t] = losses.maxExpected-current_losses[arm]

    return np.mean(np.cumsum(regrets, axis=1), axis=0)



learner = EXP3(gamma=0.01, eta=0.01, arms=3)
graph = ERGraph(arms=3, r=0.5)
losses = Losses([Bernoulli(0.2), Bernoulli(0.3), Bernoulli(0.4)])

regrets = applyLearner(learner, graph, losses, horizon=1000, repeat=100)
pylab.plot(regrets,'r-')
pylab.show()

learner = DuplEXP3(arms=5)
graph = ERGraph(arms=5, r=0.5)
losses = Losses([Bernoulli(0.2), Bernoulli(0.3), Bernoulli(0.4), Bernoulli(0.6), Bernoulli(0.8)])
# il faut r<logT/2N, donc là j'ai mis un T tout petit...
regrets = applyLearner(learner, graph, losses, horizon=10, repeat=10)
pylab.plot(regrets,'m-')
pylab.show()
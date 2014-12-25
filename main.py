import numpy as np
import pylab

from losses import *
from arms import *
from learners import *
from graphs import *

def applyLearner(learner, graph, losses, horizon=100, repeat=1):
    regrets = np.zeros((repeat, horizon))
    for repetition in range(repeat):
        learner.start()
        for t in range(horizon):
            current_losses = losses.getLosses()
            arm = learner.getArm(t)
            observed = graph.getObserved(arm)

            learner.observe(observed, current_losses, t)
            regrets[repetition, t] = current_losses[arm]-losses.minExpected

    return np.mean(np.cumsum(regrets, axis=1), axis=0)

n_arms=20
eps=0.1

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

graph = ERGraph(arms=n_arms, r=0.5)


learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=5000, repeat=100
)
pylab.plot(regrets,'r-',label='EXP3')
learner = DuplEXP3(arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=5000, repeat=100
)
pylab.plot(regrets,'b-',label='DuplEXP3')
pylab.show()
import numpy as np
import pylab

from joblib import Parallel, delayed

from losses import *
from arms import *
from learners import *
from graphs import *

def do_run(learner, graph, losses, horizon=100):
    regrets = np.zeros(horizon)
    learner.start()
    for t in range(horizon):
        current_losses = losses.getLosses()
        arm = learner.getArm(t)
        observed = graph.getObserved(arm)
        learner.observe(observed, current_losses, t)
        regrets[t] = current_losses[arm]-losses.minExpected
    return regrets

def applyLearner(learner, graph, losses, horizon=100, repeat=4, n_jobs=4):
    regrets = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(do_run)(learner, graph, losses, horizon)
        for i in range(repeat)
    )
    return np.mean(np.cumsum(regrets, axis=1), axis=0)

n_arms=50
eps=0.1

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

graph = BAGraph(arms=n_arms, m=10, m0=20)

learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=1000, repeat=20, n_jobs=6
)
pylab.plot(regrets,'r-',label='EXP3')

learner = BAEXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=1000, repeat=20, n_jobs=6
)
pylab.plot(regrets,'k-',label='BAEXP3')

learner = DuplEXP3(arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=1000, repeat=20, n_jobs=6
)
pylab.plot(regrets,'b-',label='DuplEXP3')
pylab.show()
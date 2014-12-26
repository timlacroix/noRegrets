import numpy as np
import matplotlib.pyplot as plt

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

n_arms = 20
eps = 0.1
n_iterations = 5000
r = 0.5

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

graph = ERGraph(arms=n_arms, r=r)


learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=100
)
plt.figure(1)
plt.plot(regrets, 'r-', label='EXP3', linewidth=2)
learner = DuplEXP3(arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=100
)
plt.plot(regrets, 'b-', label='DuplEXP3', linewidth=2)
upper_bound = 4 * np.sqrt((n_iterations / r + n_arms**2) * np.log(n_arms)) + \
    np.sqrt(n_iterations)
plt.plot(upper_bound * np.ones(n_iterations), label='Upper Bound', linewidth=2,
         color='purple', linestyle="-")
plt.legend(loc=2, fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('cumulated regret', fontsize=20)

plt.savefig('dupl_big_r.pdf')



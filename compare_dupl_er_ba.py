import numpy as np
import matplotlib.pyplot as plt

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
    if hasattr(learner, 'Mocc'):
        dupl_r_estimate = 1/np.mean(np.array(learner.Mocc[1:]))
    else:
        dupl_r_estimate = 0
    return regrets


def applyLearner(do_run_function, learner, graph, losses, horizon=100,
                 repeat=4, n_jobs=4):
    regrets = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(do_run_function)(learner, graph, losses, horizon)
        for i in range(repeat)
    )
    # outputs = np.array(outputs)
    # regrets, dupl_r_estimate = zip(*outputs)
    return np.mean(np.cumsum(regrets, axis=1), axis=0)


def estimateR(graph, T, losses):
    k = int(np.e * np.log(T) / 2) + 1
    C = int(2 * np.log(T) / graph.arms) + 1
    j = 0
    c = 0
    M = np.zeros(k)
    regret = []
    for t in range(C):
        I = randint(graph.arms)
        current_losses = losses.getLosses()
        regret.append(current_losses[I]-losses.minExpected)
        Obs = graph.getObserved(I)
        c += np.sum(Obs) - 1
    if c/(C*(graph.arms - 1)) <= 3/(2*graph.arms):
        return 0, t+1, regret
    else:
        for t in range(C, T):
            I = randint(graph.arms)
            current_losses = losses.getLosses()
            regret.append(current_losses[I]-losses.minExpected)
            Obs = graph.getObserved(I)
            for i in range(graph.arms):
                M[j] += int(i != I)
                j += Obs[i] * int(i != I)
                if j == k:
                    return 1/(np.max(M) + 1), t+1, regret


def do_run_dupl(learner,graph, losses, horizon=100):
    r_, it, regret = estimateR(graph, horizon, losses)
    regrets = np.zeros(horizon)
    regrets[0:it] = regret
    if r_ < 1e-12:  # we use vanilla EXP3
        eta = np.sqrt(2 * np.log(graph.arms) / (horizon * graph.arms))
        learner = EXP3(gamma=0.01, eta=eta, arms=graph.arms)
        learner.start()
        regrets[it:horizon] = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    else:
        A = int(np.log(horizon-it)/(graph.arms * r_)) + 1
        learner = GeneralDuplExp3(arms=graph.arms, A=A)
        learner.start()
        regrets[it:horizon] = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    return regrets


# BA !
n_arms = 75
n_jobs = 6
eps = 0.1
n_iterations = 15000
rep = 24

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

## DuplEXP3 on BA Graph
graph = BAGraph(arms=n_arms, m=1, m0=2, r=1)
rr_ba = applyLearner(
    do_run_dupl, None, graph, losses,
    horizon=n_iterations, repeat=rep, n_jobs=n_jobs
)

## DuplEXP3 on ER Graph
r = sum(sum(graph.adjacency))/(n_arms)*(n_arms-1)

graph = ERGraph(arms=n_arms, r=r)
rr_er = applyLearner(
    do_run_dupl, None, graph, losses,
    horizon=n_iterations, repeat=rep, n_jobs=n_jobs
)

## Plot
plt.figure(2)
plt.plot(rr_ba, 'r-', label='DuplEXP3-BA', linewidth=2)
plt.plot(rr_er, 'b-', label='DuplEXP3-ER', linewidth=2)

plt.legend(loc=2, fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('cumulated regret', fontsize=20)
plt.savefig(str(n_arms) + 'compare_dupl_er_ba.pdf')
plt.close()


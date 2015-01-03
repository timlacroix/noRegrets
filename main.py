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
    return regrets, dupl_r_estimate


def applyLearner(do_run_function, learner, graph, losses, horizon=100,
                 repeat=4, n_jobs=4):
    outputs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(do_run_function)(learner, graph, losses, horizon)
        for i in range(repeat)
    )
    outputs = np.array(outputs)
    regrets, dupl_r_estimate = zip(*outputs)
    return np.mean(np.cumsum(regrets, axis=1), axis=0), np.mean(dupl_r_estimate)


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


def do_run_dupl(learner, graph, losses, horizon=100):
    r_, it, regret = estimateR(graph, horizon, losses)
    regrets = np.zeros(horizon)
    regrets[0:it] = regret
    if r_ < 1e-12:  # we use vanilla EXP3
        eta = np.sqrt(2 * np.log(graph.arms) / (horizon * graph.arms))
        learner = EXP3(gamma=0.01, eta=eta, arms=graph.arms)
        learner.start()
        regrets[it:horizon], dupl_r_estimate = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    else:
        A = int(np.log(horizon-it)/(graph.arms * r_)) + 1
        learner = GeneralDuplExp3(arms=graph.arms, A=A)
        learner.start()
        regrets[it:horizon], dupl_r_estimate = do_run(learner, graph, losses,
                                                      horizon=horizon-it)
    return regrets, dupl_r_estimate


def update_online(old_mean, new_value, n, old_M2):
    delta = new_value - old_mean
    new_mean = old_mean + delta / n
    new_M2 = old_M2 + delta**2
    return new_mean, new_M2


n_arms = 30
eps = 0.1
n_iterations = 1000
r = 0.5
rep = 100

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])
graph = ERGraph(arms=n_arms, r=r)
learner = BAEXP3(gamma=0.01, eta=0.01, arms=n_arms)
global_reg = np.zeros(n_iterations)
global_r_mean = 0
global_M2 = 0

for i in range(1):
    reg, r_mean = applyLearner(do_run_dupl, learner, graph, losses,
                               horizon=n_iterations, repeat=25, n_jobs=1)
    global_reg, useless = update_online(global_reg, reg, i+1, global_reg)
    global_r_mean, global_M2 = update_online(global_r_mean, r_mean, i+1, global_M2)
    print(global_r_mean, np.sqrt(global_M2/(i+1)))   


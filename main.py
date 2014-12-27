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
    return regrets


def applyLearner(learner, graph, losses, horizon=100, repeat=4, n_jobs=1):
    regrets = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(do_run)(learner, graph, losses, horizon)
        for i in range(repeat)
    )
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


def complete_algorithm(graph, losses, horizon=100, repeat=1):
    regrets = np.zeros((repeat, horizon))
    for repetition in range(repeat):
        r_, it, regret = estimateR(graph, horizon, losses)
        regrets[repetition, 0:it] = regret
        if r_ < 1e-12:  # we use vanilla EXP3
            eta = np.sqrt(2 * np.log(graph.arms) / (horizon * graph.arms))
            learner = EXP3(gamma=0.01, eta=eta, arms=graph.arms)
            learner.start()
            regrets[repetition, it:horizon] = do_run(learner, graph, losses,
                                                     horizon=horizon-it)
        else:
            A = int(np.log(horizon-it)/(graph.arms * r_)) + 1
            learner = GeneralDuplExp3(arms=graph.arms, A=A)
            learner.start()
            regrets[repetition, it:horizon] = do_run(learner, graph, losses,
                                                     horizon=horizon-it)
            print(learner.weights)
    return np.mean(np.cumsum(regrets, axis=1), axis=0)


n_arms = 30
eps = 0.1

n_iterations = 10
n_repeat = 2
n_jobs = 1

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

graph = BAGraph(arms=n_arms, m=10, m0=20)


## EXP3
learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=n_repeat, n_jobs=n_jobs
)
plt.figure(1)
plt.plot(regrets, 'r-', label='EXP3', linewidth=2)


## BAEXP3
learner = BAEXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=n_repeat, n_jobs=n_jobs
)
plt.plot(regrets, 'k-', label='BAEXP3', linewidth=2)

## DUPLEXP3
learner = DuplEXP3(arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=n_repeat, n_jobs=n_jobs
)
plt.plot(regrets, 'b-', label='DuplEXP3', linewidth=2)


# upper_bound = 4 * np.sqrt((n_iterations / r + n_arms**2) * np.log(n_arms)) + \
#     np.sqrt(n_iterations)
# plt.plot(upper_bound * np.ones(n_iterations), label='Upper Bound', linewidth=2,
#          color='purple', linestyle="-")
plt.legend(loc=2, fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('cumulated regret', fontsize=20)

plt.savefig('dupl_big_r.pdf')

## test of dupl exp 3
n_arms = 20
eps = 0.1
n_iterations = 5000
r = 0.8
rep=100

losses = Losses([Bernoulli(x) for x in [0.5] + [0.5+eps]*(n_arms-1)])

graph = ERGraph(arms=n_arms, r=r)

rr = complete_algorithm(graph, losses, horizon=n_iterations, repeat=rep)
learner = EXP3(gamma=0.01, eta=0.01, arms=n_arms)
regrets = applyLearner(
    learner, graph, losses,
    horizon=n_iterations, repeat=rep
)
plt.figure(2)
plt.plot(regrets, 'r-', label='EXP3', linewidth=2)

plt.plot(rr, 'b-', label='DuplEXP3', linewidth=2)

plt.legend(loc=2, fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('cumulated regret', fontsize=20)
plt.savefig('complete_dupl_big_r' + str(r) + '.pdf')

